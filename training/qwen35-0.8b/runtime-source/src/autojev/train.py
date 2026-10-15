import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import cast

import torch

from autojev.distributed import DistributedTraining
from autojev.evaluate import (
    Metrics,
    Prediction,
    calibration_ok,
    evaluate_logits,
    fit_temperature,
    hard_label,
    label_index,
    metrics,
    options,
    read_predictions,
    read_rows,
    selection_key,
    validate_coverage,
    write_json,
)
from autojev.events import record
from autojev.loss import decision_loss
from autojev.model import DecisionModel
from autojev.optim import CPUOffloadAdamW
from autojev.sft_pipeline import Receipt, ids_sha256, load_schedule, path_at
from autojev.types import Example, JSONValue


class Arguments(argparse.Namespace):
    train: str
    schedule: str | None
    development: str
    temperature: str
    reference: str
    public: str | None
    guard_data: str | None
    guard_reference: str | None
    quick_development: str | None
    quick_guard_data: str | None
    quick_temperature: str | None
    quick_eval_every: int
    run: str
    output: str
    base_model: str
    revision: str
    cache_dir: str | None
    epochs: int
    batch_size: int
    effective_batch_size: int
    token_budget: int
    max_length: int
    lr: float
    warmup_ratio: float
    weight_decay: float
    brier_weight: float
    save_every: int | None
    save_final_only: bool
    seed: int
    eval_every: int
    public_eval_every: int
    resume_every: int
    stop_after: int | None
    resume: str | None
    initial_checkpoint: str | None
    cpu_threads: int
    attention_mode: str
    pooling: str


def digest(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def initial_identity(path: str, base_model: str, revision: str) -> dict[str, JSONValue]:
    directory = Path(path).resolve()
    saved = cast(dict[str, JSONValue], json.loads((directory / "decision_config.json").read_text()))
    if saved.get("format_version") != 1 or saved.get("base_model") != base_model or saved.get("revision") != revision:
        raise ValueError("Initial checkpoint format/base model/revision differs from the requested model")
    temperature = saved.get("temperature")
    if (
        not isinstance(temperature, int | float)
        or isinstance(temperature, bool)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("Initial checkpoint temperature must be positive and finite")
    index = cast(dict[str, JSONValue], json.loads((directory / "model.safetensors.index.json").read_text()))
    weights = index.get("weight_map")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Initial checkpoint must have an indexed set of backbone weights")
    names = {
        "config.json",
        "decision_config.json",
        "model.safetensors.index.json",
        "readout.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "chat_template.jinja",
    }
    for name in weights.values():
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("Checkpoint shard names must be local safetensors filenames")
        names.add(name)
    hashes: dict[str, JSONValue] = {name: digest(directory / name) for name in sorted(names)}
    result: dict[str, JSONValue] = {
        "path": str(directory),
        "base_model": base_model,
        "revision": revision,
        "temperature": temperature,
        "decision_config": saved,
        "files_sha256": hashes,
        "artifact_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    release_manifest = directory / "release-manifest.json"
    if release_manifest.exists():
        result["release_manifest_sha256"] = digest(release_manifest)
    return result


def append(path: Path, value: object) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def learning_rate_factor(step: int, total_steps: int, warmup_ratio: float) -> float:
    warmup = max(1, int(warmup_ratio * total_steps))
    if step <= warmup:
        return step / warmup
    return 0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))


class ScheduledRows:
    def __init__(self, path: Path, calibration: Path, protected: Sequence[Example], run: str) -> None:
        self.path = path.absolute()
        self.schedule = load_schedule(self.path)
        self.sha256 = digest(self.path)
        self.calibration = calibration
        self.calibration_sha256 = digest(calibration)
        self.run = run
        self.ids = self.schedule["ordered_ids"]
        self.protected_ids = {row["id"] for row in protected}
        self.protected_families = {
            (str(row["source"].get("dataset", row["suite"])), row["family"]) for row in protected
        }
        if set(self.ids) & self.protected_ids:
            raise ValueError("Scheduled training IDs overlap a held-out fold")
        for source, checksum in (
            (self.schedule["plan_path"], self.schedule["plan_sha256"]),
            (self.schedule["original_train_path"], self.schedule["original_train_sha256"]),
        ):
            if digest(path_at(self.path, source)) != checksum:
                raise ValueError(f"Scheduled source changed: {source}")
        if path_at(self.path, self.schedule["calibration_path"]).resolve() != calibration.resolve():
            raise ValueError("--temperature must be the calibration fold bound by the schedule")
        self.tranche_for_id = {
            identifier: part["index"] for part in self.schedule["tranches"] for identifier in part["ordered_ids"]
        }
        self.receipts: list[dict[str, JSONValue]] = []
        self.cache: dict[int, dict[str, Example]] = {}

    def _read(self, index: int) -> tuple[dict[str, JSONValue], list[Example]]:
        if digest(self.path) != self.sha256 or digest(self.calibration) != self.calibration_sha256:
            raise ValueError("The frozen schedule or calibration fold changed")
        part = self.schedule["tranches"][index]
        manifest = path_at(self.path, part["manifest_path"])
        receipt = cast(Receipt, json.loads(manifest.read_text()))
        expected = {
            "schema_version": 1,
            "complete": True,
            "schedule_sha256": self.sha256,
            "tranche": index,
            "rows": len(part["ordered_ids"]),
            "ordered_ids_sha256": ids_sha256(part["ordered_ids"]),
            "calibration_sha256": self.calibration_sha256,
            "previous_receipts_sha256": [value["manifest_sha256"] for value in self.receipts[:index]],
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Tranche {index} readiness receipt differs from the frozen schedule")
        data_path = path_at(self.path, part["data_path"])
        audit_path = path_at(self.path, receipt["audit_path"])
        for path, checksum in (
            (data_path, receipt["data_sha256"]),
            (audit_path, receipt["audit_sha256"]),
            (path_at(self.path, part["synthetic_path"]), receipt["synthetic_sha256"]),
            (path_at(self.path, self.schedule["controls_audit_path"]), receipt["controls_audit_sha256"]),
        ):
            if digest(path) != checksum:
                raise ValueError(f"Tranche {index} bound file changed: {path}")
        audit = cast(dict[str, JSONValue], json.loads(audit_path.read_text()))
        if (
            audit.get("complete") is not True
            or audit.get("phase") != "tranche"
            or audit.get("schedule_sha256") != self.sha256
            or audit.get("tranche") != index
            or audit.get("accepted_rows") != len(part["synthetic_ids"])
        ):
            raise ValueError(f"Tranche {index} has no complete matching synthetic audit")
        audited_train = cast(dict[str, JSONValue], cast(dict[str, JSONValue], audit["outputs"])["train"])
        if audited_train.get("sha256") != receipt["synthetic_sha256"]:
            raise ValueError(f"Tranche {index} synthetic rows differ from their audit")
        controls = cast(
            dict[str, JSONValue], json.loads(path_at(self.path, self.schedule["controls_audit_path"]).read_text())
        )
        if controls.get("complete") is not True or controls.get("phase") != "controls":
            raise ValueError("The fixed synthetic controls must pass their complete audit first")
        rows = read_rows(data_path)
        if [row["id"] for row in rows] != part["ordered_ids"]:
            raise ValueError(f"Tranche {index} text IDs/order differ from the frozen schedule")
        families = {(str(row["source"].get("dataset", row["suite"])), row["family"]) for row in rows}
        if families & self.protected_families:
            raise ValueError(f"Tranche {index} has a family overlapping a held-out fold")
        provenance = cast(dict[str, JSONValue], dict(receipt))
        provenance.update(manifest_path=str(manifest), manifest_sha256=digest(manifest), data_path=str(data_path))
        return provenance, rows

    def batch(self, step: int, batch_size: int, before_wait: Callable[[], None] | None = None) -> list[Example]:
        selected_ids = self.ids[step * batch_size : (step + 1) * batch_size]
        if not selected_ids:
            raise ValueError("The training cursor is outside the frozen schedule")
        needed = {self.tranche_for_id[identifier] for identifier in selected_ids}
        for index in sorted(needed):
            if index in self.cache:
                continue
            manifest = path_at(self.path, self.schedule["tranches"][index]["manifest_path"])
            waiting_started = time.monotonic()
            waiting = False
            while not manifest.exists():
                if not waiting and before_wait is not None:
                    before_wait()
                waiting = True
                record(
                    "training_waiting_for_data",
                    run=self.run,
                    step=step,
                    next_step=step + 1,
                    tranche=index,
                    schedule_sha256=self.sha256,
                    manifest=str(manifest),
                    waiting_seconds=time.monotonic() - waiting_started,
                )
                time.sleep(20)
            provenance, rows = self._read(index)
            if index == len(self.receipts):
                self.receipts.append(provenance)
                record(
                    "training_tranche_loaded",
                    run=self.run,
                    step=step,
                    tranche=index,
                    receipt=provenance,
                    waiting_seconds=time.monotonic() - waiting_started if waiting else 0,
                )
            elif index >= len(self.receipts) or self.receipts[index] != provenance:
                raise ValueError("A previously validated tranche receipt changed, or a tranche was skipped")
            self.cache[index] = {row["id"]: row for row in rows}
        result = [self.cache[self.tranche_for_id[identifier]][identifier] for identifier in selected_ids]
        self.cache = {index: rows for index, rows in self.cache.items() if index in needed}
        return sorted(result, key=length_estimate)

    def snapshot(self, examples_seen: int) -> dict[str, JSONValue]:
        if not 0 <= examples_seen <= len(self.ids):
            raise ValueError("Invalid consumed-row cursor")
        return {
            "schedule_path": str(self.path),
            "schedule_sha256": self.sha256,
            "scheduled_rows": len(self.ids),
            "examples_seen": examples_seen,
            "consumed_ids_sha256": ids_sha256(self.ids[:examples_seen]),
            "validated_tranches": cast(JSONValue, copy.deepcopy(self.receipts)),
        }

    def restore(self, saved: dict[str, JSONValue], examples_seen: int) -> None:
        receipts = saved.get("validated_tranches")
        if not isinstance(receipts, list) or len(receipts) > len(self.schedule["tranches"]):
            raise ValueError("Invalid saved tranche receipt prefix")
        if examples_seen > sum(len(part["ordered_ids"]) for part in self.schedule["tranches"][: len(receipts)]):
            raise ValueError("The saved cursor includes rows without validated tranche receipts")
        for index, previous in enumerate(receipts):
            provenance, _ = self._read(index)
            if provenance != previous:
                raise ValueError(f"Tranche {index} changed since the saved optimizer cursor")
            self.receipts.append(provenance)
        if self.snapshot(examples_seen) != saved:
            raise ValueError("The saved training schedule/cursor differs from this run")


def length_estimate(row: Example) -> int:
    measured = row["source"].get("input_tokens")
    if isinstance(measured, int) and not isinstance(measured, bool):
        return measured + 16
    return (
        len(json.dumps([row["state"], row["question"]], ensure_ascii=False)) // 3
        + 192
        + 512 * len(row.get("images", []))
    )


def fixed_subset(path: str | None, full: Sequence[Example]) -> list[Example]:
    if path is None:
        return []
    rows = read_rows(Path(path))
    originals = {row["id"]: json.dumps(row, ensure_ascii=False) for row in full}
    if (
        not rows
        or len({row["id"] for row in rows}) != len(rows)
        or any(originals.get(row["id"]) != json.dumps(row, ensure_ascii=False) for row in rows)
    ):
        raise ValueError(f"Quick evaluation file is not an exact, unique subset of its full fold: {path}")
    return rows


def microbatches(rows: Sequence[Example], batch_size: int, token_budget: int) -> list[list[Example]]:
    result: list[list[Example]] = []
    pending: list[Example] = []
    longest = 0
    for row in rows:
        length = length_estimate(row)
        if pending and (len(pending) == batch_size or max(longest, length) * (len(pending) + 1) > token_budget):
            result.append(pending)
            pending, longest = [], 0
        pending.append(row)
        longest = max(longest, length)
    if pending:
        result.append(pending)
    return result


def targets(rows: Sequence[Example], device: torch.device) -> torch.Tensor:
    values = torch.zeros((len(rows), 255), dtype=torch.float32, device=device)
    for index, row in enumerate(rows):
        labels, target = options(row["question"]), row["target"]
        if isinstance(target, list):
            distribution = target
        elif row["question"]["type"] == "noul":
            positive = float(cast(float, target))
            distribution = [1.0 - positive, positive]
        else:
            distribution = [float(label == target) for label in labels]
        if (
            len(distribution) != len(labels)
            or any(not math.isfinite(p) or p < 0 for p in distribution)
            or abs(sum(distribution) - 1) > 1e-6
        ):
            raise ValueError(f"Invalid training target: {row['id']}")
        values[index, : len(distribution)] = torch.tensor(distribution, device=device)
    return values


def augment(rows: Sequence[Example], rng: random.Random) -> list[Example]:
    result = copy.deepcopy(list(rows))
    for row in result:
        if row["question"]["type"] == "choice":
            criteria = row["question"]["criteria"]
            target = row["target"]
            weights = dict(zip(criteria, target, strict=True)) if isinstance(target, list) else None
            items = list(criteria.items())
            rng.shuffle(items)
            row["question"]["criteria"] = dict(items)
            if weights is not None:
                row["target"] = [weights[key] for key, _ in items]
    return result


@torch.inference_mode()
def infer(model: DecisionModel, rows: Sequence[Example], args: Arguments) -> list[list[float]]:
    was_training = model.training
    model.eval()
    result: list[list[float]] = []
    for batch in microbatches(rows, args.batch_size, args.token_budget):
        logits = model(model.prepare(batch, max_length=args.max_length)).detach().cpu()
        for row, values in zip(batch, logits, strict=True):
            result.append(cast(list[float], values[: len(options(row["question"]))].tolist()))
    model.train(was_training)
    return result


def save_predictions(path: Path, predictions: Sequence[Prediction]) -> None:
    with path.open("w") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def select_checkpoint(root: Path, step: int) -> None:
    pending = root / "selected.pending"
    pending.unlink(missing_ok=True)
    pending.symlink_to(f"step-{step:05d}", target_is_directory=True)
    os.replace(pending, root / "selected")
    sync_directory(root)


def save_selected(
    model: DecisionModel,
    root: Path,
    temperature: float,
    step: int,
    provenance: dict[str, str],
    initial_artifact: dict[str, JSONValue] | None = None,
    training_schedule: dict[str, JSONValue] | None = None,
) -> None:
    destination = root / f"step-{step:05d}"
    metadata: dict[str, JSONValue] = {"step": step, "provenance": cast(JSONValue, provenance)}
    if initial_artifact is not None:
        metadata["initial_artifact"] = initial_artifact
    if training_schedule is not None:
        metadata["training_schedule"] = training_schedule
    model.save(destination, temperature=temperature, **metadata)
    for file in destination.rglob("*"):
        if file.is_file():
            with file.open("rb") as stream:
                os.fsync(stream.fileno())
    sync_directory(destination)
    select_checkpoint(root, step)


def archive_interrupted_tail(run: Path, output: Path, step: int) -> None:
    suffix = str(time.time_ns())
    archive = run / "interrupted" / suffix
    for name in ("training.jsonl", "evaluations.jsonl", "public-evaluations.jsonl", "quick-evaluations.jsonl"):
        path = run / name
        if not path.exists():
            continue
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        tail = [line for line in lines if json.loads(line)["step"] > step]
        if tail:
            archive.mkdir(parents=True, exist_ok=True)
            (archive / name).write_text("\n".join(tail) + "\n")
            path.write_text("".join(line + "\n" for line in lines if json.loads(line)["step"] <= step))
    for prefix in (
        "development",
        "temperature",
        "public",
        "guard",
        "quick-development",
        "quick-temperature",
        "quick-guard",
    ):
        for path in run.glob(f"{prefix}-*.jsonl"):
            suffix_step = path.stem.rsplit("-", 1)[1]
            if suffix_step.isdigit() and int(suffix_step) > step:
                archive.mkdir(parents=True, exist_ok=True)
                path.rename(archive / path.name)
    for path in output.glob("step-*"):
        if path.is_dir() and path.name.rsplit("-", 1)[1].isdigit() and int(path.name.rsplit("-", 1)[1]) > step:
            destination = output / "interrupted" / suffix
            destination.mkdir(parents=True, exist_ok=True)
            path.rename(destination / path.name)
    record("training_resumed", run=run.name, resume_step=step, archived_tail=archive.exists(), attempt=suffix)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full-weight cross-entropy training with periodic, fixed-fold evaluation."
    )
    for name, default in [
        ("train", "data/train.jsonl"),
        ("development", "data/dev.jsonl"),
        ("temperature", "data/temperature.jsonl"),
        ("reference", "progress/baselines/jev-development.predictions.jsonl"),
    ]:
        parser.add_argument(f"--{name}", default=default)
    parser.add_argument(
        "--schedule", help="Frozen global ID schedule with immutable audited tranches; replaces --train"
    )
    parser.add_argument("--public")
    parser.add_argument("--guard-data", help="Separate development cohort whose calibration also gates selection")
    parser.add_argument("--guard-reference", help="Complete fixed Jev predictions for --guard-data")
    parser.add_argument("--quick-development", help="Frozen subset of development for diagnostic curves only")
    parser.add_argument("--quick-guard-data", help="Frozen subset of the guard cohort for diagnostic curves only")
    parser.add_argument(
        "--quick-temperature", help="Frozen subset of temperature data, used only to calibrate quick curves"
    )
    parser.add_argument("--quick-eval-every", type=int, default=20)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--revision", default="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
    parser.add_argument("--cache-dir")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--effective-batch-size", type=int, default=256)
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--brier-weight", type=float, default=0.0)
    parser.add_argument(
        "--save-final-only", action="store_true", help="Save only the final model, without optimizer snapshots"
    )
    parser.add_argument("--save-every", type=int, help="Save model snapshots at this interval and at the final step, without optimizer snapshots")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--public-eval-every", type=int, default=100)
    parser.add_argument("--resume-every", type=int, default=100)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume")
    parser.add_argument(
        "--initial-checkpoint", help="Start a new full-weight SFT stage from this artifact with a fresh optimizer"
    )
    parser.add_argument("--cpu-threads", type=int, default=32)
    parser.add_argument("--attention-mode", choices=("causal", "noncausal_full_attention"), default="causal")
    parser.add_argument("--pooling", choices=("last", "mean"), default="last")
    args = parser.parse_args(namespace=Arguments())
    if not math.isfinite(args.warmup_ratio) or not 0 < args.warmup_ratio < 1:
        raise ValueError("--warmup-ratio must be finite and strictly between zero and one")
    if not math.isfinite(args.brier_weight) or args.brier_weight < 0:
        raise ValueError("--brier-weight must be finite and nonnegative")
    if args.save_every is not None and (args.save_every < 1 or args.save_final_only):
        raise ValueError("--save-every must be positive and cannot be combined with --save-final-only")
    fixed_saving = args.save_final_only or args.save_every is not None
    if fixed_saving and (args.schedule or args.resume or args.stop_after):
        raise ValueError("Final-only saving requires a fresh, complete fixed-data run")
    if (
        min(
            args.epochs,
            args.batch_size,
            args.effective_batch_size,
            args.token_budget,
            args.eval_every,
            args.public_eval_every,
            args.resume_every,
            args.quick_eval_every,
        )
        < 1
    ):
        raise ValueError("Batch, epoch and interval settings must be positive")
    if args.stop_after is not None and args.stop_after < 1:
        raise ValueError("A pilot must include at least one training step")
    if bool(args.guard_data) != bool(args.guard_reference):
        raise ValueError("--guard-data and --guard-reference must be supplied together")
    quick_paths = (args.quick_development, args.quick_guard_data, args.quick_temperature)
    if any(quick_paths) and (not all(quick_paths) or not args.guard_data):
        raise ValueError("Quick evaluation needs all three frozen subsets and a full guard cohort/reference")
    if args.initial_checkpoint and not os.environ.get("AUTOJEV_EVENTS"):
        raise ValueError("Set AUTOJEV_EVENTS to the new stage's event log before continuing from a checkpoint")
    if args.schedule and (args.epochs != 1 or not os.environ.get("AUTOJEV_EVENTS")):
        raise ValueError("A streamed schedule requires one epoch and an explicit AUTOJEV_EVENTS path")
    distributed = DistributedTraining.initialize()
    if distributed.world_size > 1 and args.schedule:
        raise ValueError("DDP currently requires a fixed --train file, not a streamed schedule")
    run, output = Path(args.run), Path(args.output)
    if args.initial_checkpoint and any(
        path.resolve().is_relative_to(Path(args.initial_checkpoint).resolve()) for path in (run, output)
    ):
        raise ValueError("New run/output directories must be outside the initial checkpoint artifact")
    run.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if (run / "config.json").exists() and not args.resume:
        raise ValueError("Run already exists; resume explicitly or choose a new run ID")
    train = [] if args.schedule else read_rows(Path(args.train))
    development, temperature_rows = (read_rows(Path(path)) for path in (args.development, args.temperature))
    if (not train and not args.schedule) or not development or not temperature_rows:
        raise ValueError("Training, development and temperature folds must be nonempty")
    reference_predictions = read_predictions(Path(args.reference))
    validate_coverage(development, reference_predictions)
    reference = metrics(reference_predictions)
    public_rows = read_rows(Path(args.public)) if args.public else []
    guard_rows = read_rows(Path(args.guard_data)) if args.guard_data else []
    guard_reference: Metrics | None = None
    guard_predictions: list[Prediction] = []
    if args.guard_data:
        if not guard_rows:
            raise ValueError("The guard cohort must be nonempty")
        guard_predictions = read_predictions(Path(cast(str, args.guard_reference)))
        validate_coverage(guard_rows, guard_predictions)
        guard_reference = metrics(guard_predictions)
    quick_development = fixed_subset(args.quick_development, development)
    quick_guard = fixed_subset(args.quick_guard_data, guard_rows)
    quick_temperature = fixed_subset(args.quick_temperature, temperature_rows)
    quick_ids = {row["id"] for row in quick_development}
    quick_guard_ids = {row["id"] for row in quick_guard}
    quick_reference = (
        metrics([value for value in reference_predictions if value["id"] in quick_ids]) if quick_development else None
    )
    quick_guard_reference = (
        metrics([value for value in guard_predictions if value["id"] in quick_guard_ids]) if quick_guard else None
    )
    partitions = {
        "train": train,
        "development": development,
        "temperature": temperature_rows,
        "public": public_rows,
        "guard": guard_rows,
    }
    families: dict[str, set[tuple[str, str]]] = {}
    identifiers: dict[str, set[str]] = {}
    for name, rows in partitions.items():
        identifiers[name] = {row["id"] for row in rows}
        if len(identifiers[name]) != len(rows):
            raise ValueError(f"Duplicate IDs in {name}")
        families[name] = {(str(row["source"].get("dataset", row["suite"])), row["family"]) for row in rows}
        for other in identifiers:
            if name != other and (identifiers[name] & identifiers[other] or families[name] & families[other]):
                raise ValueError(f"Partitions overlap: {name}/{other}")
    scheduled = (
        ScheduledRows(
            Path(args.schedule),
            Path(args.temperature),
            development + temperature_rows + public_rows + guard_rows,
            run.name,
        )
        if args.schedule
        else None
    )
    if scheduled and scheduled.schedule["seed"] != args.seed:
        raise ValueError("The training seed must match the frozen schedule seed")
    train_rows = len(scheduled.ids) if scheduled else len(train)
    hashes = {
        name: digest(path)
        for name, path in [
            ("development", args.development),
            ("temperature", args.temperature),
            ("reference", args.reference),
        ]
    }
    hashes["train_schedule" if scheduled else "train"] = scheduled.sha256 if scheduled else digest(args.train)
    if args.public:
        hashes["public"] = digest(args.public)
    if args.guard_data:
        hashes["guard_data"] = digest(args.guard_data)
        hashes["guard_reference"] = digest(cast(str, args.guard_reference))
    for key, path in (
        ("quick_development", args.quick_development),
        ("quick_guard_data", args.quick_guard_data),
        ("quick_temperature", args.quick_temperature),
    ):
        if path is not None:
            hashes[key] = digest(path)
    initial_artifact = (
        initial_identity(args.initial_checkpoint, args.base_model, args.revision) if args.initial_checkpoint else None
    )
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Git must be installed to record training provenance")
    revision = subprocess.check_output([git, "rev-parse", "HEAD"], text=True).strip()
    package = Path(__file__).resolve().parent
    code_hashes = {
        name: digest(package / name)
        for name in (
            "train.py",
            "model.py",
            "optim.py",
            "evaluate.py",
            "types.py",
            "events.py",
            "distributed.py",
            "loss.py",
        )
    }
    if scheduled:
        code_hashes["sft_pipeline.py"] = digest(package / "sft_pipeline.py")
    code_hashes["uv.lock"] = digest(package.parents[1] / "uv.lock")
    config = {
        **vars(args),
        "data_sha256": hashes,
        "code_sha256": code_hashes,
        "git_commit": revision,
        "initial_artifact": initial_artifact,
        "events_path": os.getenv("AUTOJEV_EVENTS", "progress/events.jsonl"),
        "train_rows": train_rows,
        "development_rows": len(development),
        "temperature_rows": len(temperature_rows),
        "guard_rows": len(guard_rows),
        "quick_development_rows": len(quick_development),
        "quick_guard_rows": len(quick_guard),
        "quick_temperature_rows": len(quick_temperature),
    }
    config["world_size"] = distributed.world_size
    distributed.barrier()
    if not args.resume and distributed.primary:
        write_json(run / "config.json", config)
        if scheduled:
            scheduled.batch(0, args.effective_batch_size)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.monotonic()
    model = DecisionModel(
        checkpoint=args.initial_checkpoint,
        train=True,
        device=str(distributed.device),
        base_model=args.base_model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        gradient_checkpointing=True,
        cpu_threads=args.cpu_threads,
        attention_mode=args.attention_mode,
        pooling=args.pooling,
    )
    optimizer = (
        CPUOffloadAdamW(model.named_parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if distributed.primary
        else None
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    groups: list[list[Example]] = []
    for epoch in range(args.epochs):
        ordered = list(train)
        random.Random(args.seed + epoch).shuffle(ordered)
        for offset in range(0, len(ordered), args.effective_batch_size):
            groups.append(sorted(ordered[offset : offset + args.effective_batch_size], key=length_estimate))
    total_steps = math.ceil(train_rows / args.effective_batch_size) if scheduled else len(groups)
    step, examples_seen = 0, 0
    last_resume_step: int | None = None
    best: Metrics | None = None
    best_guard: Metrics | None = None
    best_step: int | None = None
    selected_temperature: float | None = None
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True) if distributed.primary else None
        saved_optimizer = state.pop("optimizer") if state is not None else None
        state = distributed.broadcast(state)
        if state["config"].get("world_size", 1) != distributed.world_size:
            raise ValueError("Resume world size differs from the saved run")
        if state["data_sha256"] != hashes or state["total_steps"] != total_steps:
            raise ValueError("Resume data/schedule differs from the saved run")
        if state["config"]["code_sha256"] != code_hashes:
            raise ValueError("Training implementation differs from the saved run")
        if state["config"].get("initial_artifact") != initial_artifact:
            raise ValueError("Initial checkpoint content or provenance differs from the saved stage")
        if state["config"].get("events_path") != config["events_path"]:
            raise ValueError("Resume event log differs from the saved stage")
        for key in vars(args):
            if key in {"resume", "stop_after"}:
                continue
            if state["config"][key] != vars(args)[key]:
                raise ValueError(f"Resume configuration differs: {key}")
        step, examples_seen = state["step"], state["examples_seen"]
        if scheduled:
            if examples_seen != min(step * args.effective_batch_size, train_rows):
                raise ValueError("The saved consumed-row cursor differs from its global update")
            scheduled.restore(state["training_schedule"], examples_seen)
        if optimizer is not None:
            assert saved_optimizer is not None
            optimizer.load_state_dict(saved_optimizer)
        del saved_optimizer
        last_resume_step = step
        if args.stop_after is not None and args.stop_after <= step:
            raise ValueError("The requested stopping step must follow the saved step")
        best, best_step, selected_temperature = state["best"], state["best_step"], state["selected_temperature"]
        best_guard = state.get("best_guard")
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state
        if distributed.primary:
            archive_interrupted_tail(run, output, step)
            if best_step is not None:
                selected = output / f"step-{best_step:05d}"
                if not selected.is_dir():
                    raise ValueError("The checkpoint selected by the resume state is missing")
                select_checkpoint(output, best_step)
            else:
                (output / "selected").unlink(missing_ok=True)
    training_model = distributed.wrap(model)
    if distributed.primary:
        record(
            "training_started",
            run=run.name,
            git_commit=revision,
            data_sha256=hashes,
            initial_artifact=initial_artifact,
            initialization="exact_resume"
            if args.resume
            else "checkpoint_fresh_optimizer"
            if initial_artifact
            else "base_fresh_optimizer",
            step=step,
            total_steps=total_steps,
            training_schedule=scheduled.snapshot(examples_seen) if scheduled else None,
            trainable_parameters=sum(p.numel() for p in parameters),
        )

    def evaluate(include_public: bool = False, diagnostic: bool = False) -> None:
        nonlocal best, best_guard, best_step, selected_temperature
        began = time.monotonic()
        temperature_logits = infer(model, temperature_rows, args)
        fitted = fit_temperature(
            temperature_logits, [label_index(options(row["question"]), hard_label(row)) for row in temperature_rows]
        )
        logits = infer(model, development, args)
        raw_predictions, fitted_predictions = (
            evaluate_logits(development, logits),
            evaluate_logits(development, logits, fitted),
        )
        raw, calibrated = metrics(raw_predictions), metrics(fitted_predictions)
        guard_raw: Metrics | None = None
        guard_calibrated: Metrics | None = None
        guard_ok = True
        if guard_rows:
            guard_logits = infer(model, guard_rows, args)
            guard_raw = metrics(evaluate_logits(guard_rows, guard_logits))
            guard_predictions = evaluate_logits(guard_rows, guard_logits, fitted)
            guard_calibrated = metrics(guard_predictions)
            guard_ok = calibration_ok(guard_calibrated, cast(Metrics, guard_reference))
            save_predictions(run / f"guard-{step:05d}.jsonl", guard_predictions)
        public_ok = calibration_ok(calibrated, reference)
        eligible = public_ok and guard_ok
        improved = (
            not diagnostic
            and eligible
            and (best is None or selection_key(calibrated, step) < selection_key(best, cast(int, best_step)))
        )
        if fixed_saving:
            improved = step == total_steps or (args.save_every is not None and step > 0 and step % args.save_every == 0)
        if improved:
            save_selected(
                model,
                output,
                fitted,
                step,
                {"run": run.name, "git_commit": revision, **hashes, **code_hashes},
                initial_artifact,
                scheduled.snapshot(examples_seen) if scheduled else None,
            )
            best, best_guard, best_step, selected_temperature = calibrated, guard_calibrated, step, fitted
        save_predictions(run / f"development-{step:05d}.jsonl", fitted_predictions)
        save_predictions(
            run / f"temperature-{step:05d}.jsonl", evaluate_logits(temperature_rows, temperature_logits, fitted)
        )
        guard_target = (
            guard_calibrated is not None
            and guard_reference is not None
            and guard_ok
            and guard_calibrated["accuracy"] > guard_reference["accuracy"]
        )
        value = record(
            "evaluation",
            run=run.name,
            step=step,
            examples_seen=examples_seen,
            raw=raw,
            fitted=calibrated,
            temperature=fitted,
            jev=reference,
            eligible=eligible,
            public_calibration_ok=public_ok,
            guard_raw=guard_raw,
            guard_fitted=guard_calibrated,
            guard_jev=guard_reference,
            guard_calibration_ok=guard_ok,
            guard_empirical_target_met=guard_target,
            development_target_met=eligible and calibrated["accuracy"] > reference["accuracy"],
            diagnostic=diagnostic,
            selected_step=best_step,
            elapsed_seconds=time.monotonic() - started,
            evaluation_seconds=time.monotonic() - began,
        )
        append(run / "evaluations.jsonl", value)
        print(json.dumps(value), flush=True)
        if include_public and public_rows:
            predictions = evaluate_logits(public_rows, infer(model, public_rows, args), fitted)
            save_predictions(run / f"public-{step:05d}.jsonl", predictions)
            value = record(
                "public_evaluation",
                run=run.name,
                step=step,
                temperature=fitted,
                metrics=metrics(predictions),
                status="known_suite_development_check",
            )
            append(run / "public-evaluations.jsonl", value)

    def save_resume() -> None:
        nonlocal last_resume_step
        assert optimizer is not None
        began = time.monotonic()
        state = {
            "optimizer": optimizer.state_dict(),
            "step": step,
            "examples_seen": examples_seen,
            "total_steps": total_steps,
            "data_sha256": hashes,
            "config": config,
            "best": best,
            "best_guard": best_guard,
            "best_step": best_step,
            "selected_temperature": selected_temperature,
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        }
        if scheduled:
            state["training_schedule"] = scheduled.snapshot(examples_seen)
        pending = output / "resume.pt.pending"
        torch.save(state, pending)
        with pending.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(pending, output / "resume.pt")
        sync_directory(output)
        last_resume_step = step
        record(
            "resume_saved",
            run=run.name,
            step=step,
            bytes=(output / "resume.pt").stat().st_size,
            seconds=time.monotonic() - began,
        )

    def evaluate_quick() -> None:
        began = time.monotonic()
        temperature_logits = infer(model, quick_temperature, args)
        fitted = fit_temperature(
            temperature_logits, [label_index(options(row["question"]), hard_label(row)) for row in quick_temperature]
        )
        logits = infer(model, quick_development, args)
        predictions = evaluate_logits(quick_development, logits, fitted)
        guard_logits = infer(model, quick_guard, args)
        guard_predictions = evaluate_logits(quick_guard, guard_logits, fitted)
        save_predictions(run / f"quick-development-{step:05d}.jsonl", predictions)
        save_predictions(run / f"quick-guard-{step:05d}.jsonl", guard_predictions)
        save_predictions(
            run / f"quick-temperature-{step:05d}.jsonl", evaluate_logits(quick_temperature, temperature_logits, fitted)
        )
        value = record(
            "quick_evaluation",
            run=run.name,
            step=step,
            examples_seen=examples_seen,
            raw=metrics(evaluate_logits(quick_development, logits)),
            fitted=metrics(predictions),
            guard_raw=metrics(evaluate_logits(quick_guard, guard_logits)),
            guard_fitted=metrics(guard_predictions),
            temperature=fitted,
            jev=quick_reference,
            guard_jev=quick_guard_reference,
            diagnostic=True,
            selection_candidate=False,
            selected_step=best_step,
            temperature_rows=len(quick_temperature),
            development_rows=len(quick_development),
            guard_rows=len(quick_guard),
            elapsed_seconds=time.monotonic() - started,
            evaluation_seconds=time.monotonic() - began,
        )
        append(run / "quick-evaluations.jsonl", value)
        print(json.dumps(value), flush=True)

    def save_before_wait() -> None:
        if step > 0 and last_resume_step != step:
            save_resume()

    if step == 0:
        distributed.on_primary(lambda: evaluate(include_public=bool(public_rows)))
    stop = min(total_steps, args.stop_after) if args.stop_after is not None else total_steps
    while step < stop:
        rows = scheduled.batch(step, args.effective_batch_size, save_before_wait) if scheduled else groups[step]
        began = time.monotonic()
        model.train()
        group = augment(rows, random.Random(args.seed + 100003 * (step + 1)))
        model.zero_grad(set_to_none=True)
        if optimizer is not None:
            optimizer.zero_grad()
        if distributed.world_size > 1:
            torch.manual_seed(args.seed + 100003 * (step + 1) + distributed.rank)
        total_loss = 0.0
        total_cross_entropy = 0.0
        total_brier = 0.0
        input_tokens = 0
        batches = distributed.batches(group, lambda rows: microbatches(rows, args.batch_size, args.token_budget))
        for batch, count in batches:
            prepared = model.prepare(batch, max_length=args.max_length)
            logits = training_model(prepared)
            target = targets(batch, logits.device)
            components = decision_loss(logits, target, args.brier_weight)
            loss = components.total
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step + 1}")
            (loss * count * distributed.world_size / len(group)).backward()
            total_loss += float(loss.detach()) * count
            total_cross_entropy += float(components.cross_entropy.detach()) * count
            total_brier += float(components.brier.detach()) * count
            input_tokens += prepared.input_tokens if count else 0
            del logits, loss, components, target, prepared
        total_loss, input_tokens = distributed.totals(total_loss, input_tokens)
        total_cross_entropy, _ = distributed.totals(total_cross_entropy, 0)
        total_brier, _ = distributed.totals(total_brier, 0)
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
        if not math.isfinite(gradient_norm):
            raise RuntimeError(f"Nonfinite gradient at step {step + 1}")
        step += 1
        factor = learning_rate_factor(step, total_steps, args.warmup_ratio)
        optimizer_started = time.monotonic()
        if optimizer is not None:
            for group_parameters in optimizer.param_groups:
                group_parameters["lr"] = args.lr * factor
            optimizer.step()
        distributed.broadcast_parameters(model)
        torch.cuda.synchronize()
        examples_seen += len(group)
        if distributed.primary:
            value = record(
                "training_step",
                run=run.name,
                step=step,
                loss=total_loss / len(group),
                cross_entropy=total_cross_entropy / len(group),
                brier=total_brier / len(group),
                brier_weight=args.brier_weight,
                learning_rate=args.lr * factor,
                examples_seen=examples_seen,
                group_examples=len(group),
                input_tokens=input_tokens,
                gradient_norm=gradient_norm,
                step_seconds=time.monotonic() - began,
                optimizer_seconds=time.monotonic() - optimizer_started,
                elapsed_seconds=time.monotonic() - started,
                gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9,
            )
            append(run / "training.jsonl", value)
            print(json.dumps(value), flush=True)
        if step % args.eval_every == 0 or step == stop or (args.save_every is not None and step % args.save_every == 0):
            distributed.on_primary(
                partial(
                    evaluate,
                    include_public=step % args.public_eval_every == 0 or step == total_steps,
                    diagnostic=step % args.eval_every != 0 and step != total_steps,
                )
            )
        elif quick_development and step % args.quick_eval_every == 0:
            distributed.on_primary(evaluate_quick)
        if not fixed_saving and (step % args.resume_every == 0 or step == stop):
            distributed.on_primary(save_resume)
    if not distributed.primary:
        distributed.barrier()
        distributed.close()
        return
    summary = {
        "run": run.name,
        "steps": step,
        "planned_steps": total_steps,
        "complete": step == total_steps,
        "examples_seen": examples_seen,
        "best_step": None if fixed_saving else best_step,
        "saved_step": best_step,
        "checkpoint_policy": "periodic_final" if args.save_every is not None else ("final_only" if args.save_final_only else "dev_selected"),
        "selected_temperature": selected_temperature,
        "selected_metrics": best,
        "selected_guard_metrics": best_guard,
        "jev": reference,
        "guard_jev": guard_reference,
        "initial_artifact": initial_artifact,
        "data_sha256": hashes,
        "git_commit": revision,
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint": str(output / "selected") if best else None,
    }
    if scheduled:
        summary["training_schedule"] = scheduled.snapshot(examples_seen)
    write_json(run / "summary.json", summary)
    record("training_finished" if step == total_steps else "training_paused", **summary)
    print(json.dumps(summary, indent=2), flush=True)
    distributed.barrier()
    distributed.close()


if __name__ == "__main__":
    main()
