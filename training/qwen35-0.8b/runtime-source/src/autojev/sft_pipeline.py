from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TypedDict, cast

from autojev.events import record
from autojev.sft_synthetic import COUNTS, Slot, load_plan
from autojev.types import Example, JSONValue

SEED = 20260921
TRANCHES = 10
NEW_ROWS = 5000
REPLAY_ROWS = 1920
TOTAL_ROWS = 69200


class Tranche(TypedDict):
    index: int
    synthetic_ids: list[str]
    synthetic_families: list[str]
    replay_ids: list[str]
    ordered_ids: list[str]
    data_path: str
    synthetic_path: str
    manifest_path: str
    counts: dict[str, JSONValue]


class Schedule(TypedDict):
    schema_version: int
    seed: int
    total_rows: int
    ordered_ids: list[str]
    plan_path: str
    plan_sha256: str
    original_train_path: str
    original_train_sha256: str
    replay_ids: list[str]
    selected_replay_sha256: str
    selected_replay_ids_sha256: str
    public_calibration_path: str
    public_calibration_sha256: str
    calibration_path: str
    calibration_ids: list[str]
    controls_audit_path: str
    synthetic_calibration_path: str
    synthetic_confirmation_path: str
    control_ids: dict[str, list[str]]
    audit_order: list[str]
    assignments: list[dict[str, JSONValue]]
    code_sha256: dict[str, str]
    tranches: list[Tranche]


class Receipt(TypedDict):
    schema_version: int
    complete: bool
    schedule_sha256: str
    tranche: int
    rows: int
    data_sha256: str
    synthetic_sha256: str
    ordered_ids_sha256: str
    audit_path: str
    audit_sha256: str
    calibration_sha256: str
    controls_audit_sha256: str
    previous_receipts_sha256: list[str]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def path_at(schedule_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else schedule_path.parent / path


def relative(path: Path, schedule_path: Path) -> str:
    return os.path.relpath(path.absolute(), schedule_path.parent.absolute())


def rows(path: Path) -> Iterator[Example]:
    with path.open() as stream:
        for line in stream:
            yield cast(Example, json.loads(line))


def document(path: Path) -> dict[str, JSONValue]:
    return cast(dict[str, JSONValue], json.loads(path.read_bytes()))


def load_schedule(path: Path) -> Schedule:
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    schedule = cast(Schedule, json.loads(raw))
    tranches = schedule["tranches"]
    if schedule["schema_version"] != 1 or [item["index"] for item in tranches] != list(range(TRANCHES)):
        raise ValueError(f"Schedule must contain exactly {TRANCHES} ordered tranches")
    concatenated: list[str] = []
    for item in tranches:
        new, replay, ordered = item["synthetic_ids"], item["replay_ids"], item["ordered_ids"]
        if (
            len(new) != NEW_ROWS
            or len(replay) != REPLAY_ROWS
            or len(ordered) != NEW_ROWS + REPLAY_ROWS
            or len(set(new + replay)) != len(ordered)
            or set(new + replay) != set(ordered)
        ):
            raise ValueError(f"Tranche {item['index']} has invalid coverage or repeated IDs")
        concatenated.extend(ordered)
    if (
        concatenated != schedule["ordered_ids"]
        or len(concatenated) != schedule["total_rows"]
        or len(concatenated) != TOTAL_ROWS
        or len(set(concatenated)) != TOTAL_ROWS
    ):
        raise ValueError("Global schedule order, count or uniqueness differs")
    return schedule


def immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to replace immutable artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o444)
    temporary.replace(path)


def encoded(example: Example) -> bytes:
    return (json.dumps(example, ensure_ascii=False) + "\n").encode()


def rank(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def make_schedule(
    plan_path: Path, original: Path, public_calibration: Path, controls_audit: Path, calibration: Path, output: Path
) -> Schedule:
    plan = load_plan(plan_path)
    groups: dict[tuple[str, str], dict[str, list[Slot]]] = defaultdict(lambda: defaultdict(list))
    for slot in plan["slots"]:
        if slot["split"] == "train":
            groups[slot["task"], slot["language"]][slot["family"]].append(slot)
    allocated: list[list[Slot]] = [[] for _ in range(TRANCHES)]
    for key, families in sorted(groups.items()):
        if len(families) % TRANCHES or any(len(pair) != 2 for pair in families.values()):
            raise ValueError(f"Cannot assign exact paired task/language quotas: {key}")
        quota = len(families) // TRANCHES
        pilot = sorted((family for family, pair in families.items() if pair[0]["pilot"]), key=rank)
        pilot_set = set(pilot)
        remaining = sorted((family for family in families if family not in pilot_set), key=rank)
        if len(pilot) > quota:
            raise ValueError("Pilot exceeds the first tranche quota")
        cursor = 0

        for index in range(TRANCHES // 2):
            chosen = pilot[:] if index == 0 else []
            take = 2 * quota - len(chosen)
            chosen += remaining[cursor : cursor + take]
            cursor += take
            for half in range(2):
                allocated[2 * index + half].extend(
                    slot for family in chosen[half * quota : (half + 1) * quota] for slot in families[family]
                )

    old: dict[str, dict[str, JSONValue]] = {}
    by_suite: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in rows(original):
        if row["id"] in old:
            raise ValueError("Original training IDs repeat")
        old[row["id"]] = {
            "id": row["id"],
            "family": row["family"],
            "suite": row["suite"],
            "kind": "replay",
            "label": row["label"],
            "image_count": len(row.get("images", [])),
        }
        by_suite[row["suite"]][row["family"]].append(row["id"])
    if len(old) != 96000:
        raise ValueError("Expected the frozen original 96,000 training rows")
    replay: list[list[str]] = [[] for _ in range(TRANCHES)]
    for suite, replay_families in sorted(by_suite.items()):
        size = sum(map(len, replay_families.values()))
        if size % (5 * TRANCHES):
            raise ValueError(f"Source quota is not divisible across {TRANCHES} tranches: {suite}")
        quota = size // 5
        selected: list[str] = []
        for family in sorted(replay_families, key=lambda value: rank(f"replay:{suite}:{value}")):
            if len(selected) + len(replay_families[family]) <= quota:
                selected.extend(sorted(replay_families[family], key=rank))
        if len(selected) != quota:
            raise ValueError(f"Whole-family replay sampling did not fill {suite}: {len(selected)}/{quota}")
        for index in range(TRANCHES):
            replay[index].extend(selected[index * (quota // TRANCHES) : (index + 1) * (quota // TRANCHES)])

    assignments: list[dict[str, JSONValue]] = []
    tranches: list[Tranche] = []
    for index, slots in enumerate(allocated):
        slots.sort(key=lambda slot: (rank(f"tranche:{index}:{slot['family']}"), slot["variant"]))
        task_counts = Counter(slot["task"] for slot in slots)
        if (
            dict(task_counts) != {task: count // TRANCHES for task, count in COUNTS.items()}
            or len(replay[index]) != REPLAY_ROWS
        ):
            raise ValueError("Tranche task or replay quotas differ")
        for task, count in task_counts.items():
            if Counter(slot["language"] for slot in slots if slot["task"] == task) != {
                "EN": count * 7 // 10,
                "DE": count // 10,
                "ES": count // 10,
                "FR": count // 10,
            }:
                raise ValueError("Tranche task/language margins differ")
        new_ids = [slot["id"] for slot in slots]
        ordered = new_ids + replay[index]
        random.Random(int(rank(f"training-order:{index}"), 16)).shuffle(ordered)
        folder = Path("data/continuation/tranches") / f"{index:03d}"
        tranches.append(
            {
                "index": index,
                "synthetic_ids": new_ids,
                "synthetic_families": list(dict.fromkeys(slot["family"] for slot in slots)),
                "replay_ids": replay[index],
                "ordered_ids": ordered,
                "data_path": relative(folder / "train.jsonl", output),
                "synthetic_path": relative(folder / "synthetic.jsonl", output),
                "manifest_path": relative(Path("progress/continuation/tranches") / f"{index:03d}.json", output),
                "counts": {
                    "synthetic_tasks": dict(task_counts),
                    "replay_suites": dict(Counter(str(old[id_]["suite"]) for id_ in replay[index])),
                    "replay_labels": dict(Counter(f"{old[id_]['suite']}:{old[id_]['label']}" for id_ in replay[index])),
                    "replay_image_rows": sum(bool(old[id_]["image_count"]) for id_ in replay[index]),
                },
            }
        )
        for slot in slots:
            assignments.append(
                {
                    "id": slot["id"],
                    "family": slot["family"],
                    "kind": "synthetic",
                    "tranche": index,
                    "task": slot["task"],
                    "language": slot["language"],
                    "domain": slot["domain"],
                    "phenomenon": slot["phenomenon"],
                    "style": slot["style"],
                    "length": slot["length"],
                    "ontology": slot["ontology"],
                    "pilot": slot["pilot"],
                    "menu_size": len(slot["option_labels"]),
                    "label": slot["desired_label"],
                }
            )
        assignments.extend({**old[id_], "tranche": index} for id_ in replay[index])
    global_ids = [id_ for tranche in tranches for id_ in tranche["ordered_ids"]]
    if len(global_ids) != TOTAL_ROWS or len(set(global_ids)) != TOTAL_ROWS:
        raise ValueError("Schedule IDs repeat or total differs")
    if sum(slot["pilot"] for slot in allocated[0]) != 2000 or any(
        slot["pilot"] for group in allocated[1:] for slot in group
    ):
        raise ValueError("All 2,000 pilot examples must be in tranche zero")
    public_ids = [row["id"] for row in rows(public_calibration)]
    controls = {
        split: [slot["id"] for slot in plan["slots"] if slot["split"] == split]
        for split in ("calibration", "confirmation")
    }
    if len(set(public_ids)) != 2500 or len(public_ids) != 2500 or any(len(ids) != 1000 for ids in controls.values()):
        raise ValueError("Expected 2,500 public calibration and 1,000 examples per synthetic control")
    replay_ids = [id_ for group in replay for id_ in group]
    selected_set = set(replay_ids)
    replay_hash = hashlib.sha256()
    for row in rows(original):
        if row["id"] in selected_set:
            replay_hash.update(encoded(row))
    return {
        "schema_version": 1,
        "seed": SEED,
        "total_rows": TOTAL_ROWS,
        "ordered_ids": global_ids,
        "plan_path": relative(plan_path, output),
        "plan_sha256": digest(plan_path),
        "original_train_path": relative(original, output),
        "original_train_sha256": digest(original),
        "replay_ids": replay_ids,
        "selected_replay_sha256": replay_hash.hexdigest(),
        "selected_replay_ids_sha256": ids_sha256(replay_ids),
        "public_calibration_path": relative(public_calibration, output),
        "public_calibration_sha256": digest(public_calibration),
        "calibration_path": relative(calibration, output),
        "calibration_ids": public_ids + controls["calibration"],
        "controls_audit_path": relative(controls_audit, output),
        "synthetic_calibration_path": relative(controls_audit.parent / "calibration.jsonl", output),
        "synthetic_confirmation_path": relative(controls_audit.parent / "confirmation.jsonl", output),
        "control_ids": controls,
        "audit_order": controls["calibration"]
        + controls["confirmation"]
        + [id_ for tranche in tranches for id_ in tranche["synthetic_ids"]],
        "assignments": assignments,
        "code_sha256": {"src/autojev/sft_pipeline.py": digest(Path(__file__))},
        "tranches": tranches,
    }


def verify_output(report: dict[str, JSONValue], split: str, path: Path, expected: list[str]) -> list[Example]:
    outputs = cast(dict[str, dict[str, JSONValue]], report["outputs"])
    result = list(rows(path))
    if (
        outputs[split]["sha256"] != digest(path)
        or len(result) != len(expected)
        or {row["id"] for row in result} != set(expected)
    ):
        raise ValueError(f"Audited {split} data differs from its receipt or scheduled IDs")
    return result


def freeze_tranche(schedule_path: Path, index: int, synthetic: Path, audit_path: Path) -> Receipt:
    schedule = load_schedule(schedule_path)
    schedule_hash = digest(schedule_path)
    if not 0 <= index < TRANCHES:
        raise ValueError(f"Tranche index must be 0..{TRANCHES - 1}")
    if schedule["code_sha256"]["src/autojev/sft_pipeline.py"] != digest(Path(__file__)):
        raise ValueError("Packaging code changed after the schedule freeze")
    if digest(path_at(schedule_path, schedule["plan_path"])) != schedule["plan_sha256"]:
        raise ValueError("Generation plan changed after the schedule freeze")
    original = path_at(schedule_path, schedule["original_train_path"])
    if digest(original) != schedule["original_train_sha256"]:
        raise ValueError("Original training dataset changed")
    controls_path = path_at(schedule_path, schedule["controls_audit_path"])
    controls = document(controls_path)
    if (
        controls.get("complete") is not True
        or controls.get("phase") != "controls"
        or controls.get("accepted_rows") != 2000
        or cast(dict[str, JSONValue], controls["plan"])["sha256"] != schedule["plan_sha256"]
    ):
        raise ValueError("Complete frozen controls audit is required before any training tranche")
    control_paths = {
        "calibration": schedule["synthetic_calibration_path"],
        "confirmation": schedule["synthetic_confirmation_path"],
    }
    control_rows = {
        split: verify_output(controls, split, path_at(schedule_path, value), schedule["control_ids"][split])
        for split, value in control_paths.items()
    }
    public = path_at(schedule_path, schedule["public_calibration_path"])
    if digest(public) != schedule["public_calibration_sha256"]:
        raise ValueError("Public calibration data changed")
    temperature = {row["id"]: row for row in [*rows(public), *control_rows["calibration"]]}
    if len(temperature) != 3500 or set(temperature) != set(schedule["calibration_ids"]):
        raise ValueError("Combined calibration data must contain the exact 3,500 scheduled IDs")
    temperature_path = path_at(schedule_path, schedule["calibration_path"])
    immutable(temperature_path, b"".join(encoded(temperature[id_]) for id_ in schedule["calibration_ids"]))
    temperature_hash = digest(temperature_path)
    previous: list[str] = []
    prior_hashes = {digest(path_at(schedule_path, value)) for value in control_paths.values()}
    for prior in schedule["tranches"][:index]:
        path = path_at(schedule_path, prior["manifest_path"])
        receipt = cast(Receipt, document(path))
        if (
            receipt["complete"] is not True
            or receipt["schedule_sha256"] != schedule_hash
            or receipt["tranche"] != prior["index"]
            or receipt["previous_receipts_sha256"] != previous
            or receipt["calibration_sha256"] != temperature_hash
            or receipt["controls_audit_sha256"] != digest(controls_path)
            or receipt["data_sha256"] != digest(path_at(schedule_path, prior["data_path"]))
            or receipt["synthetic_sha256"] != digest(path_at(schedule_path, prior["synthetic_path"]))
            or receipt["audit_sha256"] != digest(path_at(schedule_path, receipt["audit_path"]))
        ):
            raise ValueError("An earlier immutable tranche receipt or its inputs changed")
        previous.append(digest(path))
        prior_hashes.add(receipt["synthetic_sha256"])
    audit = document(audit_path)
    actual_prior = {str(item["sha256"]) for item in cast(list[dict[str, JSONValue]], audit["prior_inputs"])}
    if (
        audit.get("complete") is not True
        or audit.get("phase") != "tranche"
        or audit.get("accepted_rows") != NEW_ROWS
        or audit.get("schedule_sha256") != schedule_hash
        or audit.get("tranche") != index
        or cast(dict[str, JSONValue], audit["plan"])["sha256"] != schedule["plan_sha256"]
        or not prior_hashes <= actual_prior
    ):
        raise ValueError("Tranche audit is incomplete or does not bind the schedule and all previous accepted data")
    tranche = schedule["tranches"][index]
    new_rows = verify_output(audit, "train", synthetic, tranche["synthetic_ids"])
    replay_set = set(tranche["replay_ids"])
    selected = {row["id"]: row for row in rows(original) if row["id"] in replay_set}
    selected.update({row["id"]: row for row in new_rows})
    if set(selected) != set(tranche["ordered_ids"]):
        raise ValueError("Packaged training IDs differ from the schedule")
    new_path, data_path = (
        path_at(schedule_path, tranche["synthetic_path"]),
        path_at(schedule_path, tranche["data_path"]),
    )
    immutable(new_path, synthetic.read_bytes())
    immutable(data_path, b"".join(encoded(selected[id_]) for id_ in tranche["ordered_ids"]))
    result: Receipt = {
        "schema_version": 1,
        "complete": True,
        "schedule_sha256": schedule_hash,
        "tranche": index,
        "rows": len(selected),
        "data_sha256": digest(data_path),
        "synthetic_sha256": digest(new_path),
        "ordered_ids_sha256": ids_sha256(tranche["ordered_ids"]),
        "audit_path": relative(audit_path, schedule_path),
        "audit_sha256": digest(audit_path),
        "calibration_sha256": temperature_hash,
        "controls_audit_sha256": digest(controls_path),
        "previous_receipts_sha256": previous,
    }
    receipt_path = path_at(schedule_path, tranche["manifest_path"])
    exists = receipt_path.exists()
    immutable(receipt_path, (json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode())
    if not exists:
        record(
            "training_tranche_ready",
            tranche=index,
            rows=len(selected),
            schedule_sha256=schedule_hash,
            receipt=str(receipt_path),
            receipt_sha256=digest(receipt_path),
            data_sha256=result["data_sha256"],
        )
    return result


def quick_key(identifier: str) -> str:
    return rank(f"quick-evaluation:{identifier}")


def quick_public(path: Path, per_suite: int) -> list[Example]:
    groups: dict[str, list[Example]] = defaultdict(list)
    for row in rows(path):
        groups[row["suite"]].append(row)
    if len(groups) != 5:
        raise ValueError("Quick public panels require five source suites")
    selected = [
        row
        for _, group in sorted(groups.items())
        for row in sorted(group, key=lambda row: quick_key(row["id"]))[:per_suite]
    ]
    if len(selected) != 5 * per_suite or len({row["family"] for row in selected}) != len(selected):
        raise ValueError("Expected enough distinct public source families")
    return selected


def make_quick_plan(output: Path, generation_plan: Path) -> dict[str, JSONValue]:
    if output.exists():
        raise ValueError("Quick evaluation IDs are already frozen")
    paths = {
        "public": Path("data/transfer/public.jsonl"),
        "guard": Path("data/transfer/synthetic.jsonl"),
        "public_temperature": Path("data/continuation/public-temperature.jsonl"),
    }
    public = quick_public(paths["public"], 200)
    public_temperature = quick_public(paths["public_temperature"], 100)
    cells: dict[tuple[str, str, str], list[Example]] = defaultdict(list)
    for row in rows(paths["guard"]):
        source = row["source"]
        cells[str(source["task"]), str(source["condition"]), str(source["language"])].append(row)
    if len(cells) != 200:
        raise ValueError("Guard needs ten tasks × five conditions × four languages")
    guard: list[Example] = []
    tasks = sorted({cell[0] for cell in cells})
    for task in tasks:
        conditions = sorted(
            {cell[1] for cell in cells if cell[0] == task}, key=lambda value: quick_key(f"{task}:{value}")
        )
        other_languages = sorted(("DE", "ES", "FR"), key=lambda value: quick_key(f"{task}:{value}"))
        for index, condition in enumerate(conditions):
            quotas = {
                "EN": 2 if index < 3 else 1,
                **{language: int(index >= 3 or language != other_languages[index]) for language in other_languages},
            }
            for language, count in sorted(quotas.items()):
                guard_chosen = sorted(cells[task, condition, language], key=lambda row: quick_key(row["id"]))[:count]
                if len(guard_chosen) != count:
                    raise ValueError("Insufficient guard rows for the fixed task/condition/language quotas")
                guard.extend(guard_chosen)
    if len({row["family"] for row in guard}) != 200:
        raise ValueError("Quick guard families must be distinct")

    plan = load_plan(generation_plan)
    by_cell: dict[tuple[str, str], dict[str, list[Slot]]] = defaultdict(lambda: defaultdict(list))
    for slot in plan["slots"]:
        if slot["split"] == "calibration":
            by_cell[slot["task"], slot["language"]][slot["family"]].append(slot)
    family_targets = {task: count // 500 for task, count in COUNTS.items()}
    weights = {"EN": 7, "DE": 1, "ES": 1, "FR": 1}

    allocation = {
        (task, language): max(1, total * weight // 10)
        for task, total in family_targets.items()
        for language, weight in weights.items()
    }
    remaining_task = {
        task: total - sum(allocation[task, language] for language in weights) for task, total in family_targets.items()
    }
    remaining_language = {
        language: weight * 10 - sum(allocation[task, language] for task in family_targets)
        for language, weight in weights.items()
    }
    while sum(remaining_task.values()):
        choices = [
            (task, language)
            for task in family_targets
            for language in weights
            if remaining_task[task] > 0 and remaining_language[language] > 0
        ]
        if not choices:
            raise ValueError("Cannot fill declared paired temperature margins")
        task, language = min(
            choices,
            key=lambda cell: (
                -(family_targets[cell[0]] * weights[cell[1]] - 10 * allocation[cell]),
                quick_key(":".join(cell)),
            ),
        )
        allocation[task, language] += 1
        remaining_task[task] -= 1
        remaining_language[language] -= 1
    synthetic_ids: list[str] = []
    for cell, count in sorted(allocation.items()):
        families = by_cell[cell]
        chosen = sorted(families, key=quick_key)[:count]
        if len(chosen) != count or any(len(families[family]) != 2 for family in chosen):
            raise ValueError("Insufficient complete calibration families")
        synthetic_ids.extend(
            slot["id"] for family in chosen for slot in sorted(families[family], key=lambda slot: slot["variant"])
        )
    if len(synthetic_ids) != 200 or any(remaining_language.values()):
        raise ValueError("Synthetic temperature row or language totals differ")
    destination = Path("data/continuation")
    public_path, guard_path = destination / "quick-public.jsonl", destination / "quick-guard.jsonl"
    immutable(public_path, b"".join(map(encoded, public)))
    immutable(guard_path, b"".join(map(encoded, guard)))
    result: dict[str, JSONValue] = {
        "schema_version": 1,
        "seed": SEED,
        "selection_namespace": "quick-evaluation",
        "purpose": "Diagnostic only; selection uses the complete public and guard cohorts every50 updates and at the final update.",
        "quick_every": 20,
        "full_every": 50,
        "generation_plan": {"path": str(generation_plan), "sha256": digest(generation_plan)},
        "source_files": {key: {"path": str(path), "sha256": digest(path)} for key, path in paths.items()},
        "code_sha256": digest(Path(__file__)),
        "public": {
            "path": str(public_path),
            "sha256": digest(public_path),
            "rows": 1000,
            "ids": [row["id"] for row in public],
            "rule": "200 SHA-ranked IDs per source; distinct source families.",
        },
        "guard": {
            "path": str(guard_path),
            "sha256": digest(guard_path),
            "rows": 200,
            "ids": [row["id"] for row in guard],
            "rule": "Four SHA-ranked IDs per task/condition; three SHA-ranked conditions use twoEN and omit a distinct other language, remaining two use one per language.",
            "language_weighting": "Preserve the full guard40/20/20/20 mixture: eightEN and fourDE/ES/FR per task.",
            "task_condition_language_rows": {
                task: {
                    condition: {
                        language: sum(
                            row["source"]["task"] == task
                            and row["source"]["condition"] == condition
                            and row["source"]["language"] == language
                            for row in guard
                        )
                        for language in ("EN", "DE", "ES", "FR")
                    }
                    for condition in sorted({cell[1] for cell in cells if cell[0] == task})
                }
                for task in tasks
            },
        },
        "temperature": {
            "path": str(destination / "quick-temperature.jsonl"),
            "rows": 700,
            "public_ids": [row["id"] for row in public_temperature],
            "synthetic_ids": cast(JSONValue, synthetic_ids),
            "ids": cast(JSONValue, [row["id"] for row in public_temperature] + synthetic_ids),
            "synthetic_task_language_rows": {
                task: {language: 2 * allocation[task, language] for language in weights} for task in family_targets
            },
            "rule": "100 SHA-ranked public IDs per source plus100 whole synthetic families. Exact task and aggregate70/10/10/10 language margins; floor/minimum-one families then largest deficit with SHA tie-breaking.",
            "materialization": "After the complete synthetic controls audit; no synthetic text or fitted temperature is chosen here.",
        },
        "limitations": [
            "Small-panel sampling variation is diagnostic, not a checkpoint-selection signal.",
            "All labels/states/options are copied unchanged from their full parent folds; no model scores or confirmation records enter selection.",
            "Synthetic task/language cells use whole-family integer rounding; the exact table above replaces infeasible fractional per-cell quotas.",
        ],
    }
    immutable(output, (json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode())
    record(
        "quick_evaluation_ids_frozen",
        path=str(output),
        sha256=digest(output),
        public_rows=1000,
        guard_rows=200,
        temperature_rows=700,
    )
    return result


def materialize_quick_temperature(plan_path: Path, controls_audit: Path) -> Path:
    plan = document(plan_path)
    sources = cast(dict[str, dict[str, JSONValue]], plan["source_files"])
    public_source = sources["public_temperature"]
    public = Path(str(public_source["path"]))
    if digest(public) != public_source["sha256"]:
        raise ValueError("Public temperature source changed after quick-ID selection")
    controls = document(controls_audit)
    generation_plan = cast(dict[str, JSONValue], plan["generation_plan"])
    if (
        controls.get("complete") is not True
        or controls.get("phase") != "controls"
        or controls.get("accepted_rows") != 2000
        or cast(dict[str, JSONValue], controls["plan"])["sha256"] != generation_plan["sha256"]
    ):
        raise ValueError("Complete matching controls audit is required")
    synthetic = controls_audit.parent / "calibration.jsonl"
    outputs = cast(dict[str, dict[str, JSONValue]], controls["outputs"])
    if digest(synthetic) != outputs["calibration"]["sha256"]:
        raise ValueError("Synthetic calibration differs from its audit")
    temperature = cast(dict[str, JSONValue], plan["temperature"])
    ids = cast(list[str], temperature["ids"])
    full_rows = [*rows(public), *rows(synthetic)]
    source = {row["id"]: row for row in full_rows}
    if len(full_rows) != 3500 or len(source) != 3500 or len(ids) != 700 or not set(ids) <= source.keys():
        raise ValueError("Quick temperature IDs do not match the complete parent fold")
    destination = Path(str(temperature["path"]))
    existed = destination.exists()
    immutable(destination, b"".join(encoded(source[id_]) for id_ in ids))
    receipt = {
        "quick_plan_sha256": digest(plan_path),
        "controls_audit_sha256": digest(controls_audit),
        "path": str(destination),
        "sha256": digest(destination),
        "rows": 700,
    }
    immutable(plan_path.with_name("quick-temperature-receipt.json"), (json.dumps(receipt, indent=2) + "\n").encode())
    if not existed:
        record("quick_evaluation_temperature_ready", **receipt)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze row order once; publish each audited training tranche atomically."
    )
    parser.add_argument("command", choices=("plan", "freeze", "quick-plan", "quick-temperature"))
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--quick-plan", type=Path, default=Path("progress/continuation/quick-eval-plan.json"))
    parser.add_argument("--generation-plan", type=Path, default=Path("data/continuation/generation-plan-v2.json.gz"))
    parser.add_argument("--original-train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--public-calibration", type=Path, default=Path("data/continuation/public-temperature.jsonl"))
    parser.add_argument("--controls-audit", type=Path, default=Path("data/continuation/controls-audit/audit.json"))
    parser.add_argument("--calibration", type=Path, default=Path("data/continuation/temperature.jsonl"))
    parser.add_argument("--tranche", type=int)
    parser.add_argument("--synthetic", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if not os.getenv("AUTOJEV_EVENTS"):
        parser.error("Set AUTOJEV_EVENTS explicitly to the continuation event log")
    if args.command == "quick-plan":
        make_quick_plan(args.quick_plan, args.generation_plan)
        print(json.dumps({"path": str(args.quick_plan), "sha256": digest(args.quick_plan)}))
    elif args.command == "quick-temperature":
        print(materialize_quick_temperature(args.quick_plan, args.controls_audit))
    elif args.command == "plan":
        if args.schedule is None:
            parser.error("plan requires --schedule")
        if args.schedule.exists():
            parser.error("Schedule already exists; refusing to replace it")
        schedule = make_schedule(
            args.generation_plan,
            args.original_train,
            args.public_calibration,
            args.controls_audit,
            args.calibration,
            args.schedule,
        )
        payload = json.dumps(schedule, ensure_ascii=False, separators=(",", ":")).encode()
        immutable(args.schedule, gzip.compress(payload, mtime=0) if args.schedule.suffix == ".gz" else payload)
        load_schedule(args.schedule)
        record(
            "training_schedule_frozen",
            path=str(args.schedule),
            sha256=digest(args.schedule),
            rows=TOTAL_ROWS,
            synthetic_rows=50000,
            replay_rows=19200,
            tranches=TRANCHES,
            global_batch=256,
            updates=271,
        )
        print(
            json.dumps(
                {"path": str(args.schedule), "sha256": digest(args.schedule), "rows": TOTAL_ROWS, "updates": 271}
            )
        )
    else:
        if args.schedule is None or args.tranche is None or args.synthetic is None or args.audit is None:
            parser.error("freeze requires --schedule, --tranche, --synthetic and --audit")
        print(json.dumps(freeze_tranche(args.schedule, args.tranche, args.synthetic, args.audit), indent=2))


if __name__ == "__main__":
    main()
