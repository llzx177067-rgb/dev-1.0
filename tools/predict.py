"""Load the unmodified dev-1.0 adapter checkpoint and its decision readout.

The original trainer implements the NF4 backbone, attention hook, preprocessing,
readout and saved temperature. This entry point only verifies artifacts, freezes
the loaded model, and delegates inference to that implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "provenance" / "first-run-inference-manifest.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_files(directory: Path, identities: dict[str, dict[str, Any]]) -> None:
    directory = directory.resolve()
    for name, identity in identities.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Manifest paths must stay inside the snapshot")
        path = directory / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required snapshot file is missing: {name}")
        if path.stat().st_size != identity["bytes"] or sha256(path) != identity["sha256"]:
            raise ValueError(f"Snapshot size or SHA256 differs: {name}")


def verify_artifacts(model_dir: str | Path, base_model_path: str | Path) -> dict[str, Any]:
    manifest = read_json(PROVENANCE)
    model_dir, base_model_path = Path(model_dir), Path(base_model_path)
    verify_files(model_dir, manifest["checkpoint_files"])
    verify_files(base_model_path, manifest["base_identity"]["files"])
    saved = read_json(model_dir / "decision_config.json")
    for key, expected in manifest["decision_identity"].items():
        if saved.get(key) != expected:
            raise ValueError(f"Decision identity differs: {key}")
    return saved


def _runtime() -> tuple[Any, Any, Any]:
    # Keep artifact inspection and --help available without importing CUDA libs.
    sys.path.insert(0, str(ROOT / "training/qwen35-0.8b/runtime-source/src"))
    sys.path.insert(0, str(ROOT / "training/qwen35-0.8b"))
    import torch
    from qwen35_qlora import QuantizedDecisionModel, configure_runtime
    return torch, QuantizedDecisionModel, configure_runtime


def load_decider(
    model_dir: str | Path, base_model_path: str | Path, *, cpu_threads: int = 4,
) -> Any:
    """Load exact first-run weights on visible CUDA device 0; never merge adapters.

    Set CUDA_VISIBLE_DEVICES before starting Python to select a physical GPU.
    The supplied base snapshot must match the pinned Base revision byte for byte.
    """
    if cpu_threads < 1:
        raise ValueError("CPU threads must be positive")
    saved = verify_artifacts(model_dir, base_model_path)
    torch, model_class, configure_runtime = _runtime()
    if not torch.cuda.is_available():
        raise RuntimeError("dev-1.0 uses the original CUDA NF4 implementation; CUDA is required")
    args = argparse.Namespace(
        model_path=str(Path(base_model_path).resolve()),
        cpu_threads=cpu_threads,
        attention_mode=saved["attention_mode"],
        rank=8,
        lora_targets="all-linear",
        deterministic=True,
    )
    configure_runtime(args)
    model = model_class(args, checkpoint=str(Path(model_dir).resolve()))
    model.requires_grad_(False)
    model.eval()
    if model.temperature != saved["temperature"]:
        raise RuntimeError("Saved temperature was not restored exactly")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Inference model unexpectedly has trainable parameters")
    return model


def normalize_rows(value: Any) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else [value]
    if not rows:
        raise ValueError("Provide at least one decision")
    result = []
    for row in rows:
        if not isinstance(row, dict) or "state" not in row or not isinstance(row.get("question"), dict):
            raise ValueError("Each decision needs state and question")
        # Only public model inputs enter the original processor.
        clean = {"state": row["state"], "question": row["question"]}
        if "images" in row:
            clean["images"] = row["images"]
        result.append(clean)
    return result


def _answer(question: dict[str, Any], probabilities: list[float]) -> dict[str, Any]:
    from autojev.model import answer
    return answer(question, probabilities)


def predict_rows(model: Any, rows: Any, *, batch_size: int = 1) -> list[dict[str, Any]]:
    """Delegate forward, valid-option masking and temperature exactly once.

    Default batch size 1 is suitable for arbitrary requests up to 8192 tokens.
    Larger batches increase padded activation memory; inputs are never truncated.
    """
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    clean = normalize_rows(rows)
    # DecisionModel.predict applies its saved model.temperature once. No refit,
    # softmax repetition, threshold adjustment or option permutation is added.
    distributions = model.predict(clean, batch_size=batch_size)
    if len(distributions) != len(clean):
        raise RuntimeError("The model returned an incomplete prediction batch")
    return [_answer(row["question"], probabilities) for row, probabilities in zip(clean, distributions, strict=True)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path, help="Downloaded ModelScope dev-1.0 snapshot")
    parser.add_argument("--base-model-path", required=True, type=Path, help="Exact pinned Qwen3.5-0.8B-Base snapshot")
    parser.add_argument("--input", required=True, type=Path, help="UTF-8 JSON decision object or array")
    parser.add_argument("--output", type=Path, help="New JSON output file; omit to print to stdout")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--verify-only", action="store_true", help="CPU file/config checks, without loading GPU weights")
    args = parser.parse_args()
    rows = normalize_rows(read_json(args.input))
    if args.output and args.output.exists():
        raise FileExistsError("Output already exists; choose a new output path")
    saved = verify_artifacts(args.model_dir, args.base_model_path)
    if args.verify_only:
        output = {"status": "verified", "model": "dev-1.0", "requests": len(rows), "base_model": saved["base_model"], "base_revision": saved["revision"], "temperature": saved["temperature"], "gpu_inference_executed": False}
    else:
        model = load_decider(args.model_dir, args.base_model_path, cpu_threads=args.cpu_threads)
        predictions = predict_rows(model, rows, batch_size=args.batch_size)
        output = {"model": "dev-1.0", "base_model": saved["base_model"], "base_revision": saved["revision"], "temperature": saved["temperature"], "predictions": predictions}
    text = json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
