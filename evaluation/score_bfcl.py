"""CPU-only rescore of published predictions; this does not run model inference.

Scope: Decision Index 0.2.1 BFCL tool-name selection projection. A case is
correct only when every candidate-tool yes/no choice is correct. Arguments,
call order and tool execution are outside this evaluation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

HERE = Path(__file__).resolve().parent
QWEN = "Qwen/Qwen3.5-0.8B"
DEV = "dev-1.0"
MODELS = (QWEN, DEV)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_records(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _empty() -> dict[str, int]:
    return {key: 0 for key in ("cases", "correct_cases", "fields", "tp", "fp", "fn", "tn")}


def _metrics(counts: dict[str, int]) -> dict[str, Any]:
    result = dict(counts)
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    result.update(
        case_exact_accuracy=ratio(counts["correct_cases"], counts["cases"]),
        field_accuracy=ratio(counts["tp"] + counts["tn"], counts["fields"]),
        positive_recall=ratio(counts["tp"], counts["tp"] + counts["fn"]),
        positive_precision=ratio(counts["tp"], counts["tp"] + counts["fp"]),
        false_positive_rate=ratio(counts["fp"], counts["fp"] + counts["tn"]),
    )
    return result


def validate_record(record: dict[str, Any]) -> None:
    if record.get("schema") != "dev-bfcl-predictions-1" or record.get("catalog_id") != 1:
        raise ValueError("Unknown prediction schema or benchmark catalog")
    if not isinstance(record.get("id"), str) or not isinstance(record.get("request_id"), str):
        raise ValueError("Every case needs source and request IDs")
    if not re.fullmatch(r"[0-9a-f]{64}", record.get("payload_sha256", "")):
        raise ValueError("Every case needs its original payload SHA256")
    fields = record.get("field_order")
    tools = record.get("candidate_tools")
    if not isinstance(fields, list) or not fields or len(set(fields)) != len(fields):
        raise ValueError("Candidate field order must be nonempty and unique")
    if not isinstance(tools, list) or len(tools) != len(fields):
        raise ValueError("Candidate tool order must cover every field")
    rebuilt = ["tool_" + re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") for name in tools]
    if rebuilt != fields:
        raise ValueError("Candidate tool names and field order differ")
    if set(record.get("expected", {})) != set(fields) or set(record.get("option_order", {})) != set(fields):
        raise ValueError("Expected labels and option order must cover every candidate")
    if set(record.get("answers", {})) != set(MODELS):
        raise ValueError("Require exactly the two published models on the same cases")
    for model in MODELS:
        if set(record["answers"][model]) != set(fields):
            raise ValueError("Model answers omit or add a candidate field")
    for field in fields:
        options = record["option_order"][field]
        if not isinstance(options, list) or len(options) != 2 or set(options) != {"yes", "no"}:
            raise ValueError("Each candidate needs its original yes/no option order")
        if record["expected"][field] not in options:
            raise ValueError("Expected label is outside the candidate options")
        for model in MODELS:
            answer = record["answers"][model][field]
            if answer.get("type") != "choice" or answer.get("choice") not in options:
                raise ValueError("Every model answer must be a valid choice")
            probabilities = answer.get("probabilities", {})
            if set(probabilities) != set(options):
                raise ValueError("Probability support omits or adds an option")
            values = [probabilities[key] for key in options]
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                raise ValueError("Probabilities must be finite numbers in [0, 1]")
            if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-6):
                raise ValueError("Probability mass must sum to one")
            if answer["choice"] != max(options, key=probabilities.__getitem__):
                raise ValueError("Saved choice does not match its original option-order argmax")


def score_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot score an empty prediction artifact")
    totals = {model: _empty() for model in MODELS}
    tracks = defaultdict(lambda: {model: _empty() for model in MODELS})
    paired = {key: 0 for key in ("both_correct", "dev_only_correct", "qwen_only_correct", "both_wrong")}
    seen_ids, seen_requests = set(), set()
    for record in records:
        validate_record(record)
        if record["id"] in seen_ids or record["request_id"] in seen_requests:
            raise ValueError("Duplicate case or request ID")
        seen_ids.add(record["id"])
        seen_requests.add(record["request_id"])
        correct = {}
        for model in MODELS:
            exact = True
            for counts in (totals[model], tracks[record["track"]][model]):
                counts["cases"] += 1
                for field in record["field_order"]:
                    gold = record["expected"][field]
                    selected = record["answers"][model][field]["choice"]
                    counts["fields"] += 1
                    counts[{("yes", "yes"): "tp", ("no", "yes"): "fp", ("yes", "no"): "fn", ("no", "no"): "tn"}[(gold, selected)]] += 1
                matches = all(record["answers"][model][field]["choice"] == record["expected"][field] for field in record["field_order"])
                counts["correct_cases"] += int(matches)
                exact = matches
            correct[model] = exact
        key = "both_correct" if all(correct.values()) else "dev_only_correct" if correct[DEV] else "qwen_only_correct" if correct[QWEN] else "both_wrong"
        paired[key] += 1
    return {
        "schema": "dev-bfcl-saved-predictions-rescore-1",
        "scope": "DI 0.2.1 BFCL tool-name selection projection; not full BFCL leaderboard",
        "new_inference_executed": False,
        "training_data": False,
        "metric": "case exact accuracy: every candidate-tool field must match",
        "models": {model: _metrics(value) for model, value in totals.items()},
        "tracks": {track: {model: _metrics(value) for model, value in group.items()} for track, group in sorted(tracks.items())},
        "paired_cases": paired,
        "same_cases_for_both_models": True,
    }


def verify_expected(summary: dict[str, Any], manifest: dict[str, Any]) -> None:
    for model, expected in manifest["expected_counts"].items():
        measured = summary["models"][model]
        for key, value in expected.items():
            if measured[key] != value:
                raise ValueError(f"Published count mismatch: {model} / {key}")
    if summary["paired_cases"] != manifest["expected_paired_cases"]:
        raise ValueError("Published paired-case counts differ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=HERE / "bfcl-predictions.jsonl.gz")
    parser.add_argument("--manifest", type=Path, default=HERE / "bfcl-predictions-manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="New JSON summary output")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new summary output path")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if sha256(args.predictions) != manifest["artifact_sha256"]:
        raise ValueError("Published prediction artifact SHA256 differs")
    summary = score_records(load_records(args.predictions))
    verify_expected(summary, manifest)
    summary["prediction_artifact_sha256"] = manifest["artifact_sha256"]
    summary["source_fingerprints"] = manifest["source_fingerprints"]
    summary["published_counts_verified"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "cases": summary["models"][DEV]["cases"], "qwen_correct": summary["models"][QWEN]["correct_cases"], "dev_correct": summary["models"][DEV]["correct_cases"], "new_inference_executed": False}))


if __name__ == "__main__":
    main()
