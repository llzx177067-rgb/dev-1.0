"""Independently reread frozen derivatives and verify exact retained semantics."""
import argparse
import gzip
import hashlib
import itertools
import json
import unicodedata
from collections import Counter
from pathlib import Path


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_jsonl(path):
    with path.open("rb") as stream:
        for line in stream:
            yield json.loads(line)


def semantic_bytes(row):
    # This function intentionally has no dependency on prepare_data.py.
    body = dict(row)
    body["source"] = dict(row["source"])
    body["source"].pop("input_tokens", None)
    return (json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--data", type=Path, default=Path("data/qwen35-final"))
    args = parser.parse_args()
    root = args.root.resolve()
    folder = args.data if args.data.is_absolute() else root / args.data
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed_full_corpus":
        raise ValueError("Manifest not complete")
    ledger = folder / "quarantine-ledger.jsonl"
    if file_sha(ledger) != manifest["quarantine_ledger_sha256"]:
        raise ValueError("Quarantine ledger hash mismatch")
    excluded = {row["id"]: row for row in read_jsonl(ledger)}
    if len(excluded) != manifest["quarantine_rows"]:
        raise ValueError("Quarantine ID/count mismatch")
    id_diagnostics = folder / manifest["split_local_id_false_positive_ledger"]["file"]
    if file_sha(id_diagnostics) != manifest["split_local_id_false_positive_ledger"]["sha256"]:
        raise ValueError("Split-local ID diagnostic ledger SHA mismatch")
    if sum(1 for _ in read_jsonl(id_diagnostics)) != manifest["split_local_id_false_positive_ledger"]["rows"]:
        raise ValueError("Split-local ID diagnostic count mismatch")
    fixture = root / "audit/qwen35-08b/qlora-length-smoke-input.jsonl"
    fixture_semantics = {row["id"]: semantic_bytes(row) for row in read_jsonl(fixture)} if fixture.exists() else {}
    fixture_seen = set()
    results = {}
    seen_excluded = set()
    aliases = manifest["source_aliases"]

    def family_identity(source, family):
        normalized = " ".join(unicodedata.normalize("NFKC", source).casefold().split())
        return aliases.get(normalized, normalized), family.removeprefix("tasksource:")

    excluded_families = {family_identity(row["canonical_source"], row["family"]) for row in excluded.values()}
    if len(excluded_families) != manifest["quarantine_families"]:
        raise ValueError("Quarantine family count mismatch")
    for fold, details in manifest["folds"].items():
        original = root / manifest["inputs"][fold]["path"]
        output = folder / f"{fold}.jsonl"
        audit = folder / details["length_sidecar"]
        if file_sha(original) != manifest["inputs"][fold]["sha256"]:
            raise ValueError("Input SHA mismatch: " + fold)
        if file_sha(output) != details["output_sha256"] or file_sha(audit) != details["length_sidecar_sha256"]:
            raise ValueError("Output/sidecar SHA mismatch: " + fold)
        measured_stream = gzip.open(audit, "rt", encoding="utf-8")

        def expected_retained():
            for row, raw in itertools.zip_longest(read_jsonl(original), measured_stream):
                if row is None or raw is None:
                    raise ValueError("Source/length record count mismatch")
                measured = json.loads(raw)
                if measured["id"] != row["id"]:
                    raise ValueError("Sidecar ID/order mismatch")
                if fold == "train" and row["id"] in excluded:
                    if excluded[row["id"]]["semantic_sha256"] != hashlib.sha256(semantic_bytes(row)).hexdigest():
                        raise ValueError("Quarantine semantic mismatch")
                    seen_excluded.add(row["id"])
                    continue
                yield row, measured

        count = token_total = max_length = image_count = 0
        input_digest = hashlib.sha256()
        output_digest = hashlib.sha256()
        source_counts, type_counts = Counter(), Counter()
        families = set()
        for before, after in itertools.zip_longest(expected_retained(), read_jsonl(output)):
            if before is None or after is None:
                raise ValueError("Retained input/output row count mismatch")
            expected, measured = before
            if after["id"] != expected["id"] or after["source"]["input_tokens"] != measured["input_tokens"]:
                raise ValueError("Output ID/order/length mismatch")
            family = family_identity(after["source"]["dataset"], after["family"])
            if fold == "train" and family in excluded_families:
                raise ValueError("A quarantined family remains in train")
            families.add(family)
            expected_payload, output_payload = semantic_bytes(expected), semantic_bytes(after)
            if expected_payload != output_payload:
                raise ValueError("Unexpected semantic/order mutation: " + expected["id"])
            if after["id"] in fixture_semantics:
                if fixture_semantics[after["id"]] != output_payload:
                    raise ValueError("GPU fixture semantic mutation")
                fixture_seen.add(after["id"])
            if expected["question"]["type"] == "choice" and list(expected["question"]["criteria"].items()) != list(after["question"]["criteria"].items()):
                raise ValueError("Criteria insertion order changed")
            input_digest.update(expected_payload)
            output_digest.update(output_payload)
            count += 1
            token_total += measured["input_tokens"]
            max_length = max(max_length, measured["input_tokens"])
            image_count += bool(after.get("images"))
            source_counts[after["source"]["dataset"]] += 1
            type_counts[after["question"]["type"]] += 1
        measured_stream.close()
        checks = {
            "retained_rows": count == details["output_rows"],
            "exact_semantics_and_order": input_digest.digest() == output_digest.digest(),
            "input_semantic_sha256": input_digest.hexdigest() == details["retained_input_semantic_ordered_sha256"],
            "output_semantic_sha256": output_digest.hexdigest() == details["output_semantic_ordered_sha256"],
            "length_total": token_total == details["output_input_tokens"],
            "length_max": max_length == details["output_maximum_input_tokens"] <= manifest["max_length"],
            "images": image_count == details["output_image_rows"],
            "source_counts": dict(source_counts) == details["source_counts"],
            "type_counts": dict(type_counts) == details["type_counts"],
            "family_counts": len(families) == details["families"],
            "all_holdout_rows_preserved": fold == "train" or count == manifest["inputs"][fold]["expected_rows"],
        }
        if not all(checks.values()):
            raise ValueError(f"Verification failed {fold}: {checks}")
        results[fold] = {"rows": count, "checks": checks, "semantic_sha256": output_digest.hexdigest()}
        print(json.dumps({"fold": fold, "rows": count, "passed": True}), flush=True)
    if seen_excluded != set(excluded):
        raise ValueError("Quarantine ledger contains absent records")
    if fixture_seen != set(fixture_semantics):
        raise ValueError("Some GPU fixture IDs are absent from final data")
    result = {"status": "passed", "scope": "Full independent input/output reread; identity, target, labels, criteria order, retained ordering, metadata lengths and counts; not a second tokenizer execution", "manifest_sha256": file_sha(folder / "manifest.json"), "verifier_sha256": file_sha(Path(__file__)), "quarantine_ids_verified": len(seen_excluded), "candidate_fixture_rows_retained_and_semantically_equal": len(fixture_seen), "folds": results}
    destination = folder / "independent-verification.json"
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
