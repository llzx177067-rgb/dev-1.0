"""Freeze a Qwen3.5-specific derivative without changing decision semantics.

Runs without model weights. All original evaluation folds and all 30,000 raw
Tasksource evaluation rows are protected. State-only matches are diagnostic,
never a removal rule. Only source.input_tokens is changed in retained rows.
"""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import gzip
import hashlib
import io
import itertools
import json
import math
import multiprocessing
import os
import string
import sys
import time
import unicodedata
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


EXPECTED = {
    "train": (626547, "bc284a31801a8711e5a459a4731c6220b44b1a9ce607c7f7a506e3d22ade0339"),
    "dev": (1497, "ebcedc3e726548c08da90ff569b94e4822f23312dfc3ea29b93579cfec3016c7"),
    "temperature": (983, "e735fa4e0df3482752112bc9cdff6af9ddb7be07a0a9f2fb8dd8cf2355042bf6"),
    "known": (3247, "dedf9f7d526efe6f0cda550e7cf4dccd5e6aa9a0f9a8644e6b41db1fdfa96c59"),
    "test": (4994, "e26f88d01eacdfa5d4f927268545c3341f551cc93e77c01ce7b05c9fd7930a80"),
    "image_test": (200, "1d2e0bc3c905b3a5b3319eb7122dcf9902db29814f868e4024d569509623fbac"),
}
RAW_EVAL_SHA = {
    "validation": "0e901246f91cf3a869b66bd8fb7a404a438472933868b76838c0f3bdcb64f2f3",
    "test": "a5f367eaaef12fa78ce943e0bd00d9bbde537c841d64758ca99a19b01a5a2917",
}
SOURCE_ALIASES = {
    "super_glue/boolq": "boolq", "google/boolq": "boolq",
    "google-research-datasets/paws": "paws", "paws/labeled_final": "paws",
    "glue/mnli": "multinli", "multi_nli": "multinli", "mnli": "multinli",
    "rajpurkar/squad_v2": "squad2", "squad_v2": "squad2",
    "google/civil_comments": "civil_comments",
    "tals/vitaminc": "vitaminc", "vitaminc-dev": "vitaminc",
    "amazonscience/massive": "massive",
    "nvidia/aegis-ai-content-safety-dataset-2.0": "aegis2",
}
MODEL_SHA = "824721da4d8f8ba1e534f3ee35081869243c4457f6b3bd0a9a91b611b2ec1076"
REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
PROCESSOR = None
CODES = None
DECISION_MESSAGES = None
OPEN_IMAGE = None


def compact(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def normal(value):
    if isinstance(value, str):
        return norm(value)
    if isinstance(value, dict):
        return {key: normal(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [normal(item) for item in value]
    return value


def digest(value):
    return hashlib.sha256(compact(value).encode()).hexdigest()


def rows(path):
    with Path(path).open("rb", buffering=8 * 1024 * 1024) as stream:
        for line in stream:
            yield json.loads(line)


def source(row):
    value = row["source"]
    return value if isinstance(value, str) else value.get("dataset", row.get("suite", "unknown"))


def canonical_source(row):
    value = norm(source(row))
    return SOURCE_ALIASES.get(value, value)


def family_key(row):
    value = row.get("family", row.get("group_id"))
    if value is None:
        raise ValueError("Missing family/group_id")
    if value.startswith("tasksource:"):
        value = value[len("tasksource:"):]
    return canonical_source(row), value


def raw_id_keys(row):
    result = {("global-row-id", row["id"].removeprefix("tasksource-reconstructed:"))}
    src = row["source"]
    if isinstance(src, dict) and src.get("upstream_id"):
        # Several originals use split-local numeric row IDs. Config and split
        # are part of their namespace, not evidence of cross-split leakage.
        config = src.get("config", src.get("subset", src.get("locale", "unspecified")))
        result.add((canonical_source(row), config, src.get("split", row.get("split", "unspecified")), src["upstream_id"]))
    else:
        result.add((canonical_source(row), row["id"]))
    return result


def request_key(row):
    """Task-aware, permutation-invariant complete request; no target required."""
    if isinstance(row["question"], dict):
        q = row["question"]
        kind = q["type"]
        instruction = q.get("instructions") or "Choose the best matching option."
        if kind == "noul":
            criteria = q.get("criteria") or {}
            opts = [criteria.get("false") or "No / false", criteria.get("true") or "Yes / true"]
        elif kind == "score":
            kind, opts = "choice", q["criteria"]
        else:
            opts = [key if value is None else value for key, value in q["criteria"].items()]
    else:
        kind = "choice" if row["kind"] == "score" else row["kind"]
        instruction = row["question"] or "Choose the best matching option."
        opts = ["No / false", "Yes / true"] if kind == "noul" else row["options"]
    normalized_options = [compact(normal(value)) for value in opts]
    # Boolean outcomes have fixed false/true semantics; do not swap them.
    if kind != "noul":
        normalized_options.sort()
    image_identity = row.get("source", {}).get("image_sha256") if isinstance(row.get("source"), dict) else None
    return digest([normal(row["state"]), kind, normal(instruction), normalized_options, image_identity])


def components(row):
    state = row["state"]
    values = [state] if isinstance(state, str) else [x for x in state.values() if isinstance(x, str)] if isinstance(state, dict) else []
    return {hashlib.sha256(norm(value).encode()).hexdigest() for value in values if len(norm(value)) >= 80}


def semantic(row):
    result = copy.deepcopy(row)
    if isinstance(result.get("source"), dict):
        result["source"].pop("input_tokens", None)
    return compact(result).encode() + b"\n"


def worker_init(root, metadata):
    global PROCESSOR, CODES, DECISION_MESSAGES, OPEN_IMAGE
    os.chdir(root)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    torch.set_num_threads(1)
    path = Path(root) / "reproduction/source/src/autojev/model.py"
    if sha_file(path) != MODEL_SHA:
        raise ValueError("Released prompt source hash changed")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"describe", "options", "decision_messages", "open_image"}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    # Postpone the upstream type hints; only four unmodified pure prompt/image functions.
    scope = {"json": json, "base64": base64, "io": io, "Image": Image, "MAX_OPTIONS": 255}
    module = ast.Module(body=body, type_ignores=[])
    exec(compile(module, str(path), "exec", flags=__import__("__future__").annotations.compiler_flag), scope)
    DECISION_MESSAGES, OPEN_IMAGE = scope["decision_messages"], scope["open_image"]
    PROCESSOR = AutoProcessor.from_pretrained(metadata, local_files_only=True)
    if not PROCESSOR.chat_template:
        PROCESSOR.chat_template = PROCESSOR.tokenizer.chat_template
    if not PROCESSOR.chat_template:
        raise ValueError("No Base chat template")
    PROCESSOR.tokenizer.padding_side = "left"
    PROCESSOR.image_processor.size = {"shortest_edge": 65536, "longest_edge": 262144}
    possibilities = list(string.ascii_uppercase) + ["".join(p) for p in itertools.product(string.ascii_uppercase, repeat=2)]
    CODES = [code for code in possibilities if len(PROCESSOR.tokenizer.encode(code, add_special_tokens=False)) == 1][:255]
    if len(CODES) != 255:
        raise ValueError("Invalid answer vocabulary")


def measure(batch):
    texts = [PROCESSOR.apply_chat_template(DECISION_MESSAGES(row, CODES), tokenize=False, add_generation_prompt=True, enable_thinking=False) for row in batch]
    images = [OPEN_IMAGE(image) for row in batch for image in row.get("images", [])]
    encoded = PROCESSOR(text=texts, images=images or None, padding=False, truncation=False)["input_ids"]
    return [len(ids) for ids in encoded]


def batches(path, size):
    pending = []
    for row in rows(path):
        if row.get("images"):
            if pending:
                yield pending
                pending = []
            yield [row]
        else:
            pending.append(row)
            if len(pending) == size:
                yield pending
                pending = []
    if pending:
        yield pending


def bounded_measure(executor, batches_input, workers):
    pending = deque()
    for batch in batches_input:
        pending.append((batch, executor.submit(measure, batch)))
        if len(pending) >= workers * 2:
            original, future = pending.popleft()
            yield original, future.result()
    while pending:
        original, future = pending.popleft()
        yield original, future.result()


def reuse_measure(path, size, cached_sidecar):
    with gzip.open(cached_sidecar, "rt", encoding="utf-8") as stream:
        for batch in batches(path, size):
            lengths = []
            for row in batch:
                raw = stream.readline()
                if not raw:
                    raise ValueError("Cached length sidecar ended early")
                record = json.loads(raw)
                if record["id"] != row["id"] or record["semantic_sha256"] != hashlib.sha256(semantic(row)).hexdigest():
                    raise ValueError("Cached sidecar input/order/semantic identity mismatch")
                lengths.append(record["input_tokens"])
            yield batch, lengths
        if stream.readline():
            raise ValueError("Cached length sidecar has extra records")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path("data/qwen35-final"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--reuse-lengths", type=Path, help="Reuse complete exact sidecars from an interrupted export; every ID/semantic SHA is reverified")
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output if args.output.is_absolute() else root / args.output
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    if args.workers < 1 or args.batch_size < 1:
        parser.error("workers and batch-size must be positive")
    os.chdir(root)
    paths = {fold: root / ("data/reconstructed-candidate/train.jsonl" if fold == "train" else f"data/original-rebuilt/{fold}.jsonl") for fold in EXPECTED}
    inputs = {}
    for fold, path in paths.items():
        actual = sha_file(path)
        if actual != EXPECTED[fold][1]:
            raise ValueError(f"Input identity mismatch {fold}: {actual}")
        inputs[fold] = {"path": str(path.relative_to(root)), "sha256": actual, "expected_rows": EXPECTED[fold][0]}
    raw_paths = {fold: root / f"data/tasksource/historical-d2ab1d/full/{fold}-00000-of-00001.parquet" for fold in RAW_EVAL_SHA}
    for fold, path in raw_paths.items():
        actual = sha_file(path)
        if actual != RAW_EVAL_SHA[fold]:
            raise ValueError(f"Raw heldout identity mismatch {fold}")
        inputs[f"tasksource_raw_{fold}"] = {"path": str(path.relative_to(root)), "sha256": actual, "expected_rows": 15000}
    print(compact({"phase": "input_hashes_verified", "inputs": inputs}), flush=True)
    out.mkdir(parents=True)
    metadata = root / "audit/qwen35-08b/Qwen3.5-0.8B-Base"
    metadata_hashes = {p.name: sha_file(p) for p in sorted(metadata.iterdir()) if p.is_file() and p.suffix in {".json", ".jinja", ".txt"}}
    metadata_identity = json.loads((metadata / "metadata-manifest.json").read_text(encoding="utf-8"))
    if metadata_identity["repo"] != "Qwen/Qwen3.5-0.8B-Base" or metadata_identity["revision"] != REVISION:
        raise ValueError("Processor repository/revision mismatch")
    for item in metadata_identity["files"]:
        if sha_file(metadata / item["file"]) != item["sha256"]:
            raise ValueError(f"Processor file identity mismatch: {item['file']}")
    holdout_index = {key: defaultdict(list) for key in ("id", "family", "request", "component", "image", "legacy_splitless_upstream")}
    holdout_counts = Counter()

    def index_holdout(row, fold):
        ref = {"id": row["id"], "fold": fold, "source": source(row), "family": row.get("family", row.get("group_id"))}
        for key in raw_id_keys(row):
            holdout_index["id"][key].append(ref)
        if isinstance(row["source"], dict) and row["source"].get("upstream_id"):
            holdout_index["legacy_splitless_upstream"][(canonical_source(row), row["source"]["upstream_id"])].append(ref)
        holdout_index["family"][family_key(row)].append(ref)
        holdout_index["request"][request_key(row)].append(ref)
        for key in components(row):
            holdout_index["component"][key].append(ref)
        image = row.get("source", {}).get("image_sha256") if isinstance(row.get("source"), dict) else None
        if image:
            holdout_index["image"][image].append(ref)
        holdout_counts[fold] += 1

    for fold, path in paths.items():
        if fold != "train":
            for row in rows(path):
                index_holdout(row, fold)
    import pyarrow.parquet as pq
    for fold, path in raw_paths.items():
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                index_holdout(row, "tasksource_raw_" + fold)
    for fold, n in holdout_counts.items():
        expected = 15000 if fold.startswith("tasksource_raw_") else EXPECTED[fold][0]
        if n != expected:
            raise ValueError(f"Heldout row count mismatch {fold}")
    family_reasons = defaultdict(list)
    diagnostics = Counter()
    summaries = {}
    false_positive_path = out / "split-local-id-false-positives.jsonl"
    start = time.monotonic()
    with false_positive_path.open("x", encoding="utf-8", newline="\n", buffering=8*1024*1024) as false_positive_stream, ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"), initializer=worker_init, initargs=(str(root), str(metadata))) as pool:
        for fold, path in paths.items():
            histogram = Counter()
            total = images = n = 0
            max_n = 0
            sem_hash = hashlib.sha256()
            ids = set()
            sidecar = out / f"{fold}.lengths.jsonl.gz"
            with gzip.open(sidecar, "xt", encoding="utf-8", newline="\n") as target:
                measured_batches = reuse_measure(path, args.batch_size, args.reuse_lengths / sidecar.name) if args.reuse_lengths else bounded_measure(pool, batches(path, args.batch_size), args.workers)
                for batch, lengths in measured_batches:
                    for row, length in zip(batch, lengths, strict=True):
                        if row["id"] in ids:
                            raise ValueError(f"Duplicate ID in {fold}: {row['id']}")
                        ids.add(row["id"])
                        n += 1
                        total += length
                        max_n = max(max_n, length)
                        images += bool(row.get("images"))
                        histogram[length] += 1
                        sem_hash.update(semantic(row))
                        target.write(compact({"id": row["id"], "family": row["family"], "source": source(row), "input_tokens": length, "semantic_sha256": hashlib.sha256(semantic(row)).hexdigest()}) + "\n")
                        if fold == "train":
                            key = family_key(row)
                            reasons = []
                            upstream = row["source"].get("upstream_id")
                            legacy = holdout_index["legacy_splitless_upstream"].get((canonical_source(row), upstream), [])
                            strict_matches = [ref for identity in raw_id_keys(row) for ref in holdout_index["id"].get(identity, [])]
                            if legacy and not strict_matches:
                                diagnostics["split_local_upstream_id_false_positive_rows"] += 1
                                diagnostics["split_local_upstream_id_false_positive_source:" + source(row)] += 1
                                false_positive_stream.write(compact({"id": row["id"], "source": source(row), "split": row["source"].get("split"), "upstream_id": upstream, "family": row["family"], "same_number_different_namespace_heldout": legacy, "action": "not_quarantined_for_splitless_id_collision"}) + "\n")
                            for criterion, keys in (("id", raw_id_keys(row)), ("family", [key]), ("request", [request_key(row)])):
                                found = {ref["fold"] + ":" + ref["id"]: ref for k in keys for ref in holdout_index[criterion].get(k, [])}
                                if found:
                                    reasons.append({"reason": "heldout_" + criterion, "trigger_train_id": row["id"], "heldout": list(found.values())})
                            image = row.get("source", {}).get("image_sha256")
                            if image and image in holdout_index["image"]:
                                reasons.append({"reason": "heldout_image_sha256", "trigger_train_id": row["id"], "heldout": holdout_index["image"][image]})
                            matches = {ref["fold"] + ":" + ref["id"]: ref for component in components(row) for ref in holdout_index["component"].get(component, [])}
                            if matches:
                                diagnostics["long_component_train_rows"] += 1
                            # A documented long Aegis injection spans otherwise different tasks.
                            # Conservatively protect that complete source family, with exact refs.
                            if canonical_source(row) == "aegis2" and matches:
                                reasons.append({"reason": "aegis_long_component_shared_with_heldout", "trigger_train_id": row["id"], "heldout": list(matches.values())})
                            if length > args.max_length:
                                reasons.append({"reason": "over_max_length", "trigger_train_id": row["id"], "input_tokens": length})
                            if reasons:
                                family_reasons[key].extend(reasons)
                        elif length > args.max_length:
                            raise ValueError(f"Evaluation row exceeds max length, never silently dropped: {fold}/{row['id']} {length}")
                    if n % 25000 < len(batch):
                        print(compact({"phase": "length_audit", "fold": fold, "rows": n, "max": max_n, "elapsed_seconds": round(time.monotonic()-start, 1)}), flush=True)
            if n != EXPECTED[fold][0]:
                raise ValueError(f"Row count mismatch {fold}: {n}")
            summaries[fold] = {"input_rows": n, "image_rows": images, "total_input_tokens": total, "maximum_input_tokens": max_n, "length_histogram": dict(sorted(histogram.items())), "input_semantic_ordered_sha256": sem_hash.hexdigest(), "length_sidecar": sidecar.name, "length_sidecar_sha256": sha_file(sidecar), "over_max_length_rows": sum(count for length,count in histogram.items() if length > args.max_length), "over_8128_rows": sum(count for length,count in histogram.items() if length > 8128)}
            print(compact({"phase": "fold_audited", "fold": fold, "rows": n, "max": max_n, "total_tokens": total}), flush=True)
    family_reasons = {key: values for key, values in family_reasons.items() if values}
    quarantined_families = set(family_reasons)
    quarantine_rows = Counter()
    quarantine_reasons = Counter()
    quarantine_path = out / "quarantine-ledger.jsonl"
    verify_result = {}
    with quarantine_path.open("x", encoding="utf-8", newline="\n") as quarantine:
        for fold, path in paths.items():
            destination = out / f"{fold}.jsonl"
            count = total_tokens = image_count = maximum = 0
            source_counts, type_counts, family_counts = Counter(), Counter(), set()
            selected_semantic, output_semantic = hashlib.sha256(), hashlib.sha256()
            with gzip.open(out / f"{fold}.lengths.jsonl.gz", "rt", encoding="utf-8") as audit, destination.open("xb", buffering=8*1024*1024) as target:
                for row, raw_length in itertools.zip_longest(rows(path), audit):
                    if row is None or raw_length is None:
                        raise ValueError("Length sidecar row mismatch")
                    measured = json.loads(raw_length)
                    if measured["id"] != row["id"] or measured["semantic_sha256"] != hashlib.sha256(semantic(row)).hexdigest():
                        raise ValueError("Length sidecar order/semantic mismatch")
                    if fold == "train" and family_key(row) in quarantined_families:
                        reasons = family_reasons[family_key(row)]
                        quarantine.write(compact({"id": row["id"], "family": row["family"], "canonical_source": canonical_source(row), "input_tokens": measured["input_tokens"], "semantic_sha256": measured["semantic_sha256"], "reasons": reasons}) + "\n")
                        quarantine_rows[source(row)] += 1
                        for reason in {value["reason"] for value in reasons}:
                            quarantine_reasons[reason] += 1
                        continue
                    original = semantic(row)
                    selected_semantic.update(original)
                    row["source"]["input_tokens"] = measured["input_tokens"]
                    after = semantic(row)
                    if original != after:
                        raise ValueError("Unexpected semantic mutation")
                    output_semantic.update(after)
                    target.write(compact(row).encode() + b"\n")
                    count += 1
                    total_tokens += measured["input_tokens"]
                    maximum = max(maximum, measured["input_tokens"])
                    image_count += bool(row.get("images"))
                    source_counts[source(row)] += 1
                    type_counts[row["question"]["type"]] += 1
                    family_counts.add(family_key(row))
            summaries[fold].update({"output_rows": count, "output_sha256": sha_file(destination), "output_bytes": destination.stat().st_size, "output_input_tokens": total_tokens, "output_maximum_input_tokens": maximum, "output_image_rows": image_count, "source_counts": dict(source_counts), "type_counts": dict(type_counts), "families": len(family_counts), "retained_input_semantic_ordered_sha256": selected_semantic.hexdigest(), "output_semantic_ordered_sha256": output_semantic.hexdigest(), "semantic_identity_verified": selected_semantic.digest() == output_semantic.digest()})
            verify_result[fold] = {"rows": count, "semantic_sha256": output_semantic.hexdigest(), "passed": selected_semantic.digest() == output_semantic.digest()}
    manifest = {
        "format_version": 1, "status": "completed_full_corpus", "created_date": "2026-10-12", "base_model": "Qwen/Qwen3.5-0.8B-Base", "base_revision": REVISION,
        "official_training_set": False, "parent_candidate_rows": 626547, "inputs": inputs, "processor_files_sha256": metadata_hashes,
        "prompt_source_sha256": MODEL_SHA, "prepare_script_sha256": sha_file(Path(__file__)),
        "max_length": args.max_length, "truncation": False, "augmentation_included": False, "weights_loaded": False,
        "workers": args.workers, "batch_size": args.batch_size, "folds": summaries, "holdout_protected_rows": dict(holdout_counts),
        "quarantine_rows": sum(quarantine_rows.values()), "quarantine_families": len(quarantined_families), "quarantine_source_counts": dict(quarantine_rows), "quarantine_reason_row_counts": dict(quarantine_reasons), "quarantine_ledger_sha256": sha_file(quarantine_path),
        "source_aliases": SOURCE_ALIASES, "diagnostics": dict(diagnostics), "nominal_updates_batch256_epoch1": math.ceil(summaries["train"]["output_rows"] / 256),
        "reused_exact_length_sidecars_from": str(args.reuse_lengths) if args.reuse_lengths else None,
        "split_local_id_false_positive_ledger": {"file": false_positive_path.name, "sha256": sha_file(false_positive_path), "rows": diagnostics["split_local_upstream_id_false_positive_rows"], "rule": "upstream IDs are namespaced by normalized dataset, config when available, split and upstream_id; splitless numerical collisions are diagnostics only"},
        "modifications_to_retained_rows": ["source.input_tokens replaced/added with exact new Base processor length"],
        "rules": ["Protect all five original holdout folds and all 30000 raw Tasksource validation/test rows, even quality-dropped rows", "Quarantine whole normalized (source,family) group for exact ID, upstream ID, normalized family, complete task-aware request or image-hash match", "Complete request includes normalized state, task type, instructions, option-description multiset (fixed false/true for noul), image identity; no state-only deletion", "Conservatively quarantine documented Aegis long component exact overlap, minimum 80 normalized characters", "Quarantine complete family if any unaugmented input exceeds max_length"],
        "limitations": ["Not proof against paraphrase, translation, embedded/packed examples, all source-alias forms, pretraining contamination, or changed upstream row namespaces", "Input token counts are exact for unaugmented prompts; runtime must enforce exact lengths after option shuffling", "No forced count matching or task-specific relabeling; unknown original 70-row audit and Tasksource private selection remain unrecovered", "This script protects evaluation identity but does not use evaluation targets to select training rows", "Evaluation folds retain every row; only source.input_tokens metadata changes", "Intra-training complete-request duplicates are measured separately if needed; not automatically dropped by this holdout-protection pass"],
        "elapsed_seconds": round(time.monotonic()-start, 3), "semantic_verification": verify_result,
    }
    import transformers, tokenizers, torch
    manifest["runtime"] = {"python": sys.version, "transformers": transformers.__version__, "tokenizers": tokenizers.__version__, "torch": torch.__version__}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(compact({"phase": "complete", "train_rows": summaries["train"]["output_rows"], "quarantine_rows": manifest["quarantine_rows"], "max_tokens": summaries["train"]["output_maximum_input_tokens"], "output_sha256": summaries["train"]["output_sha256"], "manifest": str(out / "manifest.json")}), flush=True)


if __name__ == "__main__":
    main()
