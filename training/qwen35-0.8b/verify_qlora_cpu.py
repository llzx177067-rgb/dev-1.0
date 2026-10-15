"""CPU-only tests for the local QLoRA data path; never loads model weights."""
import argparse
import ast
import hashlib
import itertools
import json
import random
import string
import tempfile
from pathlib import Path

import torch
from transformers import AutoProcessor
from autojev.model import DecisionModel
from autojev.exact_batch import collate_cpu, exact_microbatches
from autojev.evaluate import read_rows
from autojev.train import augment, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    helper = Path(__file__).with_name("qwen35_qlora.py")
    source = ast.parse(helper.read_text(encoding="utf-8"))
    nodes = [node for node in source.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in {"StreamRows", "tree_hash"}]
    pure = {"Path": Path, "torch": torch, "hashlib": hashlib, "json": json}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(helper), "exec"), pure)
    assert pure["tree_hash"]({"scalar": torch.tensor(1.0)})
    m = DecisionModel.__new__(DecisionModel)
    torch.nn.Module.__init__(m)
    m.device_name = "cpu"
    m.processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True)
    m.processor.chat_template = m.processor.chat_template or m.processor.tokenizer.chat_template
    m.processor.tokenizer.padding_side = "left"
    m.processor.image_processor.size = {"shortest_edge": 65536, "longest_edge": 262144}
    candidates = list(string.ascii_uppercase) + ["".join(pair) for pair in itertools.product(string.ascii_uppercase, repeat=2)]
    m.codes = [x for x in candidates if len(m.processor.tokenizer.encode(x, add_special_tokens=False)) == 1][:255]
    rows = read_rows(Path(args.input))
    pieces = [m.prepare([row], max_length=512, device="cpu") for row in rows]
    merged = collate_cpu(pieces, m.processor.tokenizer.pad_token_id)
    native = m.prepare(rows, max_length=512, device="cpu")
    assert merged.inputs.keys() == native.inputs.keys()
    assert all(torch.equal(merged.inputs[key], native.inputs[key]) for key in native.inputs)
    prepare_count = 0
    real_prepare = m.prepare
    def counted(*a, **kw):
        nonlocal prepare_count
        prepare_count += 1
        return real_prepare(*a, **kw)
    m.prepare = counted
    batches = list(exact_microbatches(m, rows, 3, 512, 512))
    assert prepare_count == len(rows)
    assert [row["id"] for group, _ in batches for row in group] == [row["id"] for row in rows]
    assert all(len(group) * batch.inputs["input_ids"].shape[1] <= 512 for group, batch in batches)
    failures = []
    for budget, length in [(100, 512), (512, 100)]:
        try:
            list(exact_microbatches(m, rows[:1], 1, budget, length))
        except ValueError:
            failures.append(True)
    assert len(failures) == 2
    soft = {"id": "soft", "state": "x", "question": {"type": "choice", "criteria": {str(i): f"description {i}" for i in range(12)}}, "target": [i / 66 for i in range(12)]}
    shuffled = augment([soft], random.Random(123))[0]
    weights = dict(zip(soft["question"]["criteria"], soft["target"], strict=True))
    assert shuffled["target"] == [weights[key] for key in shuffled["question"]["criteria"]]
    assert torch.allclose(targets([shuffled], torch.device("cpu"))[0, :12], torch.tensor(shuffled["target"]))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tiny.jsonl"
        path.write_text("".join(json.dumps({"id": i}) + "\n" for i in range(3)), encoding="utf-8")
        stream = pure["StreamRows"](path, 2)
        first = stream.group(2)
        cursor = dict(stream.cursor)
        expected = stream.group(4)
        restored = pure["StreamRows"](path, 2, cursor)
        actual = restored.group(4)
        assert [r["id"] for r in first] == [0, 1]
        assert actual == expected and restored.cursor == stream.cursor
        assert restored.group(1) == []
    result = {"status": "passed_cpu_only", "weights_loaded": False, "rows": len(rows),
              "individual_lengths": [part.inputs["input_ids"].shape[1] for part in pieces],
              "collation_matches_native_all_tensors": True, "actual_length_batch_sizes": [len(group) for group, _ in batches],
              "one_processor_encoding_per_augmented_row": True, "no_truncation_rejects_overflow": True,
              "soft_targets_follow_numeric_key_permutation": True, "stream_cursor_resume_across_epochs": True,
              "zero_dim_optimizer_tensor_hash": True, "gpu_checks_executed": False,
              "trainer_sha256": hashlib.sha256(helper.read_bytes()).hexdigest()}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
