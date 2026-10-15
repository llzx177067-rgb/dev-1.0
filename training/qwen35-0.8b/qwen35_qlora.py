"""Local NF4 LoRA Decider training, with streaming data and exact resume.

Frozen upstream data/model code is never edited. Run from the workspace root
with runtime-source/src on PYTHONPATH. No record packing or token truncation.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from autojev.exact_batch import collate_cpu, exact_microbatches, move_batch
from autojev.model import DecisionModel, MAX_OPTIONS, PreparedBatch, enable_noncausal_full_attention
from autojev.train import augment, learning_rate_factor, targets
from autojev.loss import decision_loss
from autojev.evaluate import (
    calibration_ok, evaluate_logits, fit_temperature, hard_label, label_index,
    metrics, options, read_rows, selection_key,
)

BASE = "Qwen/Qwen3.5-0.8B-Base"
REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
ATTENTION_TARGETS = r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj))"
ALL_TARGETS = r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|mlp\.(gate_proj|up_proj|down_proj))"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def log(root, event, **values):
    row = {"event": event, "time_unix": time.time(), **values}
    with (Path(root) / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
    print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)


def tree_hash(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def trainable_state(model):
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}


def frozen_state_hash(model):
    return tree_hash({name: value for name, value in model.backbone.state_dict().items() if "lora_" not in name})


def processor_input_hashes(model, rows, args):
    return [tree_hash(vars(model.prepare([row], max_length=args.max_length, device="cpu"))) for row in rows]


def environment_identity():
    packages = {distribution.metadata["Name"]: distribution.version for distribution in importlib.metadata.distributions() if distribution.metadata.get("Name")}
    return {"packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
            "python": sys.version, "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "sdpa_backends_enabled": {"flash": torch.backends.cuda.flash_sdp_enabled(),
                "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math": torch.backends.cuda.math_sdp_enabled(), "cudnn": torch.backends.cuda.cudnn_sdp_enabled()}}


def configure_runtime(args):
    torch.backends.cuda.enable_cudnn_sdp(False)
    if getattr(args, "deterministic", False):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True


def base_identity(path):
    directory = Path(path)
    manifest_path = directory / "download-verification-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "verified" or manifest.get("repo") != BASE or manifest.get("revision") != REVISION:
        raise ValueError("Local base download manifest does not identify the frozen Qwen3.5 Base")
    files = {}
    for item in manifest["files"]:
        name = item["name"]
        if Path(name).is_absolute() or Path(name).name != name or item.get("verified") is not True:
            raise ValueError("Invalid local base manifest filename or verification flag")
        file = directory / name
        actual = sha(file)
        if file.stat().st_size != item["actual_size"] or actual != item["actual_sha256"]:
            raise ValueError(f"Frozen base file changed: {name}")
        if item.get("lfs_sha256") and actual != item["lfs_sha256"]:
            raise ValueError(f"Frozen base LFS identity differs: {name}")
        files[name] = {"bytes": file.stat().st_size, "sha256": actual}
    if "config.json" not in files or "tokenizer.json" not in files:
        raise ValueError("Verified base manifest lacks required config/tokenizer")
    if "model.safetensors.index.json" in files:
        index = json.loads((directory / "model.safetensors.index.json").read_text())
        weights = index.get("weight_map")
        if not isinstance(weights, dict) or not weights:
            raise ValueError("Base weight index is empty")
        for name in set(weights.values()):
            if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors") or name not in files:
                raise ValueError("Base weight index references a missing or nonlocal verified shard")
    elif "model.safetensors" not in files:
        raise ValueError("Verified base manifest lacks a single safetensors file or a complete shard index")
    return {"repo": BASE, "revision": REVISION, "download_manifest_sha256": sha(manifest_path), "files": files}


class QuantizedDecisionModel(DecisionModel):
    def __init__(self, args, checkpoint=None):
        torch.nn.Module.__init__(self)
        import itertools
        import string
        torch.set_num_threads(args.cpu_threads)
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.device_name = "cuda:0"
        self.base_model, self.revision = BASE, REVISION
        self.attention_mode, self.pooling = args.attention_mode, "last"
        self.temperature = 1.0
        processor_path = str(Path(checkpoint) / "processor") if checkpoint else args.model_path
        self.processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
        if not self.processor.chat_template:
            self.processor.chat_template = self.processor.tokenizer.chat_template
        if not self.processor.chat_template:
            raise ValueError("The pinned base has no chat template")
        self.processor.tokenizer.padding_side = "left"
        self.processor.image_processor.size = {"shortest_edge": 65536, "longest_edge": 262144}
        tokenizer = self.processor.tokenizer
        candidates = list(string.ascii_uppercase) + ["".join(p) for p in itertools.product(string.ascii_uppercase, repeat=2)]
        self.codes = [code for code in candidates if len(tokenizer.encode(code, add_special_tokens=False)) == 1][:MAX_OPTIONS]
        self.token_ids = [tokenizer.encode(code, add_special_tokens=False)[0] for code in self.codes]
        if len(set(self.token_ids)) != MAX_OPTIONS:
            raise ValueError("The base must supply 255 distinct single-token answer codes")
        prefix = tokenizer.apply_chat_template([{"role": "user", "content": "Choose an option."}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        if any(tokenizer.encode(prefix + code, add_special_tokens=False) != prefix_ids + [index] for code, index in zip(self.codes, self.token_ids, strict=True)):
            raise ValueError("Answer code is not a single token after the assistant prefix")
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["lm_head", "model.visual"])
        original = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
            quantization_config=quant, device_map={"": self.device_name}, low_cpu_mem_usage=True,
        )
        import bitsandbytes as bnb
        if any(isinstance(module, bnb.nn.Linear4bit) for module in original.model.visual.modules()):
            raise RuntimeError("Vision must remain a frozen, unquantized BF16 module")
        if isinstance(original.lm_head, bnb.nn.Linear4bit):
            raise RuntimeError("Answer head initialization requires the unquantized tied embedding rows")
        self.readout = torch.nn.Linear(original.config.text_config.hidden_size, MAX_OPTIONS, bias=False,
                                      dtype=torch.float32, device=self.device_name)
        with torch.no_grad():
            self.readout.weight.copy_(original.lm_head.weight[self.token_ids].float())
        original = prepare_model_for_kbit_training(original, use_gradient_checkpointing=True,
                                                   gradient_checkpointing_kwargs={"use_reentrant": False})
        # PEFT upcasts all non-quantized tensors. For this 4 GB target, retain
        # frozen embedding/vision in BF16; norm and tiny recurrence scalars stay FP32.
        with torch.no_grad():
            original.model.language_model.embed_tokens.to(dtype=torch.bfloat16)
            original.model.visual.to(dtype=torch.bfloat16)
        backbone = original.model
        # Quantization flags live on the original conditional-generation root.
        # PEFT's Linear4bit dispatcher needs them after we discard the LM wrapper.
        backbone.is_loaded_in_4bit = True
        backbone.quantization_method = original.quantization_method
        del original
        if checkpoint:
            saved = json.loads((Path(checkpoint) / "decision_config.json").read_text())
            for key, expected in {"base_model": BASE, "revision": REVISION, "attention_mode": self.attention_mode, "codes": self.codes, "token_ids": self.token_ids}.items():
                if saved.get(key) != expected:
                    raise ValueError(f"Checkpoint identity differs: {key}")
            self.temperature = saved["temperature"]
            self.backbone = PeftModel.from_pretrained(backbone, str(Path(checkpoint) / "adapter"), is_trainable=True)
            self.readout.load_state_dict(load_file(str(Path(checkpoint) / "readout.safetensors")))
        else:
            lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, bias="none",
                              target_modules=ATTENTION_TARGETS if args.lora_targets == "attention" else ALL_TARGETS)
            self.backbone = get_peft_model(backbone, lora)
        if self.attention_mode == "noncausal_full_attention":
            enable_noncausal_full_attention(backbone.language_model)
        self.readout.requires_grad_(True)
        if any(parameter.requires_grad and "lora_" not in name and not name.startswith("readout.") for name, parameter in self.named_parameters()):
            raise RuntimeError("A frozen base parameter unexpectedly requires gradients")
        import bitsandbytes as bnb
        quantized_count = sum(isinstance(module, bnb.nn.Linear4bit) for module in self.backbone.modules())
        adapter_count = sum(hasattr(module, "lora_A") and isinstance(getattr(module, "base_layer", None), bnb.nn.Linear4bit) for module in self.backbone.modules())
        if not quantized_count or not adapter_count:
            raise RuntimeError("The quantized backbone did not receive real Linear4bit LoRA adapters")
        self.quantized_module_count, self.quantized_adapter_module_count = quantized_count, adapter_count
        self.train()

    def forward(self, batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = self.backbone(**batch.inputs, use_cache=False).last_hidden_state[:, -1]
        with torch.autocast("cuda", enabled=False):
            logits = self.readout(hidden.float())
        mask = torch.arange(MAX_OPTIONS, device=logits.device)[None] >= torch.tensor(batch.counts, device=logits.device)[:, None]
        return logits.masked_fill(mask, -1e9)

    def save_adapter(self, destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        self.backbone.save_pretrained(str(destination / "adapter"), safe_serialization=True)
        save_file({"weight": self.readout.weight.detach().cpu().contiguous()}, str(destination / "readout.safetensors"))
        self.processor.save_pretrained(str(destination / "processor"))
        write_json(destination / "decision_config.json", {
            "format_version": "qwen35-qlora-decider-1", "base_model": BASE, "revision": REVISION,
            "attention_mode": self.attention_mode, "pooling": "last", "temperature": self.temperature,
            "codes": self.codes, "token_ids": self.token_ids, "quantization": "nf4-double-bfloat16",
            "frozen_embedding_and_visual_dtype": "bfloat16", "readout_dtype": "float32",
        })


class StreamRows:
    def __init__(self, path, epochs, cursor=None):
        self.path, self.epochs = Path(path), epochs
        self.cursor = dict(cursor or {"offset": 0, "epoch": 0, "rows_seen": 0})
        self.stream = self.path.open("rb")
        self.stream.seek(self.cursor["offset"])

    def group(self, count):
        result = []
        while len(result) < count and self.cursor["epoch"] < self.epochs:
            line = self.stream.readline()
            if not line:
                self.cursor["epoch"] += 1
                self.cursor["offset"] = 0
                self.stream.seek(0)
                continue
            self.cursor["offset"] = self.stream.tell()
            if not line.strip():
                raise ValueError("Blank lines in frozen training input")
            result.append(json.loads(line))
            self.cursor["rows_seen"] += 1
        return result


def infer(model, rows, args, *, fold=None, step=None):
    values = []
    began, last_report, processed, last_count = time.monotonic(), time.monotonic(), 0, 0
    model.eval()
    with torch.inference_mode():
        for batch, cpu in exact_microbatches(model, rows, args.batch_size, args.token_budget, args.max_length):
            prepared = move_batch(cpu, model.device_name)
            logits = model(prepared).cpu().tolist()
            values.extend(value[:len(options(row["question"]))] for row, value in zip(batch, logits, strict=True))
            processed += len(batch)
            now = time.monotonic()
            if fold and (processed == len(rows) or processed - last_count >= 128 or now - last_report >= 30):
                log(args.output, "evaluation_progress", step=step, fold=fold, processed=processed, total=len(rows),
                    elapsed_seconds=now - began, examples_per_second=processed / max(now - began, 1e-9))
                last_report, last_count = now, processed
    model.train()
    return values


def update(model, optimizer, rows, args, step, total_steps):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    rows = augment(rows, random.Random(args.seed + 100003 * step))
    if getattr(args, "length_bucket", False):
        # Only reorder already augmented rows inside this optimizer update.
        # Frozen lengths are hints; exact_microbatches still encodes every
        # complete row once and checks the actual augmented padded-token budget.
        rows = sorted(rows, key=lambda row: row["source"]["input_tokens"])
    loss_sum, tokens, maximum, padded_max = 0.0, 0, 0, 0
    began = time.monotonic()
    for batch, cpu in exact_microbatches(model, rows, args.batch_size, args.token_budget, args.max_length):
        width = cpu.inputs["input_ids"].shape[1]
        maximum, padded_max = max(maximum, width), max(padded_max, len(batch) * width)
        prepared = move_batch(cpu, model.device_name)
        logits = model(prepared)
        target = targets(batch, logits.device)
        loss = decision_loss(logits, target, 0.0).total
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at update {step}")
        (loss * len(batch) / len(rows)).backward()
        loss_sum += float(loss.detach()) * len(batch)
        tokens += prepared.input_tokens
        del prepared, logits, loss, target
    norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0))
    if not math.isfinite(norm):
        raise RuntimeError("Nonfinite gradient norm")
    factor = learning_rate_factor(step, total_steps, args.warmup_ratio)
    for group in optimizer.param_groups:
        group["lr"] = args.lr * factor
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return {"step": step, "loss": loss_sum / len(rows), "gradient_norm": norm, "learning_rate": args.lr * factor,
            "examples": len(rows), "input_tokens": tokens, "maximum_length": maximum,
            "maximum_padded_tokens": padded_max, "seconds": time.monotonic() - began,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(value):
    random.setstate(value["python"])
    torch.set_rng_state(value["torch"])
    torch.cuda.set_rng_state_all(value["cuda"])


def checkpoint(root, model, optimizer, state, best=False, final=False):
    root = Path(root)
    folder = root / "checkpoints" / f"step-{state['step']:07d}"
    folder.parent.mkdir(parents=True, exist_ok=True)
    if not folder.exists():
        pending = folder.with_name(folder.name + ".pending")
        if pending.exists():
            # Only a checkpoint attempt under this run's exact known prefix.
            shutil.rmtree(pending)
        model.save_adapter(pending)
        payload = {**state, "optimizer": optimizer.state_dict(), "rng": rng_state(),
                   "trainable_sha256": tree_hash(trainable_state(model))}
        torch.save(payload, pending / "state.pt")
        write_json(pending / "files.json", {str(path.relative_to(pending)): sha(path) for path in sorted(pending.rglob("*")) if path.is_file()})
        os.replace(pending, folder)
    pointer = {"checkpoint": str(folder.relative_to(root)), "step": state["step"]}
    write_json(root / "latest.json", pointer)
    if best:
        write_json(root / "best.json", pointer)
    if final:
        write_json(root / "final.json", pointer)
    keep = {json.loads(path.read_text())["checkpoint"] for path in (root / "latest.json", root / "best.json", root / "final.json") if path.exists()}
    for path in folder.parent.glob("step-*"):
        if path.is_dir() and not path.name.endswith(".pending") and str(path.relative_to(root)) not in keep:
            shutil.rmtree(path)
    return folder


def load_state(path):
    path = Path(path)
    hashes = json.loads((path / "files.json").read_text())
    for name, expected in hashes.items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha(path / name) != expected:
            raise ValueError(f"Checkpoint file identity differs: {name}")
    return torch.load(path / "state.pt", map_location="cpu", weights_only=True)


def smoke(args):
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Smoke output must be empty")
    root.mkdir(parents=True, exist_ok=True)
    rows = read_rows(Path(args.smoke_input))
    if not rows:
        raise ValueError("Smoke input must be nonempty")
    torch.cuda.reset_peak_memory_stats()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    log(root, "model_loading", phase="smoke", attention_mode=args.attention_mode)
    model = QuantizedDecisionModel(args)
    log(root, "model_ready", phase="smoke", gpu_allocated_bytes=torch.cuda.memory_allocated(),
        quantized_modules=model.quantized_module_count, quantized_adapter_modules=model.quantized_adapter_module_count)
    model.eval()
    frozen_before = frozen_state_hash(model)
    inputs_before = processor_input_hashes(model, rows, args)
    parameter_order = [name for name, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, eps=args.adam_eps, weight_decay=0.01)
    original = model.prepare([rows[0]], max_length=args.max_length, device="cpu")
    padded_inputs = {}
    for key, value in original.inputs.items():
        if key in {"input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"}:
            padded_inputs[key] = torch.nn.functional.pad(value, (17, 0), value=model.processor.tokenizer.pad_token_id if key == "input_ids" else 0)
        else:
            padded_inputs[key] = value
    with torch.inference_mode():
        expected_logits = model(move_batch(original, "cuda:0")).cpu()
        padded_logits = model(move_batch(PreparedBatch(padded_inputs, original.counts, original.input_tokens), "cuda:0")).cpu()
    valid = original.counts[0]
    padding_delta = float((expected_logits[:, :valid] - padded_logits[:, :valid]).abs().max())
    if not torch.allclose(expected_logits[:, :valid], padded_logits[:, :valid], atol=0.08, rtol=0.03):
        raise AssertionError(f"Left padding changes valid logits: {padding_delta}")
    # CPU collation must exactly equal the processor's real batch, including image order.
    probe = rows[:min(3, len(rows))]
    individual = [model.prepare([row], max_length=args.max_length, device="cpu") for row in probe]
    merged = collate_cpu(individual, model.processor.tokenizer.pad_token_id)
    native = model.prepare(probe, max_length=args.max_length, device="cpu")
    if merged.inputs.keys() != native.inputs.keys() or any(not torch.equal(merged.inputs[k], native.inputs[k]) for k in native.inputs):
        raise AssertionError("CPU exact-length collation differs from native processor batch")
    before = trainable_state(model)
    first = update(model, optimizer, rows, args, 1, 3)
    after = trainable_state(model)
    changed_head = not torch.equal(before["readout.weight"], after["readout.weight"])
    changed_lora = any("lora_" in name and not torch.equal(value, after[name]) for name, value in before.items())
    if not changed_head or not changed_lora:
        raise AssertionError("Both the decision head and LoRA must change after real backpropagation")
    state = {"step": 1, "spec": vars(args), "cursor": {"offset": 0, "epoch": 0, "rows_seen": len(rows)}}
    folder = checkpoint(root, model, optimizer, state)
    optimizer_hash = tree_hash(optimizer.state_dict())
    logits_after_first = infer(model, [rows[0]], args)
    second = update(model, optimizer, rows, args, 2, 3)
    expected_weights = trainable_state(model)
    expected_second_predictions = infer(model, rows, args)
    repeated_forward_predictions = infer(model, rows, args)
    same_weight_forward_delta = max(abs(a - b) for x, y in zip(expected_second_predictions, repeated_forward_predictions, strict=True) for a, b in zip(x, y, strict=True))
    del before, after, model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    saved = load_state(folder)
    restored = QuantizedDecisionModel(args, folder)
    if frozen_state_hash(restored) != frozen_before:
        raise AssertionError("Frozen base weights/quantization state differ after reload")
    if processor_input_hashes(restored, rows, args) != inputs_before:
        raise AssertionError("Processor input tensors differ after reload")
    if [name for name, p in restored.named_parameters() if p.requires_grad] != parameter_order:
        raise AssertionError("Optimizer parameter names/order differ after reload")
    if tree_hash(trainable_state(restored)) != saved["trainable_sha256"]:
        raise AssertionError("Adapter/head weights changed during save/reload")
    restored_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=args.lr, eps=args.adam_eps, weight_decay=0.01)
    restored_optimizer.load_state_dict(saved["optimizer"])
    if tree_hash(restored_optimizer.state_dict()) != optimizer_hash:
        raise AssertionError("Optimizer state differs after reload")
    restored_logits = infer(restored, [rows[0]], args)
    if not torch.allclose(torch.tensor(logits_after_first), torch.tensor(restored_logits), atol=0.08, rtol=0.03):
        raise AssertionError("Reloaded predictions differ")
    restore_rng(saved["rng"])
    repeated = update(restored, restored_optimizer, rows, args, 2, 3)
    actual = trainable_state(restored)
    resume_max_delta = max(float((expected_weights[name] - value).abs().max()) for name, value in actual.items())
    resume_mean_delta = sum(float((expected_weights[name] - value).abs().sum()) for name, value in actual.items()) / sum(value.numel() for value in actual.values())
    resumed_second_predictions = infer(restored, rows, args)
    second_logit_delta = max(abs(a - b) for x, y in zip(expected_second_predictions, resumed_second_predictions, strict=True) for a, b in zip(x, y, strict=True))
    second_probability_delta = max(float((torch.tensor(x).softmax(-1) - torch.tensor(y).softmax(-1)).abs().max()) for x, y in zip(expected_second_predictions, resumed_second_predictions, strict=True))
    # A controlled copy isolates forward state from small backward/update noise.
    with torch.no_grad():
        for name, parameter in restored.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(expected_weights[name].to(parameter.device))
    copied_predictions = infer(restored, rows, args)
    copied_weight_forward_delta = max(abs(a - b) for x, y in zip(expected_second_predictions, copied_predictions, strict=True) for a, b in zip(x, y, strict=True))
    diagnostic = {"first_update": first, "continuous_second_update": second, "resumed_second_update": repeated,
                  "prediction_reload_max_delta": float((torch.tensor(logits_after_first) - torch.tensor(restored_logits)).abs().max()),
                  "resume_max_delta": resume_max_delta,
                  "resume_mean_delta": resume_mean_delta, "second_step_max_logit_delta": second_logit_delta,
                  "second_step_max_probability_delta": second_probability_delta,
                  "same_weights_repeat_forward_max_logit_delta": same_weight_forward_delta,
                  "copied_expected_weights_forward_max_logit_delta": copied_weight_forward_delta,
                  "parameters": sorted([
                      {"name": name, "max_delta": float((expected_weights[name] - value).abs().max()),
                       "mean_delta": float((expected_weights[name] - value).abs().mean()),
                       "norm_expected": float(expected_weights[name].norm()), "norm_actual": float(value.norm())}
                      for name, value in actual.items()], key=lambda row: -row["max_delta"])}
    write_json(root / "resume-diagnostic.json", diagnostic)
    if same_weight_forward_delta != 0 or copied_weight_forward_delta != 0:
        raise AssertionError("Identical trainable weights must give identical forward predictions; see resume-diagnostic.json")
    # Controlled same-model replay on this GPU already showed backward
    # variability of ~9e-5 max / 1e-7 mean after Adam near-zero gradients.
    # State identity remains exact; the next optimization trajectory is
    # checked numerically rather than falsely promised to be bitwise equal.
    tolerances = {"weight_max": 2 * args.lr, "weight_mean": 1e-6, "loss_abs": 1e-6,
                  "loss_rel": 1e-5, "gradient_norm_rel": 1e-3, "logit_max": 0.05, "probability_max": 0.005}
    if (resume_max_delta > tolerances["weight_max"] or resume_mean_delta > tolerances["weight_mean"]
        or not math.isclose(second["loss"], repeated["loss"], abs_tol=tolerances["loss_abs"], rel_tol=tolerances["loss_rel"])
        or not math.isclose(second["gradient_norm"], repeated["gradient_norm"], rel_tol=tolerances["gradient_norm_rel"], abs_tol=1e-6)
        or second_logit_delta > tolerances["logit_max"] or second_probability_delta > tolerances["probability_max"]):
        raise AssertionError(f"Resume next-step numerical checks failed; see {root / 'resume-diagnostic.json'}")
    result = {"status": "passed", "rows": len(rows), "attention_mode": args.attention_mode,
              "lora_targets": args.lora_targets, "adam_eps": args.adam_eps,
              "length_bucket": args.length_bucket,
              "deterministic": args.deterministic, "trainer_sha256": sha(__file__),
              "left_padding_max_logit_delta": padding_delta, "cpu_collation_equal": True,
              "head_updated": changed_head, "lora_updated": changed_lora,
              "save_reload_weights_exact": True, "optimizer_reload_exact": True,
              "frozen_base_quant_state_exact": True, "all_processor_input_tensors_exact": True,
              "optimizer_parameter_order_exact": True, "frozen_base_state_sha256": frozen_before,
              "resume_policy": "state_exact_trajectory_numerical", "resume_numerical_tolerances": tolerances,
              "resume_next_step_max_weight_delta": resume_max_delta,
              "resume_next_step_mean_weight_delta": resume_mean_delta,
              "resume_next_step_max_logit_delta": second_logit_delta,
              "resume_next_step_max_probability_delta": second_probability_delta,
              "same_weights_repeat_forward_exact": True, "copied_expected_weights_forward_exact": True,
              "resume_next_step_weights_bitwise_equal": tree_hash(actual) == tree_hash(expected_weights),
              "first_update": first, "second_update": second, "resumed_second_update": repeated,
              "trainable_parameters": sum(p.numel() for p in restored.parameters() if p.requires_grad),
              "quantized_module_count": restored.quantized_module_count,
              "quantized_adapter_module_count": restored.quantized_adapter_module_count,
              "gpu": torch.cuda.get_device_name(), "environment": environment_identity()}
    write_json(root / "smoke-result.json", result)
    print(json.dumps(result, indent=2), flush=True)


def train(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if args.resume:
        pointer = json.loads((root / "latest.json").read_text())
        if Path(args.resume).resolve() != (root / pointer["checkpoint"]).resolve():
            raise ValueError("Resume must match this run's latest checkpoint; rollback requires a separate run")
    if (root / "config.json").exists() and not args.resume:
        raise FileExistsError("Existing training run requires --resume")
    data_hashes = {name: sha(getattr(args, name)) for name in ("train", "development", "temperature")}
    development, temperature = read_rows(Path(args.development)), read_rows(Path(args.temperature))
    if not development or not temperature:
        raise ValueError("Evaluation folds must be nonempty")
    protected_ids = {row["id"] for row in development + temperature}
    protected_families = {(str(row["source"].get("dataset", row["suite"])), row["family"]) for row in development + temperature}
    if len(protected_ids) != len(development) + len(temperature):
        raise ValueError("Development and temperature IDs overlap")
    rows_total = 0
    with Path(args.train).open("rb") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in protected_ids or (str(row["source"].get("dataset", row["suite"])), row["family"]) in protected_families:
                raise ValueError(f"Training overlaps a held-out fold: {row['id']}")
            rows_total += 1
    total_steps = math.ceil(rows_total * args.epochs / args.effective_batch_size)
    if not rows_total:
        raise ValueError("Training input is empty")
    package = Path(__import__("autojev.model", fromlist=["__file__"]).__file__).parent
    implementation = {"trainer": sha(__file__), **{
        "autojev/" + str(path.relative_to(package)): sha(path) for path in sorted(package.rglob("*.py"))
    }}
    implementation["uv.lock"] = sha(package.parents[1] / "uv.lock")
    implementation["pyproject.toml"] = sha(package.parents[1] / "pyproject.toml")
    spec = {key: value for key, value in vars(args).items() if key not in {"resume", "stop_after"}}
    spec.update(data_sha256=data_hashes, implementation_sha256=implementation, base_model=BASE, revision=REVISION,
                rows=rows_total, total_steps=total_steps, row_order="frozen_input_order_each_epoch; input candidate is pre-shuffled",
                environment=environment_identity(), quantization="nf4-double-bfloat16",
                frozen_embedding_and_visual_dtype="bfloat16", readout_dtype="float32")
    saved = load_state(args.resume) if args.resume else None
    if saved and saved["spec"] != spec:
        raise ValueError("Resume run/data/code/config differs from frozen spec")
    if not saved:
        write_json(root / "config.json", spec)
    elif args.stop_after is not None and args.stop_after <= saved["step"]:
        raise ValueError("Stopping step must follow the saved cursor")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    log(root, "model_loading", phase="train", resume=bool(args.resume), attention_mode=args.attention_mode)
    model = QuantizedDecisionModel(args, args.resume)
    log(root, "model_ready", phase="train", gpu_allocated_bytes=torch.cuda.memory_allocated(),
        quantized_modules=model.quantized_module_count, quantized_adapter_modules=model.quantized_adapter_module_count)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, eps=args.adam_eps, weight_decay=0.01)
    step, baseline, best, best_step = 0, None, None, None
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        if tree_hash(trainable_state(model)) != saved["trainable_sha256"]:
            raise ValueError("Resume adapter/head identity differs")
        restore_rng(saved["rng"])
        step, baseline, best, best_step = saved["step"], saved["baseline"], saved["best"], saved["best_step"]
    stream = StreamRows(args.train, args.epochs, saved["cursor"] if saved else None)
    def state():
        return {"step": step, "spec": spec, "cursor": dict(stream.cursor), "baseline": baseline, "best": best, "best_step": best_step}
    def evaluate():
        nonlocal baseline, best, best_step
        log(root, "evaluation_started", step=step, temperature_rows=len(temperature), development_rows=len(development))
        logits = infer(model, temperature, args, fold="temperature", step=step)
        fitted = fit_temperature(logits, [label_index(options(row["question"]), hard_label(row)) for row in temperature])
        model.temperature = fitted
        dev_logits = infer(model, development, args, fold="development", step=step)
        predictions = evaluate_logits(development, dev_logits, fitted)
        calibrated, raw = metrics(predictions), metrics(evaluate_logits(development, dev_logits))
        if baseline is None:
            baseline = calibrated
        eligible = calibration_ok(calibrated, baseline)
        improved = eligible and (best is None or selection_key(calibrated, step) < selection_key(best, best_step))
        if improved:
            best, best_step = calibrated, step
        prediction_path = root / f"development-{step:07d}.jsonl"
        with prediction_path.open("w", encoding="utf-8") as destination:
            for row in predictions:
                destination.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        log(root, "evaluation", step=step, raw=raw, calibrated=calibrated, temperature=fitted,
            eligible=eligible, improved=improved, best_step=best_step)
        checkpoint(root, model, optimizer, state(), best=improved)
    if step == 0:
        evaluate()
    log(root, "training_started" if not saved else "training_resumed", step=step, rows=rows_total,
        total_steps=total_steps, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        gpu=torch.cuda.get_device_name(), attention_mode=args.attention_mode)
    stop = min(total_steps, args.stop_after) if args.stop_after is not None else total_steps
    while step < stop:
        group = stream.group(args.effective_batch_size)
        if not group:
            raise ValueError("Frozen stream ended before the expected number of updates")
        value = update(model, optimizer, group, args, step + 1, total_steps)
        step += 1
        log(root, "training_step", **value, rows_seen=stream.cursor["rows_seen"], cursor=stream.cursor)
        if step % args.eval_every == 0 or step == stop:
            evaluate()
        elif step % args.save_every == 0:
            checkpoint(root, model, optimizer, state())
    complete = step == total_steps
    folder = checkpoint(root, model, optimizer, state(), final=complete)
    summary = {"complete": complete, "step": step, "total_steps": total_steps, "rows_seen": stream.cursor["rows_seen"],
               "best_step": best_step, "best": best, "checkpoint": str(folder), "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    write_json(root / "summary.json", summary)
    log(root, "training_finished" if complete else "training_paused", **summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Local immutable base snapshot; no network access")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train")
    parser.add_argument("--development")
    parser.add_argument("--temperature")
    parser.add_argument("--smoke-input", help="Run actual backward, save/reload and next-step-resume checks")
    parser.add_argument("--resume", help="Exact checkpoint directory bound to this run")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-targets", choices=("attention", "all-linear"), default="attention")
    parser.add_argument("--attention-mode", choices=("causal", "noncausal_full_attention"), default="noncausal_full_attention")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--effective-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--token-budget", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adam-eps", type=float, default=1e-6, help="AdamW stability denominator, frozen in the run/resume specification")
    parser.add_argument("--deterministic", action="store_true", help="Strict PyTorch deterministic kernels and cuBLAS workspace; custom Triton kernels retain their own numerical behavior")
    parser.add_argument("--length-bucket", action="store_true", help="Stable length sorting inside each effective optimizer batch after augmentation; no change to its members or stream cursor")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    configure_runtime(args)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This recipe needs a BF16-capable CUDA GPU")
    if min(args.rank, args.batch_size, args.effective_batch_size, args.max_length, args.token_budget, args.epochs, args.eval_every, args.save_every) < 1:
        parser.error("Counts and intervals must be positive")
    if not math.isfinite(args.adam_eps) or args.adam_eps <= 0:
        parser.error("Adam epsilon must be positive and finite")
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 < args.warmup_ratio < 1:
        parser.error("Learning rate and warmup must be positive and finite")
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be positive")
    args.base_identity = base_identity(args.model_path)
    try:
        if args.smoke_input:
            smoke(args)
        else:
            if not all((args.train, args.development, args.temperature)):
                parser.error("Training requires --train, --development and --temperature")
            train(args)
    except Exception as error:
        root = Path(args.output)
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "failure.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                                         "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                         "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                                         "environment": environment_identity()})
        raise


if __name__ == "__main__":
    main()
