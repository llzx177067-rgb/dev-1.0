"""Encode each augmented decision once; collate using its actual input length.

This batches independent examples with left padding. It never packs records or
truncates tokens. Pixel patches are concatenated in the same order as examples.
Only the pinned Qwen3VL processor's explicitly recognized outputs are accepted.
"""
from collections.abc import Iterator, Sequence

import torch

from autojev.model import DecisionModel, PreparedBatch
from autojev.types import Example

SEQUENCE_KEYS = {"input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"}
IMAGE_KEYS = {"pixel_values", "image_grid_thw"}


def collate_cpu(parts: Sequence[PreparedBatch], pad_token_id: int) -> PreparedBatch:
    if not parts:
        raise ValueError("Cannot collate an empty batch")
    keys = set().union(*(part.inputs for part in parts))
    if keys - SEQUENCE_KEYS - IMAGE_KEYS:
        raise ValueError(f"Unaudited processor outputs: {sorted(keys - SEQUENCE_KEYS - IMAGE_KEYS)}")
    longest = max(part.inputs["input_ids"].shape[1] for part in parts)
    combined = {}
    for name in keys:
        if name in SEQUENCE_KEYS:
            rows = []
            for part in parts:
                value = part.inputs.get(name)
                if value is None:
                    if name in {"input_ids", "attention_mask"}:
                        raise ValueError(f"Missing required processor tensor: {name}")
                    value = torch.zeros_like(part.inputs["input_ids"])
                if value.ndim != 2 or value.shape[0] != 1 or value.device.type != "cpu":
                    raise ValueError("Expected one independently encoded CPU sequence")
                amount = longest - value.shape[1]
                rows.append(torch.nn.functional.pad(value, (amount, 0), value=pad_token_id if name == "input_ids" else 0))
            combined[name] = torch.cat(rows, dim=0)
        else:
            combined[name] = torch.cat([part.inputs[name] for part in parts if name in part.inputs], dim=0)
    return PreparedBatch(combined, tuple(part.counts[0] for part in parts), sum(part.input_tokens for part in parts))


def move_batch(batch: PreparedBatch, device: str) -> PreparedBatch:
    return PreparedBatch({name: value.to(device) for name, value in batch.inputs.items()}, batch.counts, batch.input_tokens)


def exact_microbatches(
    model: DecisionModel,
    rows: Sequence[Example],
    batch_size: int,
    token_budget: int,
    max_length: int,
) -> Iterator[tuple[list[Example], PreparedBatch]]:
    if min(batch_size, token_budget, max_length) < 1:
        raise ValueError("Batch, token budget and maximum length must be positive")
    pending_rows, pending, longest = [], [], 0
    for row in rows:
        prepared = model.prepare([row], max_length=max_length, device="cpu")
        length = prepared.inputs["input_ids"].shape[1]
        if length > token_budget:
            raise ValueError(f"Single decision {row['id']} needs {length} tokens, exceeding token budget {token_budget}; no truncation")
        if pending and (len(pending) == batch_size or max(longest, length) * (len(pending) + 1) > token_budget):
            yield pending_rows, collate_cpu(pending, model.processor.tokenizer.pad_token_id)
            pending_rows, pending, longest = [], [], 0
        pending_rows.append(row)
        pending.append(prepared)
        longest = max(longest, length)
    if pending:
        yield pending_rows, collate_cpu(pending, model.processor.tokenizer.pad_token_id)
