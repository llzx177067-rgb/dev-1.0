import os
from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


class DistributedTraining:
    def __init__(self, rank: int = 0, world_size: int = 1, device: str = "cuda") -> None:
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)

    @classmethod
    def initialize(cls) -> "DistributedTraining":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size == 1:
            return cls(device="cuda" if torch.cuda.is_available() else "cpu")
        local_rank = int(os.environ["LOCAL_RANK"])
        if int(os.environ["LOCAL_WORLD_SIZE"]) != world_size or int(os.environ["RANK"]) != local_rank:
            raise ValueError("DDP training supports one node only")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
        return cls(local_rank, world_size, f"cuda:{local_rank}")

    @property
    def primary(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier()

    def on_primary(self, operation: Callable[[], object]) -> None:
        if self.primary:
            operation()
        self.barrier()

    def broadcast(self, value: Any) -> Any:
        if self.world_size == 1:
            return value
        values = [value if self.primary else None]
        dist.broadcast_object_list(values, src=0)
        return values[0]

    def wrap(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.world_size == 1:
            return model
        return DistributedDataParallel(
            model,
            device_ids=[self.device.index] if self.device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

    def batches[T](self, rows: Sequence[T], split: Callable[[Sequence[T]], list[list[T]]]) -> list[tuple[list[T], int]]:
        if self.world_size == 1:
            return [(batch, len(batch)) for batch in split(rows)]
        shards = [split(rows[rank :: self.world_size]) for rank in range(self.world_size)]
        local = shards[self.rank]
        return [
            (local[index], len(local[index])) if index < len(local) else ([rows[0]], 0)
            for index in range(max(map(len, shards)))
        ]

    def totals(self, loss: float, tokens: int) -> tuple[float, int]:
        if self.world_size == 1:
            return loss, tokens
        values = torch.tensor([loss, tokens], dtype=torch.float64, device=self.device)
        dist.all_reduce(values)
        return float(values[0]), int(values[1])

    @torch.no_grad()
    def broadcast_parameters(self, model: torch.nn.Module) -> None:
        if self.world_size > 1:
            for parameter in model.parameters():
                dist.broadcast(parameter, src=0)

    def close(self) -> None:
        if self.world_size > 1:
            dist.destroy_process_group()
