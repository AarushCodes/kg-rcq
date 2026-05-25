from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StorageBreakdown:
    shared_bits: int
    index_bits: int
    scale_bits: int
    metadata_bits: int
    total_bits: int
    bpw: float


def expert_bpw(
    *,
    num_experts: int,
    rows: int,
    cols: int,
    rank: int,
    widths: torch.Tensor,
    block_size: int,
    shared_bits_per_value: int = 16,
    scale_bits_per_block: int = 16,
    metadata_bits_per_block: int = 2,
    shared_mode: str = "right_input",
) -> StorageBreakdown:
    """Compute honest effective bits per original expert weight."""
    if widths.shape != (num_experts, rows, (cols + block_size - 1) // block_size):
        raise ValueError("widths shape must be [num_experts, rows, ceil(cols / block_size)].")

    if shared_mode == "right_input":
        shared_bits = shared_bits_per_value * (rank * cols + num_experts * rows * rank)
    elif shared_mode == "left_output":
        shared_bits = shared_bits_per_value * (rows * rank + num_experts * rank * cols)
    elif shared_mode == "none":
        shared_bits = 0
    else:
        raise ValueError(f"unsupported shared_mode: {shared_mode!r}")
    valid_per_block = []
    for block_index in range(widths.shape[-1]):
        start = block_index * block_size
        valid_per_block.append(max(0, min(block_size, cols - start)))
    valid = torch.tensor(valid_per_block, device=widths.device, dtype=torch.int64)
    index_bits = int((widths.to(torch.int64) * valid.view(1, 1, -1)).sum().item())
    rowblocks = int(widths.numel())
    scale_bits = scale_bits_per_block * rowblocks
    metadata_bits = metadata_bits_per_block * rowblocks
    total_bits = shared_bits + index_bits + scale_bits + metadata_bits
    bpw = total_bits / (num_experts * rows * cols)
    return StorageBreakdown(shared_bits, index_bits, scale_bits, metadata_bits, total_bits, bpw)
