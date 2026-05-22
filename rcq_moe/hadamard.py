from __future__ import annotations

import hashlib

import torch


def _require_power_of_two(size: int) -> None:
    if size < 1 or size & (size - 1):
        raise ValueError(f"Hadamard block size must be a positive power of two, got {size}.")


def block_hadamard_matrix(size: int, *, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return the normalized Sylvester Hadamard matrix H with H @ H.T = I."""
    _require_power_of_two(size)
    h = torch.ones((1, 1), device=device, dtype=dtype)
    while h.shape[0] < size:
        h = torch.cat(
            [
                torch.cat([h, h], dim=1),
                torch.cat([h, -h], dim=1),
            ],
            dim=0,
        )
    return h / (size**0.5)


def stable_seed(*parts: object) -> int:
    """Deterministically map model/layer/type/block identifiers into a torch seed."""
    text = "::".join(str(part) for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)


def deterministic_block_signs(
    model_name: str,
    layer_id: int,
    linear_type: str,
    block_index: int,
    size: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return deterministic +/-1 diagonal entries for Q_b = D_b H_h."""
    _require_power_of_two(size)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(model_name, layer_id, linear_type, block_index))
    signs = torch.randint(0, 2, (size,), generator=generator, dtype=torch.int8)
    signs = signs.to(device=device)
    return torch.where(signs == 0, torch.tensor(-1, device=device), torch.tensor(1, device=device)).to(dtype)


def signed_hadamard_q(
    model_name: str,
    layer_id: int,
    linear_type: str,
    block_index: int,
    size: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Construct Q_b = D_b H_h for the reference implementation."""
    signs = deterministic_block_signs(model_name, layer_id, linear_type, block_index, size, device=device, dtype=dtype)
    h = block_hadamard_matrix(size, device=device, dtype=dtype)
    return signs[:, None] * h


def pad_last_dim(x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int]:
    """Pad the final dimension to a multiple of block_size and return the valid original length."""
    valid = x.shape[-1]
    remainder = valid % block_size
    if remainder == 0:
        return x, valid
    pad = block_size - remainder
    return torch.nn.functional.pad(x, (0, pad)), valid


def rotate_row_block(row_block: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Compute z = r Q for one or more row-blocks shaped [..., h]."""
    return row_block @ q


def rotate_activation_block(activation_block: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Compute u = Q.T a for one or more activation blocks shaped [..., h]."""
    return activation_block @ q

