from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .codebooks import lloyd_max_codebook
from .hadamard import pad_last_dim, rotate_row_block, signed_hadamard_q

BlockWidth = Literal[1, 2, 4]


@dataclass(frozen=True)
class RescueConfig:
    """Mixed-bit rescue percentages for one layer+linear type."""

    name: str
    top_4bit: float
    next_2bit: float
    block_size: int = 64
    scale_bits: int = 16
    min_bit: Literal[1, 2] = 1

    @staticmethod
    def rcq_1p55(block_size: int = 64, scale_bits: int = 16) -> "RescueConfig":
        return RescueConfig("rcq_1p55", top_4bit=0.02, next_2bit=0.10, block_size=block_size, scale_bits=scale_bits)

    @staticmethod
    def rcq_1p75(block_size: int = 64, scale_bits: int = 16) -> "RescueConfig":
        return RescueConfig("rcq_1p75", top_4bit=0.05, next_2bit=0.20, block_size=block_size, scale_bits=scale_bits)

    @staticmethod
    def rcq_1p90(block_size: int = 64, scale_bits: int = 16) -> "RescueConfig":
        return RescueConfig("rcq_1p90", top_4bit=0.05, next_2bit=0.35, block_size=block_size, scale_bits=scale_bits)

    @staticmethod
    def down_mix_1bit(block_size: int = 64, scale_bits: int = 16) -> "RescueConfig":
        return RescueConfig("down_mix_1bit", top_4bit=0.05, next_2bit=0.20, block_size=block_size, scale_bits=scale_bits)

    @staticmethod
    def down_min2_5p4(block_size: int = 64, scale_bits: int = 16) -> "RescueConfig":
        return RescueConfig("down_min2_5p4", top_4bit=0.05, next_2bit=0.95, block_size=block_size, scale_bits=scale_bits, min_bit=2)

    @staticmethod
    def down_min2_20p4(block_size: int = 64, scale_bits: int = 16) -> "RescueConfig":
        return RescueConfig("down_min2_20p4", top_4bit=0.20, next_2bit=0.80, block_size=block_size, scale_bits=scale_bits, min_bit=2)


@dataclass
class QuantizedResidual:
    """Reference fake-quantized residual representation.

    `values` stores dequantized rotated-space values for clarity. Packing is intentionally
    excluded from the prototype so the math can be tested directly.
    """

    values: torch.Tensor
    widths: torch.Tensor
    scales: torch.Tensor
    valid_cols: int
    block_size: int
    model_name: str
    layer_id: int
    linear_type: str
    expert_id: int
    scores: torch.Tensor

    def dequantize(self) -> torch.Tensor:
        rows, num_blocks, block_size = self.values.shape
        blocks = []
        for block_index in range(num_blocks):
            q = signed_hadamard_q(
                self.model_name,
                self.layer_id,
                self.linear_type,
                block_index,
                block_size,
                device=self.values.device,
                dtype=self.values.dtype,
            )
            # z = r Q, therefore r = z Q.T because Q is orthogonal.
            blocks.append(self.values[:, block_index, :] @ q.T)
        padded = torch.cat(blocks, dim=1).reshape(rows, num_blocks * block_size)
        return padded[:, : self.valid_cols]


def binary_quantize_block(z: torch.Tensor, moments: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize rotated blocks with weighted sign quantization."""
    signs = torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
    denom = moments.sum(dim=-1, keepdim=True).clamp_min(eps)
    alpha = (moments * z.abs()).sum(dim=-1, keepdim=True) / denom
    dequant = alpha * signs
    score = (moments * (z - dequant).square()).sum(dim=-1)
    return dequant, alpha.squeeze(-1), score


def lloyd_quantize_block(
    z: torch.Tensor,
    moments: torch.Tensor,
    bits: Literal[2, 4],
    *,
    iterations: int = 10,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference weighted scalar Lloyd rescue quantizer in rotated space."""
    codebook = lloyd_max_codebook(bits, dtype=torch.float64, device=z.device).to(dtype=z.dtype, device=z.device)
    beta = torch.sqrt((moments * z.square()).sum(dim=-1, keepdim=True) / moments.sum(dim=-1, keepdim=True).clamp_min(eps))
    indices = torch.zeros_like(z, dtype=torch.long)

    for _ in range(iterations):
        near_zero = beta < eps
        scaled = beta.clamp_min(eps) * codebook.view(*([1] * (z.ndim - 1)), -1)
        indices = torch.argmin((z.unsqueeze(-1) - scaled.unsqueeze(-2)).square(), dim=-1)
        chosen = codebook[indices]
        numerator = (moments * z * chosen).sum(dim=-1, keepdim=True)
        denominator = (moments * chosen.square()).sum(dim=-1, keepdim=True).clamp_min(eps)
        beta = torch.clamp(numerator / denominator, min=0.0)
        if near_zero.any():
            zero_index = torch.argmin(codebook.abs())
            indices = torch.where(near_zero.expand_as(indices), zero_index.expand_as(indices), indices)

    chosen = codebook[indices]
    dequant = beta * chosen
    return dequant, beta.squeeze(-1)


def select_rescue_widths(scores: torch.Tensor, config: RescueConfig) -> torch.Tensor:
    """Select 4-bit and 2-bit row-blocks globally by score."""
    flat_scores = scores.reshape(-1)
    total = flat_scores.numel()
    widths = torch.full((total,), config.min_bit, dtype=torch.int64, device=scores.device)
    if total == 0:
        return widths.reshape_as(scores)
    if config.min_bit not in (1, 2):
        raise ValueError(f"min_bit must be 1 or 2, got {config.min_bit}.")

    n_4bit = round(total * config.top_4bit)
    n_2bit = round(total * config.next_2bit)
    order = torch.argsort(flat_scores, descending=True)
    if n_4bit:
        widths[order[:n_4bit]] = 4
    if config.min_bit == 1 and n_2bit:
        widths[order[n_4bit : n_4bit + n_2bit]] = 2
    return widths.reshape_as(scores)


def _moments_for_expert(rotated_second_moments: torch.Tensor, expert_id: int, rotated: torch.Tensor) -> torch.Tensor:
    if rotated_second_moments.ndim == 2:
        moments = rotated_second_moments[None, :, :].expand_as(rotated)
    elif rotated_second_moments.ndim == 3:
        if expert_id >= rotated_second_moments.shape[0]:
            raise ValueError("per-expert rotated_second_moments has fewer experts than residuals.")
        moments = rotated_second_moments[expert_id][None, :, :].expand_as(rotated)
    else:
        raise ValueError("rotated_second_moments must have shape [blocks, h] or [experts, blocks, h].")
    return moments


def _rotate_residual_blocks(
    residual: torch.Tensor,
    *,
    model_name: str,
    layer_id: int,
    linear_type: str,
    block_size: int,
) -> tuple[torch.Tensor, int]:
    residual, valid_cols = pad_last_dim(residual, block_size)
    rows, padded_cols = residual.shape
    blocks = residual.reshape(rows, padded_cols // block_size, block_size)
    rotated = torch.empty_like(blocks)
    for block_index in range(blocks.shape[1]):
        q = signed_hadamard_q(
            model_name,
            layer_id,
            linear_type,
            block_index,
            block_size,
            device=residual.device,
            dtype=residual.dtype,
        )
        rotated[:, block_index, :] = rotate_row_block(blocks[:, block_index, :], q)
    return rotated, valid_cols


def quantize_residuals(
    residuals: list[torch.Tensor],
    rotated_second_moments: torch.Tensor,
    config: RescueConfig,
    *,
    model_name: str = "prototype",
    layer_id: int = 0,
    linear_type: str = "up",
    expert_score_weights: torch.Tensor | None = None,
) -> list[QuantizedResidual]:
    """Quantize all expert residuals for one layer+linear type."""
    if not residuals:
        raise ValueError("residuals must contain at least one expert tensor.")
    block_size = config.block_size
    rotated_second_moments = rotated_second_moments.to(device=residuals[0].device, dtype=residuals[0].dtype)
    if rotated_second_moments.shape[-1] != block_size:
        raise ValueError("rotated_second_moments final dimension must match block_size.")
    if rotated_second_moments.ndim == 3 and rotated_second_moments.shape[0] != len(residuals):
        raise ValueError("per-expert rotated_second_moments must have one entry per residual expert.")
    if expert_score_weights is not None:
        expert_score_weights = expert_score_weights.to(device=residuals[0].device, dtype=residuals[0].dtype)
        if expert_score_weights.shape != (len(residuals),):
            raise ValueError("expert_score_weights must have shape [num_experts].")

    rotated_by_expert: list[torch.Tensor] = []
    scales_by_expert: list[torch.Tensor] = []
    scores_by_expert: list[torch.Tensor] = []
    valid_cols_by_expert: list[int] = []

    for expert_id, residual in enumerate(residuals):
        rotated, valid_cols = _rotate_residual_blocks(
            residual,
            model_name=model_name,
            layer_id=layer_id,
            linear_type=linear_type,
            block_size=block_size,
        )
        moments = _moments_for_expert(rotated_second_moments, expert_id, rotated)
        dequant, scales, scores = binary_quantize_block(rotated, moments)
        if expert_score_weights is not None:
            scores = scores * expert_score_weights[expert_id]
        rotated_by_expert.append(dequant)
        scales_by_expert.append(scales)
        scores_by_expert.append(scores)
        valid_cols_by_expert.append(valid_cols)

    all_scores = torch.stack(scores_by_expert, dim=0)
    all_widths = select_rescue_widths(all_scores, config)

    quantized: list[QuantizedResidual] = []
    for expert_id, residual in enumerate(residuals):
        rotated_fp, _ = _rotate_residual_blocks(
            residual,
            model_name=model_name,
            layer_id=layer_id,
            linear_type=linear_type,
            block_size=block_size,
        )
        values = rotated_by_expert[expert_id].clone()
        scales = scales_by_expert[expert_id].clone()
        widths = all_widths[expert_id]
        moments = _moments_for_expert(rotated_second_moments, expert_id, rotated_fp)

        for bits in (2, 4):
            mask = widths == bits
            if mask.any():
                rescue_values, rescue_scales = lloyd_quantize_block(rotated_fp[mask], moments[mask], bits=bits)
                values[mask] = rescue_values
                scales[mask] = rescue_scales

        quantized.append(
            QuantizedResidual(
                values=values,
                widths=widths,
                scales=scales,
                valid_cols=valid_cols_by_expert[expert_id],
                block_size=block_size,
                model_name=model_name,
                layer_id=layer_id,
                linear_type=linear_type,
                expert_id=expert_id,
                scores=scores_by_expert[expert_id],
            )
        )
    return quantized
