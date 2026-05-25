from __future__ import annotations

from dataclasses import dataclass

import torch

from .hadamard import pad_last_dim, rotate_activation_block, signed_hadamard_q


@dataclass
class LinearCalibrationStats:
    covariance: torch.Tensor
    expert_importance: torch.Tensor
    rotated_second_moments: torch.Tensor
    per_expert_rotated_second_moments: torch.Tensor | None = None
    per_expert_router_z: torch.Tensor | None = None
    per_expert_selected_count: torch.Tensor | None = None
    global_rotated_second_moments: torch.Tensor | None = None
    down_output_covariance: torch.Tensor | None = None


def accumulate_covariance_and_moments(
    activations_by_expert: list[list[torch.Tensor]],
    router_weights_by_expert: list[list[torch.Tensor]],
    *,
    num_experts: int,
    input_dim: int,
    block_size: int,
    model_name: str,
    layer_id: int,
    linear_type: str,
    outputs_by_expert: list[list[torch.Tensor]] | None = None,
    eps: float = 1e-12,
) -> LinearCalibrationStats:
    """Accumulate router-weighted covariance and rotated diagonal moments.

    Inputs are grouped by expert because `down` activations are expert-specific.
    Each tensor in a group is shaped [input_dim].
    """
    first_activation = next((items[0] for items in activations_by_expert if items), None)
    if first_activation is None:
        raise ValueError("at least one calibration activation is required.")
    first_output = next((items[0] for items in outputs_by_expert if items), None) if outputs_by_expert is not None else None
    dtype = first_activation.dtype
    device = first_activation.device
    covariance_sum = torch.zeros((input_dim, input_dim), dtype=dtype, device=device)
    output_covariance_sum = (
        torch.zeros((first_output.numel(), first_output.numel()), dtype=dtype, device=device)
        if first_output is not None
        else None
    )
    expert_usage = torch.zeros(num_experts, dtype=dtype, device=device)
    padded_dim = ((input_dim + block_size - 1) // block_size) * block_size
    num_blocks = padded_dim // block_size
    moment_sum = torch.zeros((num_blocks, block_size), dtype=dtype, device=device)
    per_expert_moment_sum = torch.zeros((num_experts, num_blocks, block_size), dtype=dtype, device=device)
    selected_count = torch.zeros(num_experts, dtype=dtype, device=device)
    total_weight = torch.zeros((), dtype=dtype, device=device)

    for expert_id in range(num_experts):
        output_items = outputs_by_expert[expert_id] if outputs_by_expert is not None else [None] * len(activations_by_expert[expert_id])
        for activation, router_weight, output in zip(activations_by_expert[expert_id], router_weights_by_expert[expert_id], output_items):
            weight = router_weight.square()
            covariance_sum += weight * torch.outer(activation, activation)
            if output_covariance_sum is not None and output is not None:
                output_covariance_sum += weight * torch.outer(output, output)
            expert_usage[expert_id] += weight
            selected_count[expert_id] += 1
            total_weight += weight

            padded, _ = pad_last_dim(activation, block_size)
            blocks = padded.reshape(num_blocks, block_size)
            for block_index in range(num_blocks):
                q = signed_hadamard_q(
                    model_name,
                    layer_id,
                    linear_type,
                    block_index,
                    block_size,
                    device=device,
                    dtype=dtype,
                )
                rotated = rotate_activation_block(blocks[block_index], q)
                moment_sum[block_index] += weight * rotated.square()
                per_expert_moment_sum[expert_id, block_index] += weight * rotated.square()

    covariance = covariance_sum / (total_weight + eps)
    rotated_second_moments = moment_sum / (total_weight + eps)
    output_covariance = output_covariance_sum / (total_weight + eps) if output_covariance_sum is not None else None
    importance = expert_usage / (expert_usage.sum() + eps)
    cond = per_expert_moment_sum / expert_usage.clamp_min(eps).view(num_experts, 1, 1)
    shrink = selected_count / (selected_count + 4096.0)
    per_expert_moments = shrink.view(num_experts, 1, 1) * cond + (1.0 - shrink).view(num_experts, 1, 1) * rotated_second_moments
    per_expert_moments = torch.nan_to_num(per_expert_moments, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    return LinearCalibrationStats(
        covariance,
        importance,
        rotated_second_moments,
        per_expert_rotated_second_moments=per_expert_moments,
        per_expert_router_z=expert_usage,
        per_expert_selected_count=selected_count,
        global_rotated_second_moments=rotated_second_moments,
        down_output_covariance=output_covariance,
    )
