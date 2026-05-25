from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SharedDecomposition:
    """Shared low-rank factors and residuals for one layer+linear type."""

    a_factors: list[torch.Tensor]
    b_shared: torch.Tensor
    residuals: list[torch.Tensor]
    eigvals: torch.Tensor
    eigvecs: torch.Tensor
    v_r: torch.Tensor
    captured_energy: float


@dataclass
class SharedOutputDecomposition:
    """Shared output-basis factors for down projection."""

    u_shared: torch.Tensor
    c_factors: list[torch.Tensor]
    residuals: list[torch.Tensor]
    eigvals: torch.Tensor
    eigvecs: torch.Tensor
    captured_energy: float


def choose_rank(n: int, *, divisor: int = 128, minimum: int = 8, maximum: int = 64) -> int:
    """Default spec rank rule, capped for tiny prototype dimensions."""
    if n < 1:
        raise ValueError(f"n must be positive, got {n}.")
    rank = (n + divisor - 1) // divisor
    rank = max(minimum, min(maximum, rank))
    return min(rank, n)


def klt_transform(covariance: torch.Tensor, *, floor_factor: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute sorted KLT transform T = U sqrt(Lambda) and T_inv = Lambda^-1/2 U.T."""
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    covariance = 0.5 * (covariance + covariance.T)
    eigvals, eigvecs = torch.linalg.eigh(covariance)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    lam_floor = floor_factor * eigvals.mean().clamp_min(torch.finfo(eigvals.dtype).eps)
    eigvals = eigvals.clamp_min(lam_floor)
    sqrt_vals = torch.sqrt(eigvals)
    t = eigvecs * sqrt_vals.view(1, -1)
    t_inv = (eigvecs.T / sqrt_vals.view(-1, 1))
    return eigvals, eigvecs, t, t_inv


def _normalize_importance(expert_importance: torch.Tensor, num_experts: int, eps: float = 1e-12) -> torch.Tensor:
    if expert_importance.shape != (num_experts,):
        raise ValueError(f"expert_importance must have shape ({num_experts},).")
    expert_importance = expert_importance.to(torch.float64)
    total = expert_importance.sum()
    if total <= eps:
        return torch.full((num_experts,), 1.0 / num_experts, dtype=torch.float64, device=expert_importance.device)
    return expert_importance / total


def decompose_shared_subspace(
    weights: list[torch.Tensor],
    covariance: torch.Tensor,
    expert_importance: torch.Tensor,
    *,
    rank: int | None = None,
) -> SharedDecomposition:
    """Compute the router-weighted KLT/SVD shared component and residuals."""
    if not weights:
        raise ValueError("weights must contain at least one expert matrix.")
    rows, cols = weights[0].shape
    if any(weight.shape != (rows, cols) for weight in weights):
        raise ValueError("all expert weights must have the same shape.")
    if covariance.shape != (cols, cols):
        raise ValueError(f"covariance must have shape ({cols}, {cols}).")

    dtype = weights[0].dtype
    device = weights[0].device
    covariance = covariance.to(device=device, dtype=dtype)
    rank = choose_rank(cols) if rank is None else min(rank, cols)
    if rank < 1:
        raise ValueError(f"rank must be positive, got {rank}.")

    eigvals, eigvecs, t, t_inv = klt_transform(covariance)
    importance = _normalize_importance(expert_importance.to(device=device), len(weights)).to(device=device, dtype=dtype)
    transformed = []
    for expert_id, weight in enumerate(weights):
        factor = torch.sqrt(importance[expert_id] * len(weights))
        transformed.append(factor * (weight @ t))
    stacked = torch.cat(transformed, dim=0)

    _, singular_values, vh = torch.linalg.svd(stacked, full_matrices=False)
    v_r = vh[:rank, :].T.contiguous()
    total_energy = singular_values.square().sum().item()
    captured = singular_values[:rank].square().sum().item() / total_energy if total_energy > 0 else 1.0

    b_shared = v_r.T @ t_inv
    a_factors = []
    residuals = []
    for weight in weights:
        a_e = weight @ t @ v_r
        residual = weight - a_e @ b_shared
        a_factors.append(a_e)
        residuals.append(residual)

    return SharedDecomposition(
        a_factors=a_factors,
        b_shared=b_shared,
        residuals=residuals,
        eigvals=eigvals,
        eigvecs=eigvecs,
        v_r=v_r,
        captured_energy=float(captured),
    )


def decompose_shared_output_subspace(
    weights: list[torch.Tensor],
    output_covariance: torch.Tensor,
    rank: int,
) -> SharedOutputDecomposition:
    """Compute a shared left/output basis W_e ~= U_shared C_e.

    `output_covariance` is the router-weighted covariance of FP down outputs
    under the calibration activations used for the down projection.
    """
    if not weights:
        raise ValueError("weights must contain at least one expert matrix.")
    rows, cols = weights[0].shape
    if any(weight.shape != (rows, cols) for weight in weights):
        raise ValueError("all expert weights must have the same shape.")
    if output_covariance.shape != (rows, rows):
        raise ValueError(f"output_covariance must have shape ({rows}, {rows}).")
    if rank < 1:
        raise ValueError(f"rank must be positive for left_output, got {rank}.")

    dtype = weights[0].dtype
    device = weights[0].device
    output_covariance = output_covariance.to(device=device, dtype=dtype)
    output_covariance = 0.5 * (output_covariance + output_covariance.T)
    eigvals, eigvecs = torch.linalg.eigh(output_covariance)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    rank = min(rank, rows)
    u_shared = eigvecs[:, :rank].contiguous()

    total_energy = eigvals.clamp_min(0).sum().item()
    captured = eigvals[:rank].clamp_min(0).sum().item() / (total_energy + 1e-12) if total_energy > 0 else 1.0

    c_factors = []
    residuals = []
    for weight in weights:
        c_e = u_shared.T @ weight
        residual = weight - u_shared @ c_e
        c_factors.append(c_e)
        residuals.append(residual)

    return SharedOutputDecomposition(
        u_shared=u_shared,
        c_factors=c_factors,
        residuals=residuals,
        eigvals=eigvals,
        eigvecs=eigvecs,
        captured_energy=float(captured),
    )
