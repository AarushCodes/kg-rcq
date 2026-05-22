from __future__ import annotations

import torch


def _normal_pdf(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * x * x) / ((2.0 * torch.pi) ** 0.5)


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / (2.0**0.5)))


def lloyd_max_codebook(
    bits: int,
    *,
    iterations: int = 200,
    tol: float = 1e-7,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate a deterministic Lloyd-Max scalar codebook for N(0, 1)."""
    if bits < 1:
        raise ValueError(f"bits must be positive, got {bits}.")
    levels = 1 << bits
    normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(1.0, dtype=dtype, device=device),
    )
    probs = (torch.arange(levels, dtype=dtype, device=device) + 0.5) / levels
    centroids = normal.icdf(probs)

    for _ in range(iterations):
        old = centroids.clone()
        boundaries = torch.empty(levels + 1, dtype=dtype, device=device)
        boundaries[0] = -torch.inf
        boundaries[-1] = torch.inf
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])

        left = boundaries[:-1]
        right = boundaries[1:]
        denom = _normal_cdf(right) - _normal_cdf(left)
        numer = _normal_pdf(left) - _normal_pdf(right)
        centroids = numer / denom.clamp_min(torch.finfo(dtype).tiny)
        if torch.max(torch.abs(centroids - old)).item() < tol:
            break
    return centroids.to(torch.float32)

