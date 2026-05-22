from __future__ import annotations

import torch


def kl_divergence_summary(fp_logits: torch.Tensor, q_logits: torch.Tensor) -> dict[str, float]:
    """Return mean/p50/p95/p99/max KL(p_fp || p_q) in nats/token."""
    if fp_logits.shape != q_logits.shape:
        raise ValueError("fp_logits and q_logits must have the same shape.")
    logp_fp = torch.log_softmax(fp_logits, dim=-1)
    logp_q = torch.log_softmax(q_logits, dim=-1)
    kl = (torch.exp(logp_fp) * (logp_fp - logp_q)).sum(dim=-1).reshape(-1)
    quantiles = torch.quantile(kl, torch.tensor([0.50, 0.95, 0.99], device=kl.device, dtype=kl.dtype))
    return {
        "mean": float(kl.mean().item()),
        "p50": float(quantiles[0].item()),
        "p95": float(quantiles[1].item()),
        "p99": float(quantiles[2].item()),
        "max": float(kl.max().item()),
    }

