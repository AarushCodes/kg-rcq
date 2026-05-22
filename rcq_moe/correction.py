from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class OnlineChannelRegression:
    """Online per-channel regression for y = alpha * y_hat + beta."""

    dim: int
    dtype: torch.dtype = torch.float32
    device: torch.device | None = None

    def __post_init__(self) -> None:
        self.count = torch.zeros((), dtype=self.dtype, device=self.device)
        self.sum_y = torch.zeros(self.dim, dtype=self.dtype, device=self.device)
        self.sum_yhat = torch.zeros(self.dim, dtype=self.dtype, device=self.device)
        self.sum_yhat2 = torch.zeros(self.dim, dtype=self.dtype, device=self.device)
        self.sum_y_yhat = torch.zeros(self.dim, dtype=self.dtype, device=self.device)

    def update(self, y: torch.Tensor, yhat: torch.Tensor) -> None:
        if y.shape != yhat.shape:
            raise ValueError("y and yhat must have the same shape.")
        if y.shape[-1] != self.dim:
            raise ValueError(f"last dimension must be {self.dim}.")
        flat_y = y.reshape(-1, self.dim).to(dtype=self.dtype, device=self.device)
        flat_yhat = yhat.reshape(-1, self.dim).to(dtype=self.dtype, device=self.device)
        self.count += flat_y.shape[0]
        self.sum_y += flat_y.sum(dim=0)
        self.sum_yhat += flat_yhat.sum(dim=0)
        self.sum_yhat2 += flat_yhat.square().sum(dim=0)
        self.sum_y_yhat += (flat_y * flat_yhat).sum(dim=0)

    def solve(self, *, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count.item() == 0:
            raise ValueError("cannot solve affine correction with zero samples.")
        mean_y = self.sum_y / self.count
        mean_yhat = self.sum_yhat / self.count
        var_yhat = self.sum_yhat2 / self.count - mean_yhat.square()
        cov = self.sum_y_yhat / self.count - mean_y * mean_yhat
        alpha = cov / (var_yhat + eps)
        beta = mean_y - alpha * mean_yhat
        return alpha, beta

