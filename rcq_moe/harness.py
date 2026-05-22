from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .metrics import kl_divergence_summary
from .official_qwen import OfficialQwen35LayerDiagnostics, convert_official_qwen35_moe_to_rcq_with_diagnostics
from .quantization import RescueConfig


@dataclass
class OfficialQwenHarnessResult:
    model: nn.Module
    kl: dict[str, float]
    layer_diagnostics: list[OfficialQwen35LayerDiagnostics]


@dataclass(frozen=True)
class OfficialQwenAblationResult:
    name: str
    rank: int
    correction: bool
    kl_mean: float
    kl_p95: float
    kl_p99: float
    max_moe_mse_before: float | None
    max_moe_mse_after: float | None


def run_official_qwen_harness(
    model: nn.Module,
    calibration_input_ids: torch.Tensor,
    eval_input_ids: torch.Tensor,
    rcq_config: RescueConfig,
    *,
    rank: int,
    fit_correction: bool,
) -> OfficialQwenHarnessResult:
    model.eval()
    with torch.no_grad():
        fp_logits = model(input_ids=eval_input_ids, use_cache=False).logits
    conversion = convert_official_qwen35_moe_to_rcq_with_diagnostics(
        model,
        calibration_input_ids,
        rcq_config,
        rank=rank,
        fit_correction=fit_correction,
    )
    with torch.no_grad():
        q_logits = conversion.model(input_ids=eval_input_ids, use_cache=False).logits
    return OfficialQwenHarnessResult(
        model=conversion.model,
        kl=kl_divergence_summary(fp_logits, q_logits),
        layer_diagnostics=conversion.layer_diagnostics,
    )


def run_official_qwen_ablation(
    model: nn.Module,
    calibration_input_ids: torch.Tensor,
    eval_input_ids: torch.Tensor,
    rcq_config: RescueConfig,
    *,
    ranks: list[int],
) -> list[OfficialQwenAblationResult]:
    results: list[OfficialQwenAblationResult] = []
    hidden_size = model.config.hidden_size
    configs: list[tuple[str, int, bool]] = [("full_rank_debug", hidden_size, True)]
    for rank in ranks:
        configs.append((f"rank_{rank}_no_correction", rank, False))
        configs.append((f"rank_{rank}_with_correction", rank, True))

    for name, rank, correction in configs:
        harness = run_official_qwen_harness(
            model,
            calibration_input_ids,
            eval_input_ids,
            rcq_config,
            rank=rank,
            fit_correction=correction,
        )
        before_values = [
            diag.moe_mse_before_correction
            for diag in harness.layer_diagnostics
            if diag.moe_mse_before_correction is not None
        ]
        after_values = [
            diag.moe_mse_after_correction
            for diag in harness.layer_diagnostics
            if diag.moe_mse_after_correction is not None
        ]
        results.append(
            OfficialQwenAblationResult(
                name=name,
                rank=rank,
                correction=correction,
                kl_mean=harness.kl["mean"],
                kl_p95=harness.kl["p95"],
                kl_p99=harness.kl["p99"],
                max_moe_mse_before=max(before_values) if before_values else None,
                max_moe_mse_after=max(after_values) if after_values else None,
            )
        )
    return results

