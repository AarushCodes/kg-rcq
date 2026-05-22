import torch

from rcq_moe.harness import run_official_qwen_ablation, run_official_qwen_harness
from rcq_moe.official_qwen import make_tiny_official_qwen35_moe
from rcq_moe.quantization import RescueConfig


def test_official_qwen_harness_returns_finite_metrics_and_diagnostics():
    torch.manual_seed(20)
    model = make_tiny_official_qwen35_moe(vocab_size=64, hidden_size=16, moe_intermediate_size=16, num_hidden_layers=2)
    calibration_ids = torch.randint(0, model.config.vocab_size, (4, 5))
    eval_ids = torch.randint(0, model.config.vocab_size, (2, 4))

    result = run_official_qwen_harness(
        model,
        calibration_ids,
        eval_ids,
        RescueConfig.rcq_1p75(block_size=8),
        rank=4,
        fit_correction=True,
    )

    assert set(result.kl) == {"mean", "p50", "p95", "p99", "max"}
    assert all(torch.isfinite(torch.tensor(value)) for value in result.kl.values())
    assert len(result.layer_diagnostics) == model.config.num_hidden_layers
    for layer in result.layer_diagnostics:
        assert layer.moe_mse_before_correction is not None
        assert layer.moe_mse_after_correction is not None
        assert layer.moe_mse_after_correction <= layer.moe_mse_before_correction + 1e-12
        assert set(layer.linear_diagnostics) == {"gate", "up", "down"}
        for diag in layer.linear_diagnostics.values():
            assert diag.bpw > 0.0
            assert 0.0 <= diag.captured_energy <= 1.0
            assert abs(sum(diag.width_percentages.values()) - 1.0) < 1e-6


def test_official_qwen_harness_full_rank_debug_kl_is_near_zero():
    torch.manual_seed(21)
    model = make_tiny_official_qwen35_moe(
        vocab_size=48,
        hidden_size=8,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_hidden_layers=1,
    )
    calibration_ids = torch.randint(0, model.config.vocab_size, (4, 5))
    eval_ids = torch.randint(0, model.config.vocab_size, (2, 4))

    result = run_official_qwen_harness(
        model,
        calibration_ids,
        eval_ids,
        RescueConfig.rcq_1p75(block_size=8),
        rank=model.config.hidden_size,
        fit_correction=True,
    )

    assert abs(result.kl["mean"]) < 1e-6
    assert result.kl["max"] < 1e-6


def test_official_qwen_ablation_returns_expected_rows():
    torch.manual_seed(22)
    model = make_tiny_official_qwen35_moe(vocab_size=48, hidden_size=8, moe_intermediate_size=8, num_hidden_layers=1)
    calibration_ids = torch.randint(0, model.config.vocab_size, (3, 4))
    eval_ids = torch.randint(0, model.config.vocab_size, (2, 4))

    rows = run_official_qwen_ablation(
        model,
        calibration_ids,
        eval_ids,
        RescueConfig.rcq_1p75(block_size=8),
        ranks=[2, 4],
    )

    assert [row.name for row in rows] == [
        "full_rank_debug",
        "rank_2_no_correction",
        "rank_2_with_correction",
        "rank_4_no_correction",
        "rank_4_with_correction",
    ]
    assert all(torch.isfinite(torch.tensor(row.kl_mean)) for row in rows)

