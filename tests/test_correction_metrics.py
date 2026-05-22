import torch

from rcq_moe.correction import OnlineChannelRegression
from rcq_moe.metrics import kl_divergence_summary
from rcq_moe.quantization import RescueConfig
from rcq_moe.toy_moe import ToyMoeLayer, fit_toy_moe_output_correction, quantize_toy_moe_layer


def test_online_channel_regression_recovers_affine_parameters():
    torch.manual_seed(7)
    yhat = torch.randn(64, 5)
    alpha_true = torch.tensor([1.5, -0.5, 0.0, 2.0, 0.75])
    beta_true = torch.tensor([0.1, -0.2, 3.0, 0.5, -1.0])
    y = alpha_true * yhat + beta_true

    stats = OnlineChannelRegression(dim=5)
    stats.update(y[:32], yhat[:32])
    stats.update(y[32:], yhat[32:])
    alpha, beta = stats.solve()

    assert torch.allclose(alpha, alpha_true, atol=1e-5)
    assert torch.allclose(beta, beta_true, atol=1e-5)


def test_toy_routed_output_correction_reduces_calibration_mse():
    torch.manual_seed(8)
    layer = ToyMoeLayer.random(hidden_size=8, intermediate_size=16, num_experts=4, top_k=2, seed=8)
    hidden = torch.randn(96, 8)
    config = RescueConfig.rcq_1p75(block_size=8)
    quantized = quantize_toy_moe_layer(layer, hidden, config, model_name="toy", layer_id=0, rank=4)

    y_fp = layer.forward(hidden)
    y_raw = quantized.forward(hidden, apply_correction=False)
    raw_mse = torch.mean((y_fp - y_raw).square())
    fit_toy_moe_output_correction(layer, quantized, hidden)
    y_corr = quantized.forward(hidden)
    corr_mse = torch.mean((y_fp - y_corr).square())

    assert corr_mse <= raw_mse + 1e-7
    counts = quantized.width_counts()
    assert set(counts) == {"gate", "up", "down"}
    assert all(set(v) == {1, 2, 4} for v in counts.values())


def test_kl_divergence_summary_reports_expected_keys_and_zero_self_kl():
    torch.manual_seed(9)
    logits = torch.randn(3, 4, 11)
    summary = kl_divergence_summary(logits, logits.clone())
    assert set(summary) == {"mean", "p50", "p95", "p99", "max"}
    assert summary["max"] == 0.0

