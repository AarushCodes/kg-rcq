import torch

from rcq_moe.quantization import RescueConfig
from rcq_moe.toy_moe import ToyMoeLayer, collect_toy_calibration_stats, quantize_toy_moe_layer


def test_toy_calibration_stats_have_expected_shapes_and_router_importance():
    torch.manual_seed(4)
    layer = ToyMoeLayer.random(hidden_size=8, intermediate_size=12, num_experts=4, top_k=2, seed=4)
    hidden = torch.randn(32, 8)
    stats = collect_toy_calibration_stats(layer, hidden, block_size=8, model_name="toy", layer_id=0)

    assert stats["gate"].covariance.shape == (8, 8)
    assert stats["up"].rotated_second_moments.shape == (1, 8)
    assert stats["down"].covariance.shape == (12, 12)
    assert stats["down"].rotated_second_moments.shape == (2, 8)
    assert torch.allclose(stats["gate"].expert_importance.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.all(torch.linalg.eigvalsh(stats["gate"].covariance) >= -1e-5)


def test_quantized_toy_moe_forward_runs_with_random_calibration_data():
    torch.manual_seed(5)
    layer = ToyMoeLayer.random(hidden_size=8, intermediate_size=16, num_experts=4, top_k=2, seed=5)
    calibration_hidden = torch.randn(64, 8)
    eval_hidden = torch.randn(11, 8)
    config = RescueConfig.rcq_1p75(block_size=8)

    quantized = quantize_toy_moe_layer(layer, calibration_hidden, config, model_name="toy", layer_id=0, rank=4)
    fp_output = layer.forward(eval_hidden)
    q_output = quantized.forward(eval_hidden)

    assert q_output.shape == fp_output.shape
    assert torch.isfinite(q_output).all()
    assert torch.mean((fp_output - q_output).square()).item() < torch.mean(fp_output.square()).item() * 10.0


def test_full_rank_toy_quantization_error_is_from_residual_quantization_not_shared_path_shape():
    torch.manual_seed(6)
    layer = ToyMoeLayer.random(hidden_size=8, intermediate_size=8, num_experts=3, top_k=2, seed=6)
    calibration_hidden = torch.randn(48, 8)
    eval_hidden = torch.randn(7, 8)
    config = RescueConfig.rcq_1p90(block_size=8)

    quantized = quantize_toy_moe_layer(layer, calibration_hidden, config, model_name="toy", layer_id=0, rank=8)
    q_output = quantized.forward(eval_hidden)

    assert q_output.shape == (7, 8)
    for q_linear in (quantized.q_gate, quantized.q_up, quantized.q_down):
        for residual in q_linear.decomposition.residuals:
            assert residual.norm().item() < 1e-4

