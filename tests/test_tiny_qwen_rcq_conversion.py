import torch

from rcq_moe.metrics import kl_divergence_summary
from rcq_moe.quantization import RescueConfig
from rcq_moe.tiny_qwen import TinyQwen35MoeConfig, TinyQwen35MoeForCausalLM, TinyQwen35MoeRCQForCausalLM, convert_tiny_qwen_to_rcq


def test_full_rank_rcq_conversion_preserves_tiny_qwen_logits_closely():
    torch.manual_seed(12)
    config = TinyQwen35MoeConfig(
        vocab_size=48,
        hidden_size=8,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
    )
    model = TinyQwen35MoeForCausalLM(config)
    calibration_ids = torch.randint(0, config.vocab_size, (4, 6))
    eval_ids = torch.randint(0, config.vocab_size, (3, 5))

    q_model = convert_tiny_qwen_to_rcq(
        model,
        calibration_ids,
        RescueConfig.rcq_1p75(block_size=8),
        rank=8,
        fit_correction=True,
    )

    assert isinstance(q_model, TinyQwen35MoeRCQForCausalLM)
    fp_logits = model(eval_ids)
    q_logits = q_model(eval_ids)
    assert q_logits.shape == fp_logits.shape
    assert torch.isfinite(q_logits).all()
    assert torch.allclose(q_logits, fp_logits, atol=2e-5, rtol=2e-4)


def test_low_rank_rcq_conversion_runs_and_reports_finite_kl():
    torch.manual_seed(13)
    config = TinyQwen35MoeConfig(
        vocab_size=64,
        hidden_size=16,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=12,
        num_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
    )
    model = TinyQwen35MoeForCausalLM(config)
    calibration_ids = torch.randint(0, config.vocab_size, (5, 7))
    eval_ids = torch.randint(0, config.vocab_size, (2, 6))

    q_model = convert_tiny_qwen_to_rcq(
        model,
        calibration_ids,
        RescueConfig.rcq_1p75(block_size=8),
        rank=4,
        fit_correction=True,
    )

    fp_logits = model(eval_ids)
    q_logits = q_model(eval_ids)
    summary = kl_divergence_summary(fp_logits, q_logits)

    assert q_logits.shape == (2, 6, config.vocab_size)
    assert all(torch.isfinite(torch.tensor(value)) for value in summary.values())
    assert summary["mean"] >= 0.0
