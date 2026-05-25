import torch

from rcq_moe.metrics import kl_divergence_summary
from rcq_moe.official_qwen import (
    OfficialQwen35MoeRCQSparseMoeBlock,
    collect_official_qwen_mlp_inputs,
    convert_official_qwen35_moe_to_rcq,
    make_tiny_official_qwen35_moe,
)
from rcq_moe.quantization import RescueConfig


def test_official_tiny_qwen35_moe_forward_and_shapes():
    torch.manual_seed(16)
    model = make_tiny_official_qwen35_moe(vocab_size=64, hidden_size=16, moe_intermediate_size=24, num_hidden_layers=1)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 5))

    outputs = model(input_ids=input_ids, use_cache=False)

    assert outputs.logits.shape == (2, 5, model.config.vocab_size)
    assert torch.isfinite(outputs.logits).all()
    experts = model.model.layers[0].mlp.experts
    assert tuple(experts.gate_up_proj.shape) == (model.config.num_experts, 2 * model.config.moe_intermediate_size, model.config.hidden_size)
    assert tuple(experts.down_proj.shape) == (model.config.num_experts, model.config.hidden_size, model.config.moe_intermediate_size)


def test_collect_official_qwen_mlp_inputs_uses_real_forward_hooks():
    torch.manual_seed(17)
    model = make_tiny_official_qwen35_moe(vocab_size=48, hidden_size=16, moe_intermediate_size=16, num_hidden_layers=2)
    input_ids = torch.randint(0, model.config.vocab_size, (3, 4))

    captures = collect_official_qwen_mlp_inputs(model, input_ids)

    assert len(captures) == 2
    assert captures[0].shape == (3, 4, model.config.hidden_size)
    assert captures[1].shape == (3, 4, model.config.hidden_size)


def test_full_rank_official_qwen_rcq_conversion_preserves_logits_with_correction():
    torch.manual_seed(18)
    model = make_tiny_official_qwen35_moe(
        vocab_size=48,
        hidden_size=8,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_hidden_layers=1,
    )
    calibration_ids = torch.randint(0, model.config.vocab_size, (4, 5))
    eval_ids = torch.randint(0, model.config.vocab_size, (2, 4))

    q_model = convert_official_qwen35_moe_to_rcq(
        model,
        calibration_ids,
        RescueConfig.rcq_1p75(block_size=8),
        rank=8,
        fit_correction=True,
    )
    fp_logits = model(input_ids=eval_ids, use_cache=False).logits
    q_logits = q_model(input_ids=eval_ids, use_cache=False).logits

    assert isinstance(q_model.model.layers[0].mlp, OfficialQwen35MoeRCQSparseMoeBlock)
    assert q_logits.shape == fp_logits.shape
    assert torch.allclose(q_logits, fp_logits, atol=3e-5, rtol=3e-4)


def test_low_rank_official_qwen_rcq_conversion_runs_and_has_finite_kl():
    torch.manual_seed(19)
    model = make_tiny_official_qwen35_moe(vocab_size=64, hidden_size=16, moe_intermediate_size=16, num_hidden_layers=2)
    calibration_ids = torch.randint(0, model.config.vocab_size, (4, 5))
    eval_ids = torch.randint(0, model.config.vocab_size, (2, 4))

    q_model = convert_official_qwen35_moe_to_rcq(
        model,
        calibration_ids,
        RescueConfig.rcq_1p75(block_size=8),
        rank=4,
        fit_correction=True,
    )
    fp_logits = model(input_ids=eval_ids, use_cache=False).logits
    q_logits = q_model(input_ids=eval_ids, use_cache=False).logits
    summary = kl_divergence_summary(fp_logits, q_logits)

    assert q_logits.shape == (2, 4, model.config.vocab_size)
    assert torch.isfinite(q_logits).all()
    assert all(torch.isfinite(torch.tensor(value)) for value in summary.values())
    assert summary["mean"] >= 0.0


def test_official_qwen_rcq_conversion_runs_with_down_none_mode():
    torch.manual_seed(20)
    model = make_tiny_official_qwen35_moe(vocab_size=32, hidden_size=8, moe_intermediate_size=8, num_hidden_layers=1)
    calibration_ids = torch.randint(0, model.config.vocab_size, (3, 4))
    eval_ids = torch.randint(0, model.config.vocab_size, (2, 4))

    q_model = convert_official_qwen35_moe_to_rcq(
        model,
        calibration_ids,
        RescueConfig.down_min2_5p4(block_size=8),
        rank=4,
        fit_correction=False,
        down_shared_mode="none",
        down_moment_mode="per_expert",
    )
    logits = q_model(input_ids=eval_ids, use_cache=False).logits

    assert isinstance(q_model.model.layers[0].mlp, OfficialQwen35MoeRCQSparseMoeBlock)
    assert q_model.model.layers[0].mlp.experts.q_down.shared_mode == "none"
    assert int((q_model.model.layers[0].mlp.experts.q_down.q_residuals[0].widths == 1).sum()) == 0
    assert torch.isfinite(logits).all()


def test_official_qwen_rcq_conversion_runs_with_down_left_output_mode():
    torch.manual_seed(21)
    model = make_tiny_official_qwen35_moe(vocab_size=32, hidden_size=8, moe_intermediate_size=8, num_hidden_layers=1)
    calibration_ids = torch.randint(0, model.config.vocab_size, (3, 4))
    eval_ids = torch.randint(0, model.config.vocab_size, (2, 4))

    q_model = convert_official_qwen35_moe_to_rcq(
        model,
        calibration_ids,
        RescueConfig.down_min2_5p4(block_size=8),
        rank=4,
        fit_correction=False,
        down_shared_mode="left_output",
        down_moment_mode="per_expert",
    )
    logits = q_model(input_ids=eval_ids, use_cache=False).logits
    q_down = q_model.model.layers[0].mlp.experts.q_down

    assert q_down.shared_mode == "left_output"
    assert q_down.output_decomposition is not None
    assert q_down.output_decomposition.u_shared.shape == (model.config.hidden_size, 4)
    assert torch.isfinite(logits).all()
