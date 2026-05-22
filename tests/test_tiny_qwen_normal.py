import torch

from rcq_moe.tiny_qwen import TinyQwen35MoeConfig, TinyQwen35MoeForCausalLM


def test_tiny_qwen_moe_normal_model_runs_and_uses_qwen_expert_shapes():
    torch.manual_seed(10)
    config = TinyQwen35MoeConfig(
        vocab_size=64,
        hidden_size=16,
        moe_intermediate_size=24,
        shared_expert_intermediate_size=12,
        num_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
    )
    model = TinyQwen35MoeForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (3, 5))

    logits = model(input_ids)

    assert logits.shape == (3, 5, config.vocab_size)
    assert torch.isfinite(logits).all()
    experts = model.layers[0].mlp.experts
    assert experts.gate_up_proj.shape == (config.num_experts, 2 * config.moe_intermediate_size, config.hidden_size)
    assert experts.down_proj.shape == (config.num_experts, config.hidden_size, config.moe_intermediate_size)


def test_tiny_qwen_router_outputs_normalized_topk_weights():
    torch.manual_seed(11)
    config = TinyQwen35MoeConfig(hidden_size=8, num_experts=5, num_experts_per_tok=3, num_hidden_layers=1)
    model = TinyQwen35MoeForCausalLM(config)
    hidden = torch.randn(7, config.hidden_size)

    logits, weights, indices = model.layers[0].mlp.gate(hidden)

    assert logits.shape == (7, config.num_experts)
    assert weights.shape == (7, config.num_experts_per_tok)
    assert indices.shape == (7, config.num_experts_per_tok)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(7), atol=1e-6)

