from __future__ import annotations

import torch

from rcq_moe.stats import LinearCalibrationStats
from rcq_moe.quantization import RescueConfig
from scripts.qwen36_single_layer_rcq_ablation import (
    CachedBatch,
    LoadedLayer,
    _add_nmse_to_summary,
    _down_v2_experiments,
    _fp_output_stats,
    build_q_moe,
    collect_sequential_down_stats_from_cache,
    evaluate_docs,
    _getattr_any,
    _rank_for_divisor,
    _text_config,
)


class TinyTokenizer:
    eos_token_id = 1

    def __call__(self, text: str, **kwargs):
        max_length = kwargs["max_length"]
        ids = [max(1, (ord(ch) % 15) + 1) for ch in text][:max_length]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def _stats(num_experts: int, input_dim: int, rows: int, block_size: int) -> LinearCalibrationStats:
    blocks = (input_dim + block_size - 1) // block_size
    return LinearCalibrationStats(
        covariance=torch.eye(input_dim),
        expert_importance=torch.full((num_experts,), 1.0 / num_experts),
        rotated_second_moments=torch.ones(blocks, block_size),
        down_output_covariance=torch.eye(rows),
    )


def test_rank_for_divisor_uses_at_least_one_rank() -> None:
    assert _rank_for_divisor(8, 256) == 1
    assert _rank_for_divisor(257, 256) == 2


def test_dict_config_helpers_do_not_require_transformers_model_registration() -> None:
    config = {"model_type": "qwen3_5_moe", "text_config": {"hidden_size": 4, "rope_parameters": {"rope_type": "default"}}}

    text_config = _text_config(config)

    assert _getattr_any(text_config, ["hidden_size"]) == 4
    assert _getattr_any(text_config, ["rope_parameters"]) == {"rope_type": "default"}
    assert _getattr_any(text_config, ["missing"], "fallback") == "fallback"


def test_nmse_denominator_helpers_use_cached_fp_outputs() -> None:
    batches = [
        CachedBatch(
            hidden=torch.zeros(2, 3),
            router_weights=torch.ones(2, 1),
            selected_experts=torch.zeros(2, 1, dtype=torch.long),
            fp_output=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        )
    ]

    stats = _fp_output_stats(batches)
    summary = {"mse": 2.5, "rmse": 2.5**0.5, "max_abs": 2.0}
    _add_nmse_to_summary(summary, stats)

    assert stats["mean_square"] == 7.5
    assert stats["variance"] == 1.25
    assert summary["nmse_mean_square"] == 2.5 / 7.5
    assert summary["nmse_variance"] == 2.0


def test_single_layer_ablation_core_runs_on_tiny_weights() -> None:
    torch.manual_seed(0)
    num_experts = 2
    hidden = 8
    intermediate = 6
    vocab = 24
    block = 4
    layer = LoadedLayer(
        embed_tokens=torch.randn(vocab, hidden),
        input_norm_weight=torch.zeros(hidden),
        post_attention_norm_weight=torch.zeros(hidden),
        q_proj_weight=torch.randn(2 * hidden, hidden),
        q_proj_bias=None,
        k_proj_weight=torch.randn(hidden, hidden),
        k_proj_bias=None,
        v_proj_weight=torch.randn(hidden, hidden),
        v_proj_bias=None,
        o_proj_weight=torch.randn(hidden, hidden),
        o_proj_bias=None,
        q_norm_weight=torch.zeros(hidden),
        k_norm_weight=torch.zeros(hidden),
        router_weight=torch.randn(num_experts, hidden),
        gate_weight=torch.randn(num_experts, intermediate, hidden),
        up_weight=torch.randn(num_experts, intermediate, hidden),
        down_weight=torch.randn(num_experts, hidden, intermediate),
        shared_gate_weight=torch.randn(intermediate, hidden),
        shared_up_weight=torch.randn(intermediate, hidden),
        shared_down_weight=torch.randn(hidden, intermediate),
        shared_expert_gate_weight=torch.randn(1, hidden),
        hidden_act="silu",
        rms_norm_eps=1e-6,
        layer_type="full_attention",
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=hidden,
        attention_bias=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
        top_k=1,
    )
    stats = {
        "gate": _stats(num_experts, hidden, intermediate, block),
        "up": _stats(num_experts, hidden, intermediate, block),
        "down": _stats(num_experts, intermediate, hidden, block),
    }

    q_moe = build_q_moe(
        layer,
        stats,
        rank_divisor=256,
        use_hadamard=True,
        weighted_scale=True,
        rescue_config=None,
    )
    summary = evaluate_docs(
        ["abc def"],
        TinyTokenizer(),
        layer,
        q_moe,
        max_tokens_per_doc=8,
        device=torch.device("cpu"),
        include_shared=False,
        activation_source="proxy_embedding_norm",
    )

    assert summary["mse"] >= 0.0
    assert summary["rmse"] >= 0.0
    assert summary["max_abs"] >= 0.0


def test_down_v2_experiment_rows_and_none_mode_bpw_fields() -> None:
    rows = dict(_down_v2_experiments(block_size=4))
    assert set(rows) >= {
        "D0_current_baseline_legacy_right_rcq_1p75_correction",
        "D3_gate_up_1p55_down_none_min2_5p4_correction",
        "D4_gate_up_1p55_down_none_min2_20p4_correction",
        "D7_gate_up_1p75_down_left_output_min2_20p4",
    }
    assert rows["D3_gate_up_1p55_down_none_min2_5p4_correction"]["down_shared_mode"] == "none"
    assert rows["D3_gate_up_1p55_down_none_min2_5p4_correction"]["down_moment_mode"] == "per_expert"
    assert rows["D3_gate_up_1p55_down_none_min2_5p4_correction"]["down_rescue_config"].name == "down_min2_5p4"
    assert rows["D5_gate_up_1p55_down_left_output_mix_1bit"]["down_shared_mode"] == "left_output"
    assert rows["D5_gate_up_1p55_down_left_output_mix_1bit"]["down_sequential_stats"] is True
    assert rows["D7_gate_up_1p75_down_left_output_min2_20p4"]["down_rescue_config"].name == "down_min2_20p4"


def test_build_q_moe_down_none_uses_min2_widths() -> None:
    torch.manual_seed(2)
    num_experts = 2
    hidden = 8
    intermediate = 6
    block = 4
    layer = LoadedLayer(
        embed_tokens=torch.randn(16, hidden),
        input_norm_weight=torch.zeros(hidden),
        post_attention_norm_weight=torch.zeros(hidden),
        q_proj_weight=torch.randn(2 * hidden, hidden),
        q_proj_bias=None,
        k_proj_weight=torch.randn(hidden, hidden),
        k_proj_bias=None,
        v_proj_weight=torch.randn(hidden, hidden),
        v_proj_bias=None,
        o_proj_weight=torch.randn(hidden, hidden),
        o_proj_bias=None,
        q_norm_weight=torch.zeros(hidden),
        k_norm_weight=torch.zeros(hidden),
        router_weight=torch.randn(num_experts, hidden),
        gate_weight=torch.randn(num_experts, intermediate, hidden),
        up_weight=torch.randn(num_experts, intermediate, hidden),
        down_weight=torch.randn(num_experts, hidden, intermediate),
        shared_gate_weight=torch.randn(intermediate, hidden),
        shared_up_weight=torch.randn(intermediate, hidden),
        shared_down_weight=torch.randn(hidden, intermediate),
        shared_expert_gate_weight=torch.randn(1, hidden),
        hidden_act="silu",
        rms_norm_eps=1e-6,
        layer_type="full_attention",
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=hidden,
        attention_bias=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
        top_k=1,
    )
    down_stats = _stats(num_experts, intermediate, hidden, block)
    down_stats.per_expert_rotated_second_moments = torch.ones(num_experts, (intermediate + block - 1) // block, block)
    stats = {
        "gate": _stats(num_experts, hidden, intermediate, block),
        "up": _stats(num_experts, hidden, intermediate, block),
        "down": down_stats,
    }

    q_moe = build_q_moe(
        layer,
        stats,
        rank_divisor=256,
        use_hadamard=True,
        weighted_scale=True,
        rescue_config=None,
        gate_rescue_config=None,
        up_rescue_config=None,
        down_rescue_config=RescueConfig.down_min2_5p4(block),
        down_shared_mode="none",
        down_moment_mode="per_expert",
    )

    assert q_moe.down.shared_mode == "none"
    assert q_moe.down.decomposition.b_shared.shape[0] == 0
    assert int((q_moe.down.widths == 1).sum()) == 0
    assert q_moe.down.bpw > 0.0


def test_build_q_moe_down_left_output_uses_output_decomposition() -> None:
    torch.manual_seed(3)
    num_experts = 2
    hidden = 8
    intermediate = 6
    block = 4
    layer = LoadedLayer(
        embed_tokens=torch.randn(16, hidden),
        input_norm_weight=torch.zeros(hidden),
        post_attention_norm_weight=torch.zeros(hidden),
        q_proj_weight=torch.randn(2 * hidden, hidden),
        q_proj_bias=None,
        k_proj_weight=torch.randn(hidden, hidden),
        k_proj_bias=None,
        v_proj_weight=torch.randn(hidden, hidden),
        v_proj_bias=None,
        o_proj_weight=torch.randn(hidden, hidden),
        o_proj_bias=None,
        q_norm_weight=torch.zeros(hidden),
        k_norm_weight=torch.zeros(hidden),
        router_weight=torch.randn(num_experts, hidden),
        gate_weight=torch.randn(num_experts, intermediate, hidden),
        up_weight=torch.randn(num_experts, intermediate, hidden),
        down_weight=torch.randn(num_experts, hidden, intermediate),
        shared_gate_weight=torch.randn(intermediate, hidden),
        shared_up_weight=torch.randn(intermediate, hidden),
        shared_down_weight=torch.randn(hidden, intermediate),
        shared_expert_gate_weight=torch.randn(1, hidden),
        hidden_act="silu",
        rms_norm_eps=1e-6,
        layer_type="full_attention",
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=hidden,
        attention_bias=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
        top_k=1,
    )
    down_stats = _stats(num_experts, intermediate, hidden, block)
    down_stats.per_expert_rotated_second_moments = torch.ones(num_experts, (intermediate + block - 1) // block, block)
    stats = {
        "gate": _stats(num_experts, hidden, intermediate, block),
        "up": _stats(num_experts, hidden, intermediate, block),
        "down": down_stats,
    }

    q_moe = build_q_moe(
        layer,
        stats,
        rank_divisor=256,
        use_hadamard=True,
        weighted_scale=True,
        rescue_config=None,
        down_rescue_config=RescueConfig.down_mix_1bit(block),
        down_shared_mode="left_output",
        down_moment_mode="per_expert",
    )

    assert q_moe.down.shared_mode == "left_output"
    assert q_moe.down.output_decomposition is not None
    assert q_moe.down.output_decomposition.u_shared.shape == (hidden, 1)
    assert q_moe.down.weight_for_expert(0).shape == (hidden, intermediate)

    batches = [
        CachedBatch(
            hidden=torch.randn(3, hidden),
            router_weights=torch.ones(3, 1),
            selected_experts=torch.tensor([[0], [1], [0]]),
            fp_output=torch.zeros(3, hidden),
        )
    ]
    sequential = collect_sequential_down_stats_from_cache(
        batches,
        layer,
        q_moe.gate,
        q_moe.up,
        block_size=block,
        device=torch.device("cpu"),
        weighted=True,
    )
    assert sequential.down_output_covariance is not None
    assert sequential.down_output_covariance.shape == (hidden, hidden)
    assert sequential.per_expert_rotated_second_moments is not None
    assert torch.isfinite(sequential.per_expert_rotated_second_moments).all()


def test_true_layer0_attention_activation_path_runs_on_tiny_weights() -> None:
    torch.manual_seed(1)
    hidden = 8
    intermediate = 6
    layer = LoadedLayer(
        embed_tokens=torch.randn(24, hidden),
        input_norm_weight=torch.zeros(hidden),
        post_attention_norm_weight=torch.zeros(hidden),
        q_proj_weight=torch.randn(2 * hidden, hidden),
        q_proj_bias=None,
        k_proj_weight=torch.randn(hidden, hidden),
        k_proj_bias=None,
        v_proj_weight=torch.randn(hidden, hidden),
        v_proj_bias=None,
        o_proj_weight=torch.randn(hidden, hidden),
        o_proj_bias=None,
        q_norm_weight=torch.zeros(hidden),
        k_norm_weight=torch.zeros(hidden),
        router_weight=torch.randn(2, hidden),
        gate_weight=torch.randn(2, intermediate, hidden),
        up_weight=torch.randn(2, intermediate, hidden),
        down_weight=torch.randn(2, hidden, intermediate),
        shared_gate_weight=torch.randn(intermediate, hidden),
        shared_up_weight=torch.randn(intermediate, hidden),
        shared_down_weight=torch.randn(hidden, intermediate),
        shared_expert_gate_weight=torch.randn(1, hidden),
        hidden_act="silu",
        rms_norm_eps=1e-6,
        layer_type="full_attention",
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=hidden,
        attention_bias=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
        top_k=1,
    )

    summary = evaluate_docs(
        ["proper attention path"],
        TinyTokenizer(),
        layer,
        None,
        max_tokens_per_doc=8,
        device=torch.device("cpu"),
        include_shared=True,
        activation_source="true_layer0_post_attention_norm",
    )

    assert summary == {"mse": 0.0, "rmse": 0.0, "max_abs": 0.0}
