from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .correction import OnlineChannelRegression
from .decomposition import SharedDecomposition, decompose_shared_subspace
from .quantization import QuantizedResidual, RescueConfig, quantize_residuals
from .stats import LinearCalibrationStats, accumulate_covariance_and_moments


@dataclass
class OfficialQwen35RCQLinearSet:
    decomposition: SharedDecomposition
    q_residuals: list[QuantizedResidual]

    def forward(self, hidden: torch.Tensor, expert_id: int) -> torch.Tensor:
        a_factor = self.decomposition.a_factors[expert_id]
        shared = (hidden @ self.decomposition.b_shared.T) @ a_factor.T
        residual = hidden @ self.q_residuals[expert_id].dequantize().T
        return shared + residual


@dataclass(frozen=True)
class OfficialQwen35LinearDiagnostics:
    linear_type: str
    captured_energy: float
    bpw: float
    width_percentages: dict[int, float]


@dataclass(frozen=True)
class OfficialQwen35LayerDiagnostics:
    layer_id: int
    linear_diagnostics: dict[str, OfficialQwen35LinearDiagnostics]
    moe_mse_before_correction: float | None = None
    moe_mse_after_correction: float | None = None


@dataclass
class OfficialQwen35RCQConversionResult:
    model: nn.Module
    layer_diagnostics: list[OfficialQwen35LayerDiagnostics]


class OfficialQwen35MoeRCQExperts(nn.Module):
    """Drop-in replacement for official Qwen3.5-MoE experts."""

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        act_fn,
        q_gate: OfficialQwen35RCQLinearSet,
        q_up: OfficialQwen35RCQLinearSet,
        q_down: OfficialQwen35RCQLinearSet,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = act_fn
        self.q_gate = q_gate
        self.q_up = q_up
        self.q_down = q_down

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_tensor in expert_hit:
            expert_idx = int(expert_idx_tensor[0].item())
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate = self.q_gate.forward(current_state, expert_idx)
            up = self.q_up.forward(current_state, expert_idx)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = self.q_down.forward(current_hidden_states, expert_idx)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


def _make_linear_set_from_state(state: dict[str, torch.Tensor | int | str], prefix: str) -> OfficialQwen35RCQLinearSet:
    residuals: list[QuantizedResidual] = []
    num_experts = int(state[f"{prefix}.num_experts"])
    residual_template = state[f"{prefix}.q0.values"]
    for expert_id in range(num_experts):
        residuals.append(
            QuantizedResidual(
                values=state[f"{prefix}.q{expert_id}.values"],
                widths=state[f"{prefix}.q{expert_id}.widths"],
                scales=state[f"{prefix}.q{expert_id}.scales"],
                valid_cols=int(state[f"{prefix}.q{expert_id}.valid_cols"]),
                block_size=int(state[f"{prefix}.q{expert_id}.block_size"]),
                model_name=str(state[f"{prefix}.q{expert_id}.model_name"]),
                layer_id=int(state[f"{prefix}.q{expert_id}.layer_id"]),
                linear_type=str(state[f"{prefix}.q{expert_id}.linear_type"]),
                expert_id=int(state[f"{prefix}.q{expert_id}.expert_id"]),
                scores=state[f"{prefix}.q{expert_id}.scores"],
            )
        )
    decomposition = SharedDecomposition(
        a_factors=[state[f"{prefix}.a{expert_id}"] for expert_id in range(num_experts)],
        b_shared=state[f"{prefix}.b_shared"],
        residuals=[torch.empty(0, dtype=residual_template.dtype, device=residual_template.device) for _ in range(num_experts)],
        eigvals=state[f"{prefix}.eigvals"],
        eigvecs=state[f"{prefix}.eigvecs"],
        v_r=state[f"{prefix}.v_r"],
        captured_energy=float(state[f"{prefix}.captured_energy"]),
    )
    return OfficialQwen35RCQLinearSet(decomposition=decomposition, q_residuals=residuals)


class OfficialQwen35MoeRCQSparseMoeBlock(nn.Module):
    """Official-Qwen compatible sparse MoE block with RCQ expert weights."""

    def __init__(
        self,
        *,
        gate: nn.Module,
        experts: OfficialQwen35MoeRCQExperts,
        shared_expert: nn.Module,
        shared_expert_gate: nn.Module,
    ):
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate
        self.correction_alpha: torch.Tensor | None = None
        self.correction_beta: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor, *args, apply_correction: bool = True, **kwargs) -> torch.Tensor:
        del args, kwargs
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(flat)
        _, routing_weights, selected_experts = self.gate(flat)
        expert_output = self.experts(flat, selected_experts, routing_weights)
        shared_expert_output = torch.sigmoid(self.shared_expert_gate(flat)) * shared_expert_output
        output = expert_output + shared_expert_output
        if apply_correction and self.correction_alpha is not None and self.correction_beta is not None:
            output = self.correction_alpha.to(output.device, output.dtype) * output + self.correction_beta.to(output.device, output.dtype)
        return output.reshape(batch_size, sequence_length, hidden_dim)


def make_tiny_official_qwen35_moe(
    *,
    vocab_size: int = 64,
    hidden_size: int = 16,
    moe_intermediate_size: int = 24,
    shared_expert_intermediate_size: int = 12,
    num_experts: int = 4,
    num_experts_per_tok: int = 2,
    num_hidden_layers: int = 1,
):
    """Instantiate a tiny official Transformers Qwen3.5-MoE CausalLM."""
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    config = Qwen3_5MoeTextConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=max(2 * hidden_size, moe_intermediate_size),
        moe_intermediate_size=moe_intermediate_size,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        layer_types=["full_attention"] * num_hidden_layers,
        head_dim=hidden_size // 2,
        max_position_embeddings=64,
        use_cache=False,
    )
    return Qwen3_5MoeForCausalLM(config)


def collect_official_qwen_mlp_inputs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Run the official model and capture inputs to each layer's MoE block."""
    captures: list[torch.Tensor | None] = [None] * len(model.model.layers)
    handles = []

    for layer_id, layer in enumerate(model.model.layers):
        def hook(_module, args, _layer_id=layer_id):
            captures[_layer_id] = args[0].detach().clone()

        handles.append(layer.mlp.register_forward_pre_hook(hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    if was_training:
        model.train()
    for handle in handles:
        handle.remove()
    if any(capture is None for capture in captures):
        raise RuntimeError("failed to capture all official Qwen MoE inputs.")
    return [capture for capture in captures if capture is not None]


def _collect_block_stats(
    block: nn.Module,
    mlp_input: torch.Tensor,
    rcq_config: RescueConfig,
    *,
    model_name: str,
    layer_id: int,
) -> dict[str, LinearCalibrationStats]:
    flat_hidden = mlp_input.reshape(-1, mlp_input.shape[-1])
    _, routing_weights, selected_experts = block.gate(flat_hidden)
    experts = block.experts
    num_experts = experts.num_experts
    hidden_dim = experts.hidden_dim
    intermediate_dim = experts.intermediate_dim

    gate_acts: list[list[torch.Tensor]] = [[] for _ in range(num_experts)]
    gate_router: list[list[torch.Tensor]] = [[] for _ in range(num_experts)]
    up_acts: list[list[torch.Tensor]] = [[] for _ in range(num_experts)]
    up_router: list[list[torch.Tensor]] = [[] for _ in range(num_experts)]
    down_acts: list[list[torch.Tensor]] = [[] for _ in range(num_experts)]
    down_router: list[list[torch.Tensor]] = [[] for _ in range(num_experts)]

    for token_id in range(flat_hidden.shape[0]):
        token = flat_hidden[token_id]
        for slot in range(selected_experts.shape[1]):
            expert_id = int(selected_experts[token_id, slot].item())
            router_weight = routing_weights[token_id, slot]
            gate_acts[expert_id].append(token)
            gate_router[expert_id].append(router_weight)
            up_acts[expert_id].append(token)
            up_router[expert_id].append(router_weight)

            packed = F.linear(token, experts.gate_up_proj[expert_id])
            gate, up = packed[:intermediate_dim], packed[intermediate_dim:]
            down_acts[expert_id].append(experts.act_fn(gate) * up)
            down_router[expert_id].append(router_weight)

    return {
        "gate": accumulate_covariance_and_moments(
            gate_acts,
            gate_router,
            num_experts=num_experts,
            input_dim=hidden_dim,
            block_size=rcq_config.block_size,
            model_name=model_name,
            layer_id=layer_id,
            linear_type="gate",
        ),
        "up": accumulate_covariance_and_moments(
            up_acts,
            up_router,
            num_experts=num_experts,
            input_dim=hidden_dim,
            block_size=rcq_config.block_size,
            model_name=model_name,
            layer_id=layer_id,
            linear_type="up",
        ),
        "down": accumulate_covariance_and_moments(
            down_acts,
            down_router,
            num_experts=num_experts,
            input_dim=intermediate_dim,
            block_size=rcq_config.block_size,
            model_name=model_name,
            layer_id=layer_id,
            linear_type="down",
        ),
    }


def _quantize_linear_set(
    weights: torch.Tensor,
    stats: LinearCalibrationStats,
    rcq_config: RescueConfig,
    *,
    model_name: str,
    layer_id: int,
    linear_type: str,
    rank: int | None,
) -> OfficialQwen35RCQLinearSet:
    expert_weights = [weights[expert_id].detach().clone() for expert_id in range(weights.shape[0])]
    decomposition = decompose_shared_subspace(expert_weights, stats.covariance, stats.expert_importance, rank=rank)
    q_residuals = quantize_residuals(
        decomposition.residuals,
        stats.rotated_second_moments,
        rcq_config,
        model_name=model_name,
        layer_id=layer_id,
        linear_type=linear_type,
    )
    return OfficialQwen35RCQLinearSet(decomposition=decomposition, q_residuals=q_residuals)


def _linear_diagnostics(
    linear_type: str,
    linear_set: OfficialQwen35RCQLinearSet,
    *,
    rows: int,
    cols: int,
    rank: int,
    block_size: int,
    scale_bits: int,
) -> OfficialQwen35LinearDiagnostics:
    from .storage import expert_bpw

    widths = torch.stack([q.widths for q in linear_set.q_residuals], dim=0)
    report = expert_bpw(
        num_experts=widths.shape[0],
        rows=rows,
        cols=cols,
        rank=rank,
        widths=widths,
        block_size=block_size,
        scale_bits_per_block=scale_bits,
    )
    total = widths.numel()
    percentages = {bit: float((widths == bit).sum().item() / total) for bit in (1, 2, 4)}
    return OfficialQwen35LinearDiagnostics(
        linear_type=linear_type,
        captured_energy=linear_set.decomposition.captured_energy,
        bpw=report.bpw,
        width_percentages=percentages,
    )


def _convert_block(
    fp_block: nn.Module,
    mlp_input: torch.Tensor,
    rcq_config: RescueConfig,
    *,
    model_name: str,
    layer_id: int,
    rank: int | None,
) -> tuple[OfficialQwen35MoeRCQSparseMoeBlock, dict[str, OfficialQwen35LinearDiagnostics]]:
    stats = _collect_block_stats(fp_block, mlp_input, rcq_config, model_name=model_name, layer_id=layer_id)
    experts = fp_block.experts
    intermediate_dim = experts.intermediate_dim
    gate_weights = experts.gate_up_proj[:, :intermediate_dim, :]
    up_weights = experts.gate_up_proj[:, intermediate_dim:, :]
    down_weights = experts.down_proj
    q_gate = _quantize_linear_set(gate_weights, stats["gate"], rcq_config, model_name=model_name, layer_id=layer_id, linear_type="gate", rank=rank)
    q_up = _quantize_linear_set(up_weights, stats["up"], rcq_config, model_name=model_name, layer_id=layer_id, linear_type="up", rank=rank)
    q_down = _quantize_linear_set(down_weights, stats["down"], rcq_config, model_name=model_name, layer_id=layer_id, linear_type="down", rank=rank)
    q_experts = OfficialQwen35MoeRCQExperts(
        num_experts=experts.num_experts,
        hidden_dim=experts.hidden_dim,
        intermediate_dim=intermediate_dim,
        act_fn=experts.act_fn,
        q_gate=q_gate,
        q_up=q_up,
        q_down=q_down,
    )
    q_block = OfficialQwen35MoeRCQSparseMoeBlock(
        gate=copy.deepcopy(fp_block.gate),
        experts=q_experts,
        shared_expert=copy.deepcopy(fp_block.shared_expert),
        shared_expert_gate=copy.deepcopy(fp_block.shared_expert_gate),
    )
    rank_gate = q_gate.decomposition.b_shared.shape[0]
    rank_up = q_up.decomposition.b_shared.shape[0]
    rank_down = q_down.decomposition.b_shared.shape[0]
    diagnostics = {
        "gate": _linear_diagnostics("gate", q_gate, rows=intermediate_dim, cols=experts.hidden_dim, rank=rank_gate, block_size=rcq_config.block_size, scale_bits=rcq_config.scale_bits),
        "up": _linear_diagnostics("up", q_up, rows=intermediate_dim, cols=experts.hidden_dim, rank=rank_up, block_size=rcq_config.block_size, scale_bits=rcq_config.scale_bits),
        "down": _linear_diagnostics("down", q_down, rows=experts.hidden_dim, cols=intermediate_dim, rank=rank_down, block_size=rcq_config.block_size, scale_bits=rcq_config.scale_bits),
    }
    return q_block, diagnostics


def _fit_correction(
    fp_block: nn.Module,
    q_block: OfficialQwen35MoeRCQSparseMoeBlock,
    mlp_input: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[float, float]:
    with torch.no_grad():
        y_fp = fp_block(mlp_input)
        y_q = q_block(mlp_input, apply_correction=False)
    before = torch.mean((y_fp - y_q).square()).item()
    stats = OnlineChannelRegression(dim=mlp_input.shape[-1], dtype=mlp_input.dtype, device=mlp_input.device)
    stats.update(y_fp, y_q)
    alpha, beta = stats.solve(eps=eps)
    q_block.correction_alpha = alpha
    q_block.correction_beta = beta
    with torch.no_grad():
        y_corr = q_block(mlp_input, apply_correction=True)
    after = torch.mean((y_fp - y_corr).square()).item()
    return before, after


def convert_official_qwen35_moe_to_rcq(
    model: nn.Module,
    calibration_input_ids: torch.Tensor,
    rcq_config: RescueConfig,
    *,
    calibration_attention_mask: torch.Tensor | None = None,
    model_name: str = "official-qwen3.5-moe",
    rank: int | None = None,
    fit_correction: bool = True,
) -> nn.Module:
    """Deep-copy and convert official Transformers Qwen3.5-MoE experts to RCQ."""
    return convert_official_qwen35_moe_to_rcq_with_diagnostics(
        model,
        calibration_input_ids,
        rcq_config,
        calibration_attention_mask=calibration_attention_mask,
        model_name=model_name,
        rank=rank,
        fit_correction=fit_correction,
    ).model


def convert_official_qwen35_moe_to_rcq_with_diagnostics(
    model: nn.Module,
    calibration_input_ids: torch.Tensor,
    rcq_config: RescueConfig,
    *,
    calibration_attention_mask: torch.Tensor | None = None,
    model_name: str = "official-qwen3.5-moe",
    rank: int | None = None,
    fit_correction: bool = True,
) -> OfficialQwen35RCQConversionResult:
    """Deep-copy and convert official Transformers Qwen3.5-MoE experts to RCQ with diagnostics."""
    model.eval()
    mlp_inputs = collect_official_qwen_mlp_inputs(model, calibration_input_ids, attention_mask=calibration_attention_mask)
    q_model = copy.deepcopy(model)
    q_model.eval()
    layer_diagnostics: list[OfficialQwen35LayerDiagnostics] = []

    for layer_id, (fp_layer, q_layer, mlp_input) in enumerate(zip(model.model.layers, q_model.model.layers, mlp_inputs)):
        q_block, linear_diags = _convert_block(
            fp_layer.mlp,
            mlp_input,
            rcq_config,
            model_name=model_name,
            layer_id=layer_id,
            rank=rank,
        )
        before = None
        after = None
        if fit_correction:
            before, after = _fit_correction(fp_layer.mlp, q_block, mlp_input)
        q_layer.mlp = q_block
        layer_diagnostics.append(
            OfficialQwen35LayerDiagnostics(
                layer_id=layer_id,
                linear_diagnostics=linear_diags,
                moe_mse_before_correction=before,
                moe_mse_after_correction=after,
            )
        )
    return OfficialQwen35RCQConversionResult(model=q_model, layer_diagnostics=layer_diagnostics)
