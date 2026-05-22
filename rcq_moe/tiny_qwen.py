from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .correction import OnlineChannelRegression
from .decomposition import SharedDecomposition, decompose_shared_subspace
from .quantization import QuantizedResidual, RescueConfig, quantize_residuals
from .stats import LinearCalibrationStats, accumulate_covariance_and_moments


@dataclass(frozen=True)
class TinyQwen35MoeConfig:
    vocab_size: int = 128
    hidden_size: int = 16
    moe_intermediate_size: int = 32
    shared_expert_intermediate_size: int = 16
    num_experts: int = 4
    num_experts_per_tok: int = 2
    num_hidden_layers: int = 2
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02


def _act_fn(name: str):
    if name == "silu":
        return F.silu
    if name == "gelu":
        return F.gelu
    raise ValueError(f"unsupported activation {name!r}")


class TinyQwen35MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance.to(dtype=x.dtype) + self.eps)
        return x * (1.0 + self.weight)


class TinyQwen35MoeMLP(nn.Module):
    def __init__(self, config: TinyQwen35MoeConfig, intermediate_size: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = _act_fn(config.hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TinyQwen35MoeExperts(nn.Module):
    """Qwen-shaped packed expert storage: [E, 2I, H] and [E, H, I]."""

    def __init__(self, config: TinyQwen35MoeConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = _act_fn(config.hidden_act)

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
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


class TinyQwen35MoeTopKRouter(nn.Module):
    def __init__(self, config: TinyQwen35MoeConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        return router_logits, router_top_value.to(router_logits.dtype), router_indices


class TinyQwen35MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: TinyQwen35MoeConfig):
        super().__init__()
        self.gate = TinyQwen35MoeTopKRouter(config)
        self.experts = TinyQwen35MoeExperts(config)
        self.shared_expert = TinyQwen35MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(flat)
        _, routing_weights, selected_experts = self.gate(flat)
        expert_output = self.experts(flat, selected_experts, routing_weights)
        shared_expert_output = torch.sigmoid(self.shared_expert_gate(flat)) * shared_expert_output
        return (expert_output + shared_expert_output).reshape(batch_size, sequence_length, hidden_dim)


class TinyQwen35MoeDecoderLayer(nn.Module):
    def __init__(self, config: TinyQwen35MoeConfig):
        super().__init__()
        self.post_attention_layernorm = TinyQwen35MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = TinyQwen35MoeSparseMoeBlock(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class TinyQwen35MoeForCausalLM(nn.Module):
    """A tiny Qwen-shaped MoE-only LM shell for RCQ conversion tests."""

    def __init__(self, config: TinyQwen35MoeConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TinyQwen35MoeDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = TinyQwen35MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, TinyQwen35MoeExperts):
                nn.init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
                nn.init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)


@dataclass
class TinyQwen35RCQLinearSet:
    decomposition: SharedDecomposition
    q_residuals: list[QuantizedResidual]

    def forward(self, hidden: torch.Tensor, expert_id: int) -> torch.Tensor:
        a_factor = self.decomposition.a_factors[expert_id]
        shared = (hidden @ self.decomposition.b_shared.T) @ a_factor.T
        residual = hidden @ self.q_residuals[expert_id].dequantize().T
        return shared + residual


class TinyQwen35MoeRCQExperts(nn.Module):
    def __init__(
        self,
        config: TinyQwen35MoeConfig,
        q_gate: TinyQwen35RCQLinearSet,
        q_up: TinyQwen35RCQLinearSet,
        q_down: TinyQwen35RCQLinearSet,
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.q_gate = q_gate
        self.q_up = q_up
        self.q_down = q_down
        self.act_fn = _act_fn(config.hidden_act)

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


class TinyQwen35MoeRCQSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: TinyQwen35MoeConfig,
        gate: TinyQwen35MoeTopKRouter,
        experts: TinyQwen35MoeRCQExperts,
        shared_expert: TinyQwen35MoeMLP,
        shared_expert_gate: nn.Linear,
    ):
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate
        self.correction_alpha: torch.Tensor | None = None
        self.correction_beta: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor, *, apply_correction: bool = True) -> torch.Tensor:
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


class TinyQwen35MoeRCQDecoderLayer(nn.Module):
    def __init__(self, norm: TinyQwen35MoeRMSNorm, mlp: TinyQwen35MoeRCQSparseMoeBlock):
        super().__init__()
        self.post_attention_layernorm = norm
        self.mlp = mlp

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class TinyQwen35MoeRCQForCausalLM(nn.Module):
    def __init__(
        self,
        config: TinyQwen35MoeConfig,
        embed_tokens: nn.Embedding,
        layers: list[TinyQwen35MoeRCQDecoderLayer],
        norm: TinyQwen35MoeRMSNorm,
        lm_head: nn.Linear,
    ):
        super().__init__()
        self.config = config
        self.embed_tokens = embed_tokens
        self.layers = nn.ModuleList(layers)
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)


def _clone_module(module: nn.Module) -> nn.Module:
    import copy

    return copy.deepcopy(module)


def _collect_qwen_block_stats(
    block: TinyQwen35MoeSparseMoeBlock,
    flat_hidden: torch.Tensor,
    rcq_config: RescueConfig,
    *,
    model_name: str,
    layer_id: int,
) -> dict[str, LinearCalibrationStats]:
    _, routing_weights, selected_experts = block.gate(flat_hidden)
    num_experts = block.experts.num_experts
    gate_acts: list[list[torch.Tensor]] = [[] for _ in range(num_experts)]
    gate_router: list[list[torch.Tensor]] = [[] for _ in range(num_experts)]
    up_acts: list[list[torch.Tensor]] = [[] for _ in range(num_experts)]
    up_router: list[list[torch.Tensor]] = [[] for _ in range(num_experts)]
    down_acts: list[list[torch.Tensor]] = [[] for _ in range(num_experts)]
    down_router: list[list[torch.Tensor]] = [[] for _ in range(num_experts)]

    intermediate = block.experts.intermediate_dim
    for token_id in range(flat_hidden.shape[0]):
        token = flat_hidden[token_id]
        for slot in range(selected_experts.shape[1]):
            expert_id = int(selected_experts[token_id, slot].item())
            router_weight = routing_weights[token_id, slot]
            gate_acts[expert_id].append(token)
            gate_router[expert_id].append(router_weight)
            up_acts[expert_id].append(token)
            up_router[expert_id].append(router_weight)

            packed = F.linear(token, block.experts.gate_up_proj[expert_id])
            gate, up = packed[:intermediate], packed[intermediate:]
            down_acts[expert_id].append(block.experts.act_fn(gate) * up)
            down_router[expert_id].append(router_weight)

    return {
        "gate": accumulate_covariance_and_moments(
            gate_acts,
            gate_router,
            num_experts=num_experts,
            input_dim=block.experts.hidden_dim,
            block_size=rcq_config.block_size,
            model_name=model_name,
            layer_id=layer_id,
            linear_type="gate",
        ),
        "up": accumulate_covariance_and_moments(
            up_acts,
            up_router,
            num_experts=num_experts,
            input_dim=block.experts.hidden_dim,
            block_size=rcq_config.block_size,
            model_name=model_name,
            layer_id=layer_id,
            linear_type="up",
        ),
        "down": accumulate_covariance_and_moments(
            down_acts,
            down_router,
            num_experts=num_experts,
            input_dim=block.experts.intermediate_dim,
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
) -> TinyQwen35RCQLinearSet:
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
    return TinyQwen35RCQLinearSet(decomposition=decomposition, q_residuals=q_residuals)


def _convert_sparse_moe_block(
    block: TinyQwen35MoeSparseMoeBlock,
    flat_hidden: torch.Tensor,
    config: TinyQwen35MoeConfig,
    rcq_config: RescueConfig,
    *,
    model_name: str,
    layer_id: int,
    rank: int | None,
) -> TinyQwen35MoeRCQSparseMoeBlock:
    stats = _collect_qwen_block_stats(block, flat_hidden, rcq_config, model_name=model_name, layer_id=layer_id)
    intermediate = config.moe_intermediate_size
    gate_weights = block.experts.gate_up_proj[:, :intermediate, :]
    up_weights = block.experts.gate_up_proj[:, intermediate:, :]
    down_weights = block.experts.down_proj
    q_experts = TinyQwen35MoeRCQExperts(
        config,
        q_gate=_quantize_linear_set(gate_weights, stats["gate"], rcq_config, model_name=model_name, layer_id=layer_id, linear_type="gate", rank=rank),
        q_up=_quantize_linear_set(up_weights, stats["up"], rcq_config, model_name=model_name, layer_id=layer_id, linear_type="up", rank=rank),
        q_down=_quantize_linear_set(down_weights, stats["down"], rcq_config, model_name=model_name, layer_id=layer_id, linear_type="down", rank=rank),
    )
    return TinyQwen35MoeRCQSparseMoeBlock(
        config,
        gate=_clone_module(block.gate),
        experts=q_experts,
        shared_expert=_clone_module(block.shared_expert),
        shared_expert_gate=_clone_module(block.shared_expert_gate),
    )


def _fit_layer_correction(
    fp_layer: TinyQwen35MoeDecoderLayer,
    q_layer: TinyQwen35MoeRCQDecoderLayer,
    hidden_states: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> None:
    normed = q_layer.post_attention_layernorm(hidden_states)
    with torch.no_grad():
        y_fp = fp_layer.mlp(normed)
        y_q = q_layer.mlp(normed, apply_correction=False)
    stats = OnlineChannelRegression(dim=hidden_states.shape[-1], dtype=hidden_states.dtype, device=hidden_states.device)
    stats.update(y_fp, y_q)
    alpha, beta = stats.solve(eps=eps)
    q_layer.mlp.correction_alpha = alpha
    q_layer.mlp.correction_beta = beta


def convert_tiny_qwen_to_rcq(
    model: TinyQwen35MoeForCausalLM,
    calibration_input_ids: torch.Tensor,
    rcq_config: RescueConfig,
    *,
    model_name: str = "tiny-qwen3.5-moe",
    rank: int | None = None,
    fit_correction: bool = True,
) -> TinyQwen35MoeRCQForCausalLM:
    """Convert a tiny Qwen-shaped FP MoE model into an RCQ-backed model."""
    config = model.config
    model.eval()
    with torch.no_grad():
        hidden_states = model.embed_tokens(calibration_input_ids)
        q_layers: list[TinyQwen35MoeRCQDecoderLayer] = []
        for layer_id, fp_layer in enumerate(model.layers):
            normed = fp_layer.post_attention_layernorm(hidden_states)
            flat_normed = normed.reshape(-1, config.hidden_size)
            q_mlp = _convert_sparse_moe_block(
                fp_layer.mlp,
                flat_normed,
                config,
                rcq_config,
                model_name=model_name,
                layer_id=layer_id,
                rank=rank,
            )
            q_layers.append(TinyQwen35MoeRCQDecoderLayer(_clone_module(fp_layer.post_attention_layernorm), q_mlp))
            hidden_states = fp_layer(hidden_states)

        q_model = TinyQwen35MoeRCQForCausalLM(
            config=config,
            embed_tokens=_clone_module(model.embed_tokens),
            layers=q_layers,
            norm=_clone_module(model.norm),
            lm_head=_clone_module(model.lm_head),
        )

        if fit_correction:
            q_hidden_states = q_model.embed_tokens(calibration_input_ids)
            for layer_id, q_layer in enumerate(q_model.layers):
                _fit_layer_correction(model.layers[layer_id], q_layer, q_hidden_states)
                q_hidden_states = q_layer(q_hidden_states)
    return q_model

