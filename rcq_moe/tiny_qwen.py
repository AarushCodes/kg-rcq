from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


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

