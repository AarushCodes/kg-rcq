from __future__ import annotations

from dataclasses import dataclass

import torch

from .decomposition import SharedDecomposition, decompose_shared_subspace
from .quantization import QuantizedResidual, RescueConfig, quantize_residuals
from .stats import LinearCalibrationStats, accumulate_covariance_and_moments


@dataclass
class ToyMoeLayer:
    gate: torch.Tensor
    expert_gate: torch.Tensor
    expert_up: torch.Tensor
    expert_down: torch.Tensor
    top_k: int

    @property
    def num_experts(self) -> int:
        return self.expert_gate.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.expert_gate.shape[2]

    @property
    def intermediate_size(self) -> int:
        return self.expert_gate.shape[1]

    @staticmethod
    def random(
        *,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> "ToyMoeLayer":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        scale = hidden_size**-0.5
        return ToyMoeLayer(
            gate=torch.randn(num_experts, hidden_size, generator=generator, dtype=dtype) * scale,
            expert_gate=torch.randn(num_experts, intermediate_size, hidden_size, generator=generator, dtype=dtype) * scale,
            expert_up=torch.randn(num_experts, intermediate_size, hidden_size, generator=generator, dtype=dtype) * scale,
            expert_down=torch.randn(num_experts, hidden_size, intermediate_size, generator=generator, dtype=dtype) * (intermediate_size**-0.5),
            top_k=top_k,
        )

    def route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = hidden @ self.gate.T
        probs = torch.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, k=self.top_k, dim=-1)
        return indices, weights

    def expert_forward(self, hidden: torch.Tensor, expert_id: int) -> torch.Tensor:
        gate = torch.nn.functional.silu(hidden @ self.expert_gate[expert_id].T)
        up = hidden @ self.expert_up[expert_id].T
        down_input = gate * up
        return down_input @ self.expert_down[expert_id].T

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        indices, weights = self.route(hidden)
        output = torch.zeros_like(hidden)
        for slot in range(self.top_k):
            expert_ids = indices[:, slot]
            router_weights = weights[:, slot]
            for expert_id in range(self.num_experts):
                mask = expert_ids == expert_id
                if mask.any():
                    output[mask] += router_weights[mask, None] * self.expert_forward(hidden[mask], expert_id)
        return output


@dataclass
class QuantizedLinearSet:
    decomposition: SharedDecomposition
    q_residuals: list[QuantizedResidual]

    def forward(self, hidden: torch.Tensor, expert_id: int) -> torch.Tensor:
        a_factor = self.decomposition.a_factors[expert_id]
        shared = (hidden @ self.decomposition.b_shared.T) @ a_factor.T
        residual = hidden @ self.q_residuals[expert_id].dequantize().T
        return shared + residual


@dataclass
class QuantizedToyMoeLayer:
    gate: torch.Tensor
    q_gate: QuantizedLinearSet
    q_up: QuantizedLinearSet
    q_down: QuantizedLinearSet
    top_k: int

    @property
    def num_experts(self) -> int:
        return len(self.q_gate.q_residuals)

    def route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = hidden @ self.gate.T
        probs = torch.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, k=self.top_k, dim=-1)
        return indices, weights

    def expert_forward(self, hidden: torch.Tensor, expert_id: int) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.q_gate.forward(hidden, expert_id))
        up = self.q_up.forward(hidden, expert_id)
        down_input = gate * up
        return self.q_down.forward(down_input, expert_id)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        indices, weights = self.route(hidden)
        output = torch.zeros_like(hidden)
        for slot in range(self.top_k):
            expert_ids = indices[:, slot]
            router_weights = weights[:, slot]
            for expert_id in range(self.num_experts):
                mask = expert_ids == expert_id
                if mask.any():
                    output[mask] += router_weights[mask, None] * self.expert_forward(hidden[mask], expert_id)
        return output


def collect_toy_calibration_stats(
    layer: ToyMoeLayer,
    hidden: torch.Tensor,
    *,
    block_size: int,
    model_name: str,
    layer_id: int,
) -> dict[str, LinearCalibrationStats]:
    indices, weights = layer.route(hidden)
    gate_acts: list[list[torch.Tensor]] = [[] for _ in range(layer.num_experts)]
    gate_router: list[list[torch.Tensor]] = [[] for _ in range(layer.num_experts)]
    up_acts: list[list[torch.Tensor]] = [[] for _ in range(layer.num_experts)]
    up_router: list[list[torch.Tensor]] = [[] for _ in range(layer.num_experts)]
    down_acts: list[list[torch.Tensor]] = [[] for _ in range(layer.num_experts)]
    down_router: list[list[torch.Tensor]] = [[] for _ in range(layer.num_experts)]

    for token_id in range(hidden.shape[0]):
        token = hidden[token_id]
        for slot in range(layer.top_k):
            expert_id = int(indices[token_id, slot].item())
            router_weight = weights[token_id, slot]
            gate_acts[expert_id].append(token)
            gate_router[expert_id].append(router_weight)
            up_acts[expert_id].append(token)
            up_router[expert_id].append(router_weight)

            gate = torch.nn.functional.silu(token @ layer.expert_gate[expert_id].T)
            up = token @ layer.expert_up[expert_id].T
            down_acts[expert_id].append(gate * up)
            down_router[expert_id].append(router_weight)

    return {
        "gate": accumulate_covariance_and_moments(
            gate_acts,
            gate_router,
            num_experts=layer.num_experts,
            input_dim=layer.hidden_size,
            block_size=block_size,
            model_name=model_name,
            layer_id=layer_id,
            linear_type="gate",
        ),
        "up": accumulate_covariance_and_moments(
            up_acts,
            up_router,
            num_experts=layer.num_experts,
            input_dim=layer.hidden_size,
            block_size=block_size,
            model_name=model_name,
            layer_id=layer_id,
            linear_type="up",
        ),
        "down": accumulate_covariance_and_moments(
            down_acts,
            down_router,
            num_experts=layer.num_experts,
            input_dim=layer.intermediate_size,
            block_size=block_size,
            model_name=model_name,
            layer_id=layer_id,
            linear_type="down",
        ),
    }


def _quantize_linear_type(
    weights: torch.Tensor,
    stats: LinearCalibrationStats,
    config: RescueConfig,
    *,
    model_name: str,
    layer_id: int,
    linear_type: str,
    rank: int | None,
) -> QuantizedLinearSet:
    expert_weights = [weights[expert_id] for expert_id in range(weights.shape[0])]
    decomposition = decompose_shared_subspace(expert_weights, stats.covariance, stats.expert_importance, rank=rank)
    q_residuals = quantize_residuals(
        decomposition.residuals,
        stats.rotated_second_moments,
        config,
        model_name=model_name,
        layer_id=layer_id,
        linear_type=linear_type,
    )
    return QuantizedLinearSet(decomposition=decomposition, q_residuals=q_residuals)


def quantize_toy_moe_layer(
    layer: ToyMoeLayer,
    calibration_hidden: torch.Tensor,
    config: RescueConfig,
    *,
    model_name: str = "toy",
    layer_id: int = 0,
    rank: int | None = None,
) -> QuantizedToyMoeLayer:
    stats = collect_toy_calibration_stats(
        layer,
        calibration_hidden,
        block_size=config.block_size,
        model_name=model_name,
        layer_id=layer_id,
    )
    return QuantizedToyMoeLayer(
        gate=layer.gate.clone(),
        q_gate=_quantize_linear_type(layer.expert_gate, stats["gate"], config, model_name=model_name, layer_id=layer_id, linear_type="gate", rank=rank),
        q_up=_quantize_linear_type(layer.expert_up, stats["up"], config, model_name=model_name, layer_id=layer_id, linear_type="up", rank=rank),
        q_down=_quantize_linear_type(layer.expert_down, stats["down"], config, model_name=model_name, layer_id=layer_id, linear_type="down", rank=rank),
        top_k=layer.top_k,
    )

