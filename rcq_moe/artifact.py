from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .official_qwen import (
    OfficialQwen35LayerDiagnostics,
    OfficialQwen35MoeRCQExperts,
    OfficialQwen35MoeRCQSparseMoeBlock,
    OfficialQwen35RCQLinearSet,
    _make_linear_set_from_state,
)


def _linear_set_to_state(linear: OfficialQwen35RCQLinearSet, prefix: str) -> dict[str, torch.Tensor | int | float | str]:
    state: dict[str, torch.Tensor | int | float | str] = {
        f"{prefix}.num_experts": len(linear.q_residuals),
        f"{prefix}.shared_mode": linear.shared_mode,
        f"{prefix}.b_shared": linear.decomposition.b_shared.detach().cpu(),
        f"{prefix}.eigvals": linear.decomposition.eigvals.detach().cpu(),
        f"{prefix}.eigvecs": linear.decomposition.eigvecs.detach().cpu(),
        f"{prefix}.v_r": linear.decomposition.v_r.detach().cpu(),
        f"{prefix}.captured_energy": float(linear.decomposition.captured_energy),
    }
    for expert_id, a_factor in enumerate(linear.decomposition.a_factors):
        state[f"{prefix}.a{expert_id}"] = a_factor.detach().cpu()
    for expert_id, q in enumerate(linear.q_residuals):
        q_prefix = f"{prefix}.q{expert_id}"
        state[f"{q_prefix}.values"] = q.values.detach().cpu()
        state[f"{q_prefix}.widths"] = q.widths.detach().cpu()
        state[f"{q_prefix}.scales"] = q.scales.detach().cpu()
        state[f"{q_prefix}.scores"] = q.scores.detach().cpu()
        state[f"{q_prefix}.valid_cols"] = int(q.valid_cols)
        state[f"{q_prefix}.block_size"] = int(q.block_size)
        state[f"{q_prefix}.model_name"] = q.model_name
        state[f"{q_prefix}.layer_id"] = int(q.layer_id)
        state[f"{q_prefix}.linear_type"] = q.linear_type
        state[f"{q_prefix}.expert_id"] = int(q.expert_id)
    return state


def _rcq_state_from_model(model: nn.Module) -> dict[str, torch.Tensor | int | float | str]:
    state: dict[str, torch.Tensor | int | float | str] = {}
    for layer_id, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if not isinstance(mlp, OfficialQwen35MoeRCQSparseMoeBlock):
            raise TypeError(f"layer {layer_id} mlp is not an OfficialQwen35MoeRCQSparseMoeBlock.")
        base = f"layers.{layer_id}.mlp"
        state.update(_linear_set_to_state(mlp.experts.q_gate, f"{base}.gate"))
        state.update(_linear_set_to_state(mlp.experts.q_up, f"{base}.up"))
        state.update(_linear_set_to_state(mlp.experts.q_down, f"{base}.down"))
        if mlp.correction_alpha is not None:
            state[f"{base}.correction_alpha"] = mlp.correction_alpha.detach().cpu()
        if mlp.correction_beta is not None:
            state[f"{base}.correction_beta"] = mlp.correction_beta.detach().cpu()
    return state


def _non_expert_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    return {
        key: value.detach().cpu()
        for key, value in state.items()
        if ".mlp.experts." not in key
    }


def _metadata(model: nn.Module, diagnostics: list[OfficialQwen35LayerDiagnostics] | None) -> dict[str, Any]:
    return {
        "artifact_format": "rcq_official_qwen35_moe_v1",
        "model_type": "Qwen3_5MoeForCausalLM",
        "config": model.config.to_dict(),
        "num_layers": len(model.model.layers),
        "diagnostics": [asdict(diag) for diag in diagnostics] if diagnostics is not None else None,
    }


def save_official_qwen_rcq_artifact(
    model: nn.Module,
    output_dir: str | Path,
    *,
    diagnostics: list[OfficialQwen35LayerDiagnostics] | None = None,
) -> None:
    """Save an official Qwen3.5-MoE RCQ adapter artifact.

    The artifact intentionally stores fake-dequant reference tensors for now.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metadata.json").write_text(json.dumps(_metadata(model, diagnostics), indent=2, sort_keys=True), encoding="utf-8")
    torch.save(_non_expert_state_dict(model), output / "non_expert_state.pt")
    torch.save(_rcq_state_from_model(model), output / "rcq_state.pt")


def _load_torch_dict(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_official_qwen_rcq_artifact(input_dir: str | Path) -> nn.Module:
    """Load an official Qwen3.5-MoE RCQ adapter artifact without FP expert weights."""
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    artifact = Path(input_dir)
    metadata = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("artifact_format") != "rcq_official_qwen35_moe_v1":
        raise ValueError(f"unsupported artifact format: {metadata.get('artifact_format')!r}")

    config = Qwen3_5MoeTextConfig(**metadata["config"])
    model = Qwen3_5MoeForCausalLM(config)
    non_expert_state = _load_torch_dict(artifact / "non_expert_state.pt")
    model.load_state_dict(non_expert_state, strict=False)
    rcq_state = _load_torch_dict(artifact / "rcq_state.pt")

    for layer_id, layer in enumerate(model.model.layers):
        fp_block = layer.mlp
        base = f"layers.{layer_id}.mlp"
        q_experts = OfficialQwen35MoeRCQExperts(
            num_experts=fp_block.experts.num_experts,
            hidden_dim=fp_block.experts.hidden_dim,
            intermediate_dim=fp_block.experts.intermediate_dim,
            act_fn=fp_block.experts.act_fn,
            q_gate=_make_linear_set_from_state(rcq_state, f"{base}.gate"),
            q_up=_make_linear_set_from_state(rcq_state, f"{base}.up"),
            q_down=_make_linear_set_from_state(rcq_state, f"{base}.down"),
        )
        q_block = OfficialQwen35MoeRCQSparseMoeBlock(
            gate=fp_block.gate,
            experts=q_experts,
            shared_expert=fp_block.shared_expert,
            shared_expert_gate=fp_block.shared_expert_gate,
        )
        alpha_key = f"{base}.correction_alpha"
        beta_key = f"{base}.correction_beta"
        if alpha_key in rcq_state:
            q_block.correction_alpha = rcq_state[alpha_key]
        if beta_key in rcq_state:
            q_block.correction_beta = rcq_state[beta_key]
        layer.mlp = q_block
    model.eval()
    return model
