#!/usr/bin/env python3
"""Layer-local pretrained Qwen3.6 RCQ ablation runner.

This script is intended for constrained Kaggle T4 x2 experiments. It does not
load the full model. It loads the tokenizer, embedding matrix, one selected MoE
block, and the selected layer's post-attention RMSNorm weight. Activations are
therefore explicitly reported as embedding + post-attention-RMSNorm activations,
not true full-prefix decoder activations.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcq_moe.correction import OnlineChannelRegression
from rcq_moe.decomposition import SharedDecomposition, decompose_shared_subspace
from rcq_moe.hadamard import pad_last_dim, rotate_activation_block, signed_hadamard_q
from rcq_moe.quantization import RescueConfig, lloyd_quantize_block, select_rescue_widths
from rcq_moe.stats import LinearCalibrationStats
from rcq_moe.storage import expert_bpw


DEFAULT_MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_DATASET = "HuggingFaceFW/fineweb-edu"


@dataclass
class LoadedLayer:
    embed_tokens: torch.Tensor
    input_norm_weight: torch.Tensor
    post_attention_norm_weight: torch.Tensor
    q_proj_weight: torch.Tensor
    q_proj_bias: torch.Tensor | None
    k_proj_weight: torch.Tensor
    k_proj_bias: torch.Tensor | None
    v_proj_weight: torch.Tensor
    v_proj_bias: torch.Tensor | None
    o_proj_weight: torch.Tensor
    o_proj_bias: torch.Tensor | None
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor
    router_weight: torch.Tensor
    gate_weight: torch.Tensor
    up_weight: torch.Tensor
    down_weight: torch.Tensor
    shared_gate_weight: torch.Tensor
    shared_up_weight: torch.Tensor
    shared_down_weight: torch.Tensor
    shared_expert_gate_weight: torch.Tensor
    hidden_act: str
    rms_norm_eps: float
    layer_type: str
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    attention_bias: bool
    rope_parameters: dict[str, Any]
    top_k: int


@dataclass
class StreamingStats:
    covariance_sum: torch.Tensor
    moment_sum: torch.Tensor
    expert_usage: torch.Tensor
    total_weight: torch.Tensor
    block_size: int
    model_name: str
    layer_id: int
    linear_type: str

    @classmethod
    def create(
        cls,
        *,
        num_experts: int,
        input_dim: int,
        block_size: int,
        model_name: str,
        layer_id: int,
        linear_type: str,
        device: torch.device,
    ) -> "StreamingStats":
        padded_dim = ((input_dim + block_size - 1) // block_size) * block_size
        return cls(
            covariance_sum=torch.zeros((input_dim, input_dim), dtype=torch.float32, device=device),
            moment_sum=torch.zeros((padded_dim // block_size, block_size), dtype=torch.float32, device=device),
            expert_usage=torch.zeros(num_experts, dtype=torch.float32, device=device),
            total_weight=torch.zeros((), dtype=torch.float32, device=device),
            block_size=block_size,
            model_name=model_name,
            layer_id=layer_id,
            linear_type=linear_type,
        )

    def update(self, activations: torch.Tensor, router_weights: torch.Tensor, expert_id: int) -> None:
        if activations.numel() == 0:
            return
        acts = activations.float()
        weights = router_weights.float()
        weighted = acts * weights[:, None]
        self.covariance_sum += acts.T @ weighted
        self.expert_usage[expert_id] += weights.sum()
        self.total_weight += weights.sum()

        padded, _ = pad_last_dim(acts, self.block_size)
        blocks = padded.reshape(acts.shape[0], -1, self.block_size)
        for block_index in range(blocks.shape[1]):
            q = signed_hadamard_q(
                self.model_name,
                self.layer_id,
                self.linear_type,
                block_index,
                self.block_size,
                device=acts.device,
                dtype=acts.dtype,
            )
            rotated = rotate_activation_block(blocks[:, block_index, :], q)
            self.moment_sum[block_index] += (weights[:, None] * rotated.square()).sum(dim=0)

    def finish(self) -> LinearCalibrationStats:
        denom = self.total_weight.clamp_min(1e-12)
        covariance = self.covariance_sum / denom
        moments = self.moment_sum / denom
        importance = self.expert_usage / self.expert_usage.sum().clamp_min(1e-12)
        return LinearCalibrationStats(covariance=covariance, expert_importance=importance, rotated_second_moments=moments)


@dataclass
class QuantizedLinear:
    decomposition: SharedDecomposition
    residuals: list[torch.Tensor]
    widths: torch.Tensor
    bpw: float

    def weight_for_expert(self, expert_id: int) -> torch.Tensor:
        shared = self.decomposition.a_factors[expert_id] @ self.decomposition.b_shared
        return shared + self.residuals[expert_id]


@dataclass
class QuantizedMoe:
    gate: QuantizedLinear
    up: QuantizedLinear
    down: QuantizedLinear
    correction_alpha: torch.Tensor | None = None
    correction_beta: torch.Tensor | None = None


class MSEAccumulator:
    def __init__(self) -> None:
        self.sse = 0.0
        self.count = 0
        self.max_abs = 0.0

    def update(self, reference: torch.Tensor, candidate: torch.Tensor) -> None:
        delta = (reference.float() - candidate.float()).detach()
        self.sse += float(delta.square().sum().item())
        self.count += delta.numel()
        self.max_abs = max(self.max_abs, float(delta.abs().max().item()))

    def summary(self) -> dict[str, float]:
        return {
            "mse": self.sse / max(1, self.count),
            "rmse": math.sqrt(self.sse / max(1, self.count)),
            "max_abs": self.max_abs,
        }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _text_config(config: Any) -> Any:
    if isinstance(config, dict):
        return config.get("text_config", config)
    return getattr(config, "text_config", config)


def _getattr_any(obj: Any, names: list[str], default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _rank_for_divisor(cols: int, divisor: int) -> int:
    return max(1, math.ceil(cols / divisor))


def _act(x: torch.Tensor, name: str) -> torch.Tensor:
    if name in {"silu", "swish"}:
        return F.silu(x)
    if name == "gelu":
        return F.gelu(x)
    if name == "relu":
        return F.relu(x)
    raise ValueError(f"unsupported hidden activation {name!r}")


def _rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = hidden.float() * torch.rsqrt(hidden.float().square().mean(dim=-1, keepdim=True) + eps)
    out = out * (1.0 + weight.float())
    return out.to(hidden.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first = x[..., : x.shape[-1] // 2]
    second = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def _rotary_cos_sin(layer: LoadedLayer, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    rope = layer.rope_parameters
    rope_type = rope.get("rope_type", "default")
    if rope_type != "default":
        raise ValueError(f"unsupported rope_type {rope_type!r}; this runner currently implements default text RoPE only")
    theta = float(rope.get("rope_theta", 1_000_000.0))
    partial = float(rope.get("partial_rotary_factor", 1.0))
    rotary_dim = int(layer.head_dim * partial)
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def _apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return torch.cat((q_embed, q_pass), dim=-1), torch.cat((k_embed, k_pass), dim=-1)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, heads * n_rep, seq_len, head_dim)


def _linear_optional_bias(hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    return F.linear(
        hidden,
        weight.to(device=hidden.device, dtype=hidden.dtype),
        None if bias is None else bias.to(device=hidden.device, dtype=hidden.dtype),
    )


def _layer0_attention(hidden: torch.Tensor, layer: LoadedLayer) -> torch.Tensor:
    if layer.layer_type != "full_attention":
        raise ValueError(f"unsupported layer_type {layer.layer_type!r}; proper calibration currently requires full_attention")
    batch, seq_len, _ = hidden.shape
    normed = _rms_norm(hidden, layer.input_norm_weight.to(hidden.device), layer.rms_norm_eps)
    q_proj = _linear_optional_bias(normed, layer.q_proj_weight, layer.q_proj_bias)
    q_proj = q_proj.view(batch, seq_len, layer.num_attention_heads, layer.head_dim * 2)
    query, gate = torch.chunk(q_proj, 2, dim=-1)
    gate = gate.reshape(batch, seq_len, layer.num_attention_heads * layer.head_dim)

    key = _linear_optional_bias(normed, layer.k_proj_weight, layer.k_proj_bias)
    value = _linear_optional_bias(normed, layer.v_proj_weight, layer.v_proj_bias)
    query = _rms_norm(query, layer.q_norm_weight.to(hidden.device), layer.rms_norm_eps).transpose(1, 2)
    key = _rms_norm(
        key.view(batch, seq_len, layer.num_key_value_heads, layer.head_dim),
        layer.k_norm_weight.to(hidden.device),
        layer.rms_norm_eps,
    ).transpose(1, 2)
    value = value.view(batch, seq_len, layer.num_key_value_heads, layer.head_dim).transpose(1, 2)

    cos, sin = _rotary_cos_sin(layer, seq_len, hidden.device, hidden.dtype)
    query, key = _apply_rotary(query, key, cos, sin)
    groups = layer.num_attention_heads // layer.num_key_value_heads
    key = _repeat_kv(key, groups)
    value = _repeat_kv(value, groups)

    scores = torch.matmul(query, key.transpose(2, 3)) * (layer.head_dim**-0.5)
    causal = torch.full((seq_len, seq_len), torch.finfo(scores.dtype).min, device=hidden.device, dtype=scores.dtype)
    causal = torch.triu(causal, diagonal=1)
    scores = scores + causal[None, None, :, :]
    probs = torch.softmax(scores.float(), dim=-1).to(hidden.dtype)
    attn = torch.matmul(probs, value).transpose(1, 2).contiguous().reshape(batch, seq_len, -1)
    attn = attn * torch.sigmoid(gate)
    return _linear_optional_bias(attn, layer.o_proj_weight, layer.o_proj_bias)


def _route(hidden: torch.Tensor, router_weight: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    logits = F.linear(hidden, router_weight)
    probs = torch.softmax(logits.float(), dim=-1)
    weights, indices = torch.topk(probs, top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return weights.to(hidden.dtype), indices


def _linear_weight(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(hidden, weight.to(device=hidden.device, dtype=hidden.dtype))


def _shared_expert(hidden: torch.Tensor, layer: LoadedLayer) -> torch.Tensor:
    gate = _linear_weight(hidden, layer.shared_gate_weight)
    up = _linear_weight(hidden, layer.shared_up_weight)
    down = _linear_weight(_act(gate, layer.hidden_act) * up, layer.shared_down_weight)
    scale = torch.sigmoid(_linear_weight(hidden, layer.shared_expert_gate_weight))
    return scale * down


def _expert_moe_output(
    hidden: torch.Tensor,
    router_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    *,
    layer: LoadedLayer,
    q_moe: QuantizedMoe | None = None,
) -> torch.Tensor:
    out = torch.zeros_like(hidden)
    num_experts = layer.router_weight.shape[0]
    for expert_id in range(num_experts):
        positions = selected_experts == expert_id
        if not positions.any():
            continue
        token_idx, slot_idx = torch.where(positions)
        current = hidden[token_idx]
        weights = router_weights[token_idx, slot_idx, None]
        if q_moe is None:
            gate_w = layer.gate_weight[expert_id]
            up_w = layer.up_weight[expert_id]
            down_w = layer.down_weight[expert_id]
        else:
            gate_w = q_moe.gate.weight_for_expert(expert_id).to(device=hidden.device, dtype=hidden.dtype)
            up_w = q_moe.up.weight_for_expert(expert_id).to(device=hidden.device, dtype=hidden.dtype)
            down_w = q_moe.down.weight_for_expert(expert_id).to(device=hidden.device, dtype=hidden.dtype)
        gate = _linear_weight(current, gate_w)
        up = _linear_weight(current, up_w)
        expert_out = _linear_weight(_act(gate, layer.hidden_act) * up, down_w)
        out.index_add_(0, token_idx, (expert_out * weights).to(out.dtype))
    return out


def moe_output(hidden: torch.Tensor, layer: LoadedLayer, q_moe: QuantizedMoe | None = None, *, include_shared: bool) -> torch.Tensor:
    router_weights, selected_experts = _route(hidden, layer.router_weight.to(device=hidden.device, dtype=hidden.dtype), layer.top_k)
    out = _expert_moe_output(hidden, router_weights, selected_experts, layer=layer, q_moe=q_moe)
    if include_shared:
        out = out + _shared_expert(hidden, layer)
    if q_moe is not None and q_moe.correction_alpha is not None and q_moe.correction_beta is not None:
        out = q_moe.correction_alpha.to(out.device, out.dtype) * out + q_moe.correction_beta.to(out.device, out.dtype)
    return out


def _hidden_from_ids(input_ids: torch.Tensor, layer: LoadedLayer, device: torch.device, *, activation_source: str) -> torch.Tensor:
    embedding = F.embedding(input_ids.to(device), layer.embed_tokens.to(device=device))
    if activation_source == "proxy_embedding_norm":
        return _rms_norm(embedding, layer.post_attention_norm_weight.to(device=device), layer.rms_norm_eps).reshape(-1, embedding.shape[-1])
    if activation_source == "true_layer0_post_attention_norm":
        if input_ids.shape[0] != 1:
            raise ValueError("true layer-0 attention path expects one document/batch row at a time")
        residual = embedding
        attn = _layer0_attention(embedding, layer)
        hidden = residual + attn
        return _rms_norm(hidden, layer.post_attention_norm_weight.to(device=device), layer.rms_norm_eps).reshape(-1, hidden.shape[-1])
    raise ValueError(f"unknown activation_source {activation_source!r}")


def _load_raw_docs(dataset: str, split: str, text_field: str, count: int) -> list[str]:
    from datasets import load_dataset

    docs: list[str] = []
    stream = load_dataset(dataset, split=split, streaming=True)
    for row in stream:
        value = row.get(text_field)
        if not isinstance(value, str):
            raise ValueError(f"text field {text_field!r} is not a string in streamed row {len(docs)}")
        docs.append(value)
        if len(docs) >= count:
            return docs
    raise ValueError(f"dataset stream ended after {len(docs)} docs, need {count}")


def _tokenize_doc(tokenizer: Any, text: str, max_tokens: int, device: torch.device) -> torch.Tensor:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    )
    ids = encoded["input_ids"]
    if ids.numel() == 0:
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is None:
            raise ValueError("empty tokenized document and tokenizer has no eos_token_id fallback")
        ids = torch.tensor([[int(eos)]], dtype=torch.long)
    return ids.to(device)


def _hf_download(model_id: str, filename: str, revision: str | None, token: str | None, cache_dir: str | None) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=model_id, filename=filename, revision=revision, token=token, cache_dir=cache_dir)


def _load_config_json(model_id: str, revision: str | None, token: str | None, cache_dir: str | None) -> dict[str, Any]:
    config_path = _hf_download(model_id, "config.json", revision, token, cache_dir)
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def _load_index(model_id: str, revision: str | None, token: str | None, cache_dir: str | None) -> dict[str, Any]:
    index_path = _hf_download(model_id, "model.safetensors.index.json", revision, token, cache_dir)
    return json.loads(Path(index_path).read_text(encoding="utf-8"))


def _tensor_shape_map(index: dict[str, Any]) -> dict[str, list[int]]:
    metadata = index.get("metadata", {})
    shapes = metadata.get("all_checkpoint_keys", None)
    if isinstance(shapes, dict):
        return {str(k): [int(x) for x in v] for k, v in shapes.items() if isinstance(v, list)}
    return {}


def _download_shards_for_keys(
    model_id: str,
    index: dict[str, Any],
    keys: list[str],
    *,
    revision: str | None,
    token: str | None,
    cache_dir: str | None,
) -> dict[str, Path]:
    weight_map = index["weight_map"]
    shard_names = sorted({weight_map[key] for key in keys})
    return {
        shard: Path(_hf_download(model_id, shard, revision, token, cache_dir))
        for shard in shard_names
    }


def _read_tensors(shards: dict[str, Path], index: dict[str, Any], keys: list[str], *, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(index["weight_map"][key], []).append(key)

    tensors: dict[str, torch.Tensor] = {}
    for shard, shard_keys in by_shard.items():
        with safe_open(shards[shard], framework="pt", device="cpu") as handle:
            for key in shard_keys:
                tensors[key] = handle.get_tensor(key).to(dtype=dtype)
    return tensors


def _required_keys(layer_id: int) -> dict[str, str]:
    prefix = f"model.layers.{layer_id}"
    return {
        "embed_tokens": "model.embed_tokens.weight",
        "input_norm_weight": f"{prefix}.input_layernorm.weight",
        "post_attention_norm_weight": f"{prefix}.post_attention_layernorm.weight",
        "q_proj_weight": f"{prefix}.self_attn.q_proj.weight",
        "q_proj_bias": f"{prefix}.self_attn.q_proj.bias",
        "k_proj_weight": f"{prefix}.self_attn.k_proj.weight",
        "k_proj_bias": f"{prefix}.self_attn.k_proj.bias",
        "v_proj_weight": f"{prefix}.self_attn.v_proj.weight",
        "v_proj_bias": f"{prefix}.self_attn.v_proj.bias",
        "o_proj_weight": f"{prefix}.self_attn.o_proj.weight",
        "o_proj_bias": f"{prefix}.self_attn.o_proj.bias",
        "q_norm_weight": f"{prefix}.self_attn.q_norm.weight",
        "k_norm_weight": f"{prefix}.self_attn.k_norm.weight",
        "router_weight": f"{prefix}.mlp.gate.weight",
        "gate_up_weight": f"{prefix}.mlp.experts.gate_up_proj",
        "down_weight": f"{prefix}.mlp.experts.down_proj",
        "shared_gate_weight": f"{prefix}.mlp.shared_expert.gate_proj.weight",
        "shared_up_weight": f"{prefix}.mlp.shared_expert.up_proj.weight",
        "shared_down_weight": f"{prefix}.mlp.shared_expert.down_proj.weight",
        "shared_expert_gate_weight": f"{prefix}.mlp.shared_expert_gate.weight",
    }


def _estimate_required_bytes(index: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    shapes = _tensor_shape_map(index)
    weight_map = index["weight_map"]
    present_shapes = {key: shapes.get(key) for key in keys}
    shards = sorted({weight_map[key] for key in keys})
    return {
        "required_keys": keys,
        "required_key_shapes_if_index_reports_them": present_shapes,
        "required_shards": shards,
        "required_shard_count": len(shards),
    }


def load_layer(
    *,
    model_id: str,
    revision: str | None,
    token: str | None,
    cache_dir: str | None,
    layer_id: int,
    dtype: torch.dtype,
) -> tuple[LoadedLayer, dict[str, Any]]:
    config = _load_config_json(model_id, revision, token, cache_dir)
    text_config = _text_config(config)
    index = _load_index(model_id, revision, token, cache_dir)
    key_map = _required_keys(layer_id)
    optional_keys = {key_map["q_proj_bias"], key_map["k_proj_bias"], key_map["v_proj_bias"], key_map["o_proj_bias"]}
    missing = [key for key in key_map.values() if key not in optional_keys and key not in index["weight_map"]]
    if missing:
        raise KeyError(f"required tensors missing from safetensors index: {missing}")
    existing_keys = [key for key in key_map.values() if key in index["weight_map"]]
    shards = _download_shards_for_keys(model_id, index, existing_keys, revision=revision, token=token, cache_dir=cache_dir)
    tensors = _read_tensors(shards, index, existing_keys, dtype=dtype)
    gate_up = tensors[key_map["gate_up_weight"]]
    intermediate = gate_up.shape[1] // 2
    layer_types = list(_getattr_any(text_config, ["layer_types"], []))
    layer_type = str(layer_types[layer_id]) if layer_id < len(layer_types) else "full_attention"
    head_dim = int(_getattr_any(text_config, ["head_dim"], _getattr_any(text_config, ["hidden_size"]) // _getattr_any(text_config, ["num_attention_heads"])))
    num_attention_heads = int(_getattr_any(text_config, ["num_attention_heads"]))
    num_key_value_heads = int(_getattr_any(text_config, ["num_key_value_heads"], num_attention_heads))
    layer = LoadedLayer(
        embed_tokens=tensors[key_map["embed_tokens"]],
        input_norm_weight=tensors[key_map["input_norm_weight"]],
        post_attention_norm_weight=tensors[key_map["post_attention_norm_weight"]],
        q_proj_weight=tensors[key_map["q_proj_weight"]],
        q_proj_bias=tensors.get(key_map["q_proj_bias"]),
        k_proj_weight=tensors[key_map["k_proj_weight"]],
        k_proj_bias=tensors.get(key_map["k_proj_bias"]),
        v_proj_weight=tensors[key_map["v_proj_weight"]],
        v_proj_bias=tensors.get(key_map["v_proj_bias"]),
        o_proj_weight=tensors[key_map["o_proj_weight"]],
        o_proj_bias=tensors.get(key_map["o_proj_bias"]),
        q_norm_weight=tensors[key_map["q_norm_weight"]],
        k_norm_weight=tensors[key_map["k_norm_weight"]],
        router_weight=tensors[key_map["router_weight"]],
        gate_weight=gate_up[:, :intermediate, :].contiguous(),
        up_weight=gate_up[:, intermediate:, :].contiguous(),
        down_weight=tensors[key_map["down_weight"]],
        shared_gate_weight=tensors[key_map["shared_gate_weight"]],
        shared_up_weight=tensors[key_map["shared_up_weight"]],
        shared_down_weight=tensors[key_map["shared_down_weight"]],
        shared_expert_gate_weight=tensors[key_map["shared_expert_gate_weight"]],
        hidden_act=str(_getattr_any(text_config, ["hidden_act"], "silu")),
        rms_norm_eps=float(_getattr_any(text_config, ["rms_norm_eps"], 1e-6)),
        layer_type=layer_type,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        attention_bias=bool(_getattr_any(text_config, ["attention_bias"], False)),
        rope_parameters=dict(_getattr_any(text_config, ["rope_parameters", "rope_scaling"], {})),
        top_k=int(_getattr_any(text_config, ["num_experts_per_tok"], 8)),
    )
    summary = {
        "model_type": _getattr_any(text_config, ["model_type"]),
        "hidden_size": int(_getattr_any(text_config, ["hidden_size"], layer.router_weight.shape[1])),
        "moe_intermediate_size": int(_getattr_any(text_config, ["moe_intermediate_size"], intermediate)),
        "num_experts": int(_getattr_any(text_config, ["num_experts"], layer.router_weight.shape[0])),
        "num_experts_per_tok": layer.top_k,
        "hidden_act": layer.hidden_act,
        "rms_norm_eps": layer.rms_norm_eps,
        "layer_type": layer.layer_type,
        "num_attention_heads": layer.num_attention_heads,
        "num_key_value_heads": layer.num_key_value_heads,
        "head_dim": layer.head_dim,
        "attention_bias": layer.attention_bias,
        "rope_parameters": layer.rope_parameters,
        "required_tensor_summary": _estimate_required_bytes(index, existing_keys),
    }
    return layer, summary


def inspect_only(
    *,
    model_id: str,
    revision: str | None,
    token: str | None,
    cache_dir: str | None,
    layer_id: int,
) -> dict[str, Any]:
    config = _load_config_json(model_id, revision, token, cache_dir)
    text_config = _text_config(config)
    index = _load_index(model_id, revision, token, cache_dir)
    key_map = _required_keys(layer_id)
    optional_keys = {key_map["q_proj_bias"], key_map["k_proj_bias"], key_map["v_proj_bias"], key_map["o_proj_bias"]}
    missing = [key for key in key_map.values() if key not in optional_keys and key not in index["weight_map"]]
    layer_types = list(_getattr_any(text_config, ["layer_types"], []))
    layer_type = str(layer_types[layer_id]) if layer_id < len(layer_types) else "full_attention"
    return {
        "model_id": model_id,
        "revision": revision,
        "layer_id": layer_id,
        "model_type": _getattr_any(text_config, ["model_type"]),
        "hidden_size": _getattr_any(text_config, ["hidden_size"]),
        "moe_intermediate_size": _getattr_any(text_config, ["moe_intermediate_size"]),
        "num_hidden_layers": _getattr_any(text_config, ["num_hidden_layers"]),
        "num_experts": _getattr_any(text_config, ["num_experts"]),
        "num_experts_per_tok": _getattr_any(text_config, ["num_experts_per_tok"]),
        "layer_type": layer_type,
        "num_attention_heads": _getattr_any(text_config, ["num_attention_heads"]),
        "num_key_value_heads": _getattr_any(text_config, ["num_key_value_heads"]),
        "head_dim": _getattr_any(text_config, ["head_dim"]),
        "attention_bias": _getattr_any(text_config, ["attention_bias"]),
        "rope_parameters": _getattr_any(text_config, ["rope_parameters", "rope_scaling"], {}),
        "missing_required_tensors": missing,
        "required_tensor_summary": _estimate_required_bytes(index, [key for key in key_map.values() if key in index["weight_map"]]),
    }


def collect_stats(
    docs: list[str],
    tokenizer: Any,
    layer: LoadedLayer,
    *,
    max_tokens_per_doc: int,
    block_size: int,
    device: torch.device,
    weighted: bool,
    activation_source: str,
) -> dict[str, LinearCalibrationStats]:
    model_name = "qwen36-single-layer"
    stats = {
        "gate": StreamingStats.create(
            num_experts=layer.router_weight.shape[0],
            input_dim=layer.router_weight.shape[1],
            block_size=block_size,
            model_name=model_name,
            layer_id=0,
            linear_type="gate",
            device=device,
        ),
        "up": StreamingStats.create(
            num_experts=layer.router_weight.shape[0],
            input_dim=layer.router_weight.shape[1],
            block_size=block_size,
            model_name=model_name,
            layer_id=0,
            linear_type="up",
            device=device,
        ),
        "down": StreamingStats.create(
            num_experts=layer.router_weight.shape[0],
            input_dim=layer.down_weight.shape[2],
            block_size=block_size,
            model_name=model_name,
            layer_id=0,
            linear_type="down",
            device=device,
        ),
    }

    for doc in docs:
        input_ids = _tokenize_doc(tokenizer, doc, max_tokens_per_doc, device)
        hidden = _hidden_from_ids(input_ids, layer, device, activation_source=activation_source)
        router_weights, selected = _route(hidden, layer.router_weight.to(device=device, dtype=hidden.dtype), layer.top_k)
        for expert_id in range(layer.router_weight.shape[0]):
            positions = selected == expert_id
            if not positions.any():
                continue
            token_idx, slot_idx = torch.where(positions)
            current = hidden[token_idx]
            route = router_weights[token_idx, slot_idx]
            stat_weight = route.square() if weighted else torch.ones_like(route)
            stats["gate"].update(current, stat_weight, expert_id)
            stats["up"].update(current, stat_weight, expert_id)
            gate = _linear_weight(current, layer.gate_weight[expert_id].to(device=device, dtype=hidden.dtype))
            up = _linear_weight(current, layer.up_weight[expert_id].to(device=device, dtype=hidden.dtype))
            down_input = _act(gate, layer.hidden_act) * up
            stats["down"].update(down_input, stat_weight, expert_id)
        del input_ids, hidden, router_weights, selected
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {name: item.finish() for name, item in stats.items()}


def _binary_quantize(z: torch.Tensor, moments: torch.Tensor, *, weighted_scale: bool) -> tuple[torch.Tensor, torch.Tensor]:
    signs = torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
    if weighted_scale:
        denom = moments.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = (moments * z.abs()).sum(dim=-1, keepdim=True) / denom
    else:
        scale = z.abs().mean(dim=-1, keepdim=True)
    values = scale * signs
    score = (moments * (z - values).square()).sum(dim=-1)
    return values, score


def _quantize_residual_variant(
    residuals: list[torch.Tensor],
    moments: torch.Tensor,
    *,
    use_hadamard: bool,
    weighted_scale: bool,
    rescue_config: RescueConfig | None,
    model_name: str,
    layer_id: int,
    linear_type: str,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    if not residuals:
        raise ValueError("residuals must not be empty")
    block_size = moments.shape[-1]
    values_by_expert: list[torch.Tensor] = []
    scores_by_expert: list[torch.Tensor] = []
    valid_cols: list[int] = []

    for residual in residuals:
        padded, valid = pad_last_dim(residual.float(), block_size)
        blocks = padded.reshape(residual.shape[0], -1, block_size)
        if use_hadamard:
            transformed = torch.empty_like(blocks)
            for block_index in range(blocks.shape[1]):
                q = signed_hadamard_q(model_name, layer_id, linear_type, block_index, block_size, device=blocks.device, dtype=blocks.dtype)
                transformed[:, block_index, :] = blocks[:, block_index, :] @ q
        else:
            transformed = blocks
        expanded_moments = moments.to(device=transformed.device, dtype=transformed.dtype)[None, :, :].expand_as(transformed)
        values, scores = _binary_quantize(transformed, expanded_moments, weighted_scale=weighted_scale)
        values_by_expert.append(values)
        scores_by_expert.append(scores)
        valid_cols.append(valid)

    all_scores = torch.stack(scores_by_expert, dim=0)
    widths = torch.ones_like(all_scores, dtype=torch.int64)
    if rescue_config is not None:
        widths = select_rescue_widths(all_scores, rescue_config)
        for expert_id, residual in enumerate(residuals):
            padded, _ = pad_last_dim(residual.float(), block_size)
            fp_blocks = padded.reshape(residual.shape[0], -1, block_size)
            if use_hadamard:
                transformed_fp = torch.empty_like(fp_blocks)
                for block_index in range(fp_blocks.shape[1]):
                    q = signed_hadamard_q(model_name, layer_id, linear_type, block_index, block_size, device=fp_blocks.device, dtype=fp_blocks.dtype)
                    transformed_fp[:, block_index, :] = fp_blocks[:, block_index, :] @ q
            else:
                transformed_fp = fp_blocks
            expanded_moments = moments.to(device=transformed_fp.device, dtype=transformed_fp.dtype)[None, :, :].expand_as(transformed_fp)
            for bits in (2, 4):
                mask = widths[expert_id] == bits
                if mask.any():
                    rescue_values, _ = lloyd_quantize_block(transformed_fp[mask], expanded_moments[mask], bits=bits)
                    values_by_expert[expert_id][mask] = rescue_values

    dequant_residuals = []
    for expert_id, values in enumerate(values_by_expert):
        if use_hadamard:
            blocks = []
            for block_index in range(values.shape[1]):
                q = signed_hadamard_q(model_name, layer_id, linear_type, block_index, block_size, device=values.device, dtype=values.dtype)
                blocks.append(values[:, block_index, :] @ q.T)
            padded = torch.cat(blocks, dim=1)
        else:
            padded = values.reshape(values.shape[0], values.shape[1] * values.shape[2])
        dequant_residuals.append(padded[:, : valid_cols[expert_id]].contiguous())
    return dequant_residuals, widths


def quantize_linear(
    weights: torch.Tensor,
    stats: LinearCalibrationStats,
    *,
    rank: int,
    use_hadamard: bool,
    weighted_scale: bool,
    rescue_config: RescueConfig | None,
    model_name: str,
    layer_id: int,
    linear_type: str,
) -> QuantizedLinear:
    expert_weights = [weights[expert_id].float() for expert_id in range(weights.shape[0])]
    decomposition = decompose_shared_subspace(expert_weights, stats.covariance.float(), stats.expert_importance.float(), rank=rank)
    residuals, widths = _quantize_residual_variant(
        decomposition.residuals,
        stats.rotated_second_moments.float(),
        use_hadamard=use_hadamard,
        weighted_scale=weighted_scale,
        rescue_config=rescue_config,
        model_name=model_name,
        layer_id=layer_id,
        linear_type=linear_type,
    )
    report = expert_bpw(
        num_experts=weights.shape[0],
        rows=weights.shape[1],
        cols=weights.shape[2],
        rank=rank,
        widths=widths,
        block_size=stats.rotated_second_moments.shape[-1],
    )
    return QuantizedLinear(decomposition=decomposition, residuals=residuals, widths=widths, bpw=report.bpw)


def build_q_moe(
    layer: LoadedLayer,
    stats: dict[str, LinearCalibrationStats],
    *,
    rank_divisor: int,
    use_hadamard: bool,
    weighted_scale: bool,
    rescue_config: RescueConfig | None,
) -> QuantizedMoe:
    rank_hidden = _rank_for_divisor(layer.router_weight.shape[1], rank_divisor)
    rank_intermediate = _rank_for_divisor(layer.down_weight.shape[2], rank_divisor)
    kwargs = {
        "use_hadamard": use_hadamard,
        "weighted_scale": weighted_scale,
        "rescue_config": rescue_config,
        "model_name": "qwen36-single-layer",
        "layer_id": 0,
    }
    return QuantizedMoe(
        gate=quantize_linear(layer.gate_weight, stats["gate"], rank=rank_hidden, linear_type="gate", **kwargs),
        up=quantize_linear(layer.up_weight, stats["up"], rank=rank_hidden, linear_type="up", **kwargs),
        down=quantize_linear(layer.down_weight, stats["down"], rank=rank_intermediate, linear_type="down", **kwargs),
    )


def evaluate_docs(
    docs: list[str],
    tokenizer: Any,
    layer: LoadedLayer,
    q_moe: QuantizedMoe | None,
    *,
    max_tokens_per_doc: int,
    device: torch.device,
    include_shared: bool,
    activation_source: str,
    correction_fit: OnlineChannelRegression | None = None,
) -> dict[str, float]:
    acc = MSEAccumulator()
    for doc in docs:
        input_ids = _tokenize_doc(tokenizer, doc, max_tokens_per_doc, device)
        hidden = _hidden_from_ids(input_ids, layer, device, activation_source=activation_source)
        with torch.inference_mode():
            fp = moe_output(hidden, layer, None, include_shared=include_shared)
            if q_moe is None:
                q = fp
            else:
                q = moe_output(hidden, layer, q_moe, include_shared=include_shared)
        acc.update(fp, q)
        if correction_fit is not None:
            correction_fit.update(fp, q)
        del input_ids, hidden, fp, q
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return acc.summary()


def _fit_routed_correction(
    calib_docs: list[str],
    tokenizer: Any,
    layer: LoadedLayer,
    q_moe: QuantizedMoe,
    *,
    max_tokens_per_doc: int,
    device: torch.device,
    include_shared: bool,
    activation_source: str,
) -> dict[str, float]:
    stats = OnlineChannelRegression(dim=layer.router_weight.shape[1], dtype=torch.float32, device=device)
    before = evaluate_docs(
        calib_docs,
        tokenizer,
        layer,
        q_moe,
        max_tokens_per_doc=max_tokens_per_doc,
        device=device,
        include_shared=include_shared,
        activation_source=activation_source,
        correction_fit=stats,
    )
    alpha, beta = stats.solve()
    q_moe.correction_alpha = alpha
    q_moe.correction_beta = beta
    after = evaluate_docs(
        calib_docs,
        tokenizer,
        layer,
        q_moe,
        max_tokens_per_doc=max_tokens_per_doc,
        device=device,
        include_shared=include_shared,
        activation_source=activation_source,
    )
    return {"calib_mse_before": before["mse"], "calib_mse_after": after["mse"]}


def _width_percentages(q: QuantizedLinear) -> dict[str, float]:
    total = q.widths.numel()
    return {str(bit): float((q.widths == bit).sum().item() / total) for bit in (1, 2, 4)}


def _diagnostics(q_moe: QuantizedMoe) -> dict[str, Any]:
    return {
        "gate": {
            "captured_energy": q_moe.gate.decomposition.captured_energy,
            "bpw": q_moe.gate.bpw,
            "width_percentages": _width_percentages(q_moe.gate),
        },
        "up": {
            "captured_energy": q_moe.up.decomposition.captured_energy,
            "bpw": q_moe.up.bpw,
            "width_percentages": _width_percentages(q_moe.up),
        },
        "down": {
            "captured_energy": q_moe.down.decomposition.captured_energy,
            "bpw": q_moe.down.bpw,
            "width_percentages": _width_percentages(q_moe.down),
        },
    }


def run_ablation(
    label: str,
    layer: LoadedLayer,
    tokenizer: Any,
    calib_docs: list[str],
    eval_docs: list[str],
    stats: dict[str, LinearCalibrationStats],
    *,
    rank_divisor: int,
    use_hadamard: bool,
    weighted_scale: bool,
    rescue_config: RescueConfig | None,
    fit_correction: bool,
    max_tokens_per_doc: int,
    device: torch.device,
    include_shared: bool,
    activation_source: str,
) -> dict[str, Any]:
    started = time.time()
    q_moe = build_q_moe(
        layer,
        stats,
        rank_divisor=rank_divisor,
        use_hadamard=use_hadamard,
        weighted_scale=weighted_scale,
        rescue_config=rescue_config,
    )
    correction = None
    if fit_correction:
        correction = _fit_routed_correction(
            calib_docs,
            tokenizer,
            layer,
            q_moe,
            max_tokens_per_doc=max_tokens_per_doc,
            device=device,
            include_shared=include_shared,
            activation_source=activation_source,
        )
    calib = evaluate_docs(
        calib_docs,
        tokenizer,
        layer,
        q_moe,
        max_tokens_per_doc=max_tokens_per_doc,
        device=device,
        include_shared=include_shared,
        activation_source=activation_source,
    )
    heldout = evaluate_docs(
        eval_docs,
        tokenizer,
        layer,
        q_moe,
        max_tokens_per_doc=max_tokens_per_doc,
        device=device,
        include_shared=include_shared,
        activation_source=activation_source,
    )
    payload = {
        "label": label,
        "status": "ok",
        "rank_divisor": rank_divisor,
        "use_hadamard": use_hadamard,
        "weighted_scale": weighted_scale,
        "rescue_config": rescue_config.name if rescue_config is not None else None,
        "fit_routed_moe_output_correction": fit_correction,
        "correction": correction,
        "calibration": calib,
        "heldout": heldout,
        "diagnostics": _diagnostics(q_moe),
        "elapsed_sec": time.time() - started,
    }
    del q_moe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-layer Qwen3.6 RCQ ablation runner.")
    parser.add_argument("--model-id", default=os.environ.get("QWEN36_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--revision", default=os.environ.get("QWEN36_MODEL_REVISION"))
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--calib-docs", type=int, default=256)
    parser.add_argument("--eval-docs", type=int, default=64)
    parser.add_argument("--max-tokens-per-doc", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen36_single_layer_rcq_ablation"))
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Inspect config/index without loading tensor shards or data.")
    parser.add_argument("--include-shared-expert", action="store_true")
    parser.add_argument(
        "--activation-source",
        choices=["true_layer0_post_attention_norm", "proxy_embedding_norm"],
        default="true_layer0_post_attention_norm",
        help="Default runs layer-0 attention before the MoE. Proxy mode is for debugging only.",
    )
    parser.add_argument("--max-experiments", type=int, help="Run only the first N ablation experiments.")
    args = parser.parse_args()

    del args.local_files_only  # Kept for CLI symmetry; hf_hub_download handles cache via cache_dir.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    manifest = {
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "model_id": args.model_id,
        "revision": args.revision,
        "dataset": args.dataset,
        "split": args.split,
        "layer_id": args.layer_id,
        "calib_docs": args.calib_docs,
        "eval_docs": args.eval_docs,
        "max_tokens_per_doc": args.max_tokens_per_doc,
        "activation_source": args.activation_source,
        "activation_source_detail": (
            "embed_tokens -> layer0 input RMSNorm -> layer0 attention -> residual add -> post-attention RMSNorm"
            if args.activation_source == "true_layer0_post_attention_norm"
            else "embed_tokens + selected_layer.post_attention_layernorm; attention/prefix layers not executed"
        ),
        "include_shared_expert": args.include_shared_expert,
        "secret_env_present": {
            "HF_TOKEN": bool(os.environ.get("HF_TOKEN")),
            "HUGGING_FACE_HUB_TOKEN": bool(os.environ.get("HUGGING_FACE_HUB_TOKEN")),
        },
    }
    _write_json(args.output_dir / "run_manifest.json", manifest)

    if args.dry_run:
        summary = inspect_only(model_id=args.model_id, revision=args.revision, token=token, cache_dir=args.cache_dir, layer_id=args.layer_id)
        summary["elapsed_sec"] = time.time() - started
        _write_json(args.output_dir / "index_summary.json", summary)
        _write_json(args.output_dir / "ablation_metrics.json", {"dry_run": True, "status": "ok"})
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision, token=token, cache_dir=args.cache_dir)
    layer, layer_summary = load_layer(
        model_id=args.model_id,
        revision=args.revision,
        token=token,
        cache_dir=args.cache_dir,
        layer_id=args.layer_id,
        dtype=torch.float16,
    )
    if args.activation_source == "true_layer0_post_attention_norm" and (args.layer_id != 0 or layer.layer_type != "full_attention"):
        raise ValueError(
            "proper true activation mode currently supports only layer_id=0 with layer_type='full_attention'. "
            f"got layer_id={args.layer_id}, layer_type={layer.layer_type!r}"
        )
    _write_json(args.output_dir / "index_summary.json", layer_summary)

    docs = _load_raw_docs(args.dataset, args.split, args.text_field, args.calib_docs + args.eval_docs)
    calib_docs = docs[: args.calib_docs]
    eval_docs = docs[args.calib_docs :]

    weighted_stats = collect_stats(
        calib_docs,
        tokenizer,
        layer,
        max_tokens_per_doc=args.max_tokens_per_doc,
        block_size=args.block_size,
        device=device,
        weighted=True,
        activation_source=args.activation_source,
    )
    unweighted_stats = collect_stats(
        calib_docs,
        tokenizer,
        layer,
        max_tokens_per_doc=args.max_tokens_per_doc,
        block_size=args.block_size,
        device=device,
        weighted=False,
        activation_source=args.activation_source,
    )

    experiments = [
        ("baseline_fp_layer_local", None),
        ("production_quantization_baseline", "not_available"),
        ("two_bit_expert_quantization_baseline", "not_available"),
        ("kbvq_like_baseline", "not_available"),
        ("A0_shared_naive_1bit_no_hadamard_no_rescue_no_correction", dict(stats=weighted_stats, rank_divisor=128, use_hadamard=False, weighted_scale=False, rescue_config=None, fit_correction=False)),
        ("A1_shared_hadamard_1bit_no_rescue_no_correction", dict(stats=weighted_stats, rank_divisor=128, use_hadamard=True, weighted_scale=False, rescue_config=None, fit_correction=False)),
        ("A2_A1_activation_weighted_binary_scale", dict(stats=weighted_stats, rank_divisor=128, use_hadamard=True, weighted_scale=True, rescue_config=None, fit_correction=False)),
        ("A3_A2_mixed_bit_rescue_rcq_1p55", dict(stats=weighted_stats, rank_divisor=128, use_hadamard=True, weighted_scale=True, rescue_config=RescueConfig.rcq_1p55(args.block_size), fit_correction=False)),
        ("A3_A2_mixed_bit_rescue_rcq_1p75", dict(stats=weighted_stats, rank_divisor=128, use_hadamard=True, weighted_scale=True, rescue_config=RescueConfig.rcq_1p75(args.block_size), fit_correction=False)),
        ("A3_A2_mixed_bit_rescue_rcq_1p90", dict(stats=weighted_stats, rank_divisor=128, use_hadamard=True, weighted_scale=True, rescue_config=RescueConfig.rcq_1p90(args.block_size), fit_correction=False)),
        ("A4_A3_rcq_1p55_routed_moe_output_correction", dict(stats=weighted_stats, rank_divisor=128, use_hadamard=True, weighted_scale=True, rescue_config=RescueConfig.rcq_1p55(args.block_size), fit_correction=True)),
        ("A4_A3_rcq_1p75_routed_moe_output_correction", dict(stats=weighted_stats, rank_divisor=128, use_hadamard=True, weighted_scale=True, rescue_config=RescueConfig.rcq_1p75(args.block_size), fit_correction=True)),
        ("A4_A3_rcq_1p90_routed_moe_output_correction", dict(stats=weighted_stats, rank_divisor=128, use_hadamard=True, weighted_scale=True, rescue_config=RescueConfig.rcq_1p90(args.block_size), fit_correction=True)),
        ("router_weighted_covariance_A4_rcq_1p75", dict(stats=weighted_stats, rank_divisor=128, use_hadamard=True, weighted_scale=True, rescue_config=RescueConfig.rcq_1p75(args.block_size), fit_correction=True)),
        ("unweighted_covariance_A4_rcq_1p75", dict(stats=unweighted_stats, rank_divisor=128, use_hadamard=True, weighted_scale=True, rescue_config=RescueConfig.rcq_1p75(args.block_size), fit_correction=True)),
        ("rank_n_over_256_A4_rcq_1p75", dict(stats=weighted_stats, rank_divisor=256, use_hadamard=True, weighted_scale=True, rescue_config=RescueConfig.rcq_1p75(args.block_size), fit_correction=True)),
        ("rank_n_over_128_A4_rcq_1p75", dict(stats=weighted_stats, rank_divisor=128, use_hadamard=True, weighted_scale=True, rescue_config=RescueConfig.rcq_1p75(args.block_size), fit_correction=True)),
        ("rank_n_over_64_A4_rcq_1p75", dict(stats=weighted_stats, rank_divisor=64, use_hadamard=True, weighted_scale=True, rescue_config=RescueConfig.rcq_1p75(args.block_size), fit_correction=True)),
        ("grouped_subspaces_G1_G2_G4_G8", "not_available"),
        ("per_linear_output_affine_correction", "not_available"),
    ]
    if args.max_experiments is not None:
        experiments = experiments[: args.max_experiments]

    results: list[dict[str, Any]] = []
    for label, spec in experiments:
        if spec is None:
            calib = evaluate_docs(
                calib_docs,
                tokenizer,
                layer,
                None,
                max_tokens_per_doc=args.max_tokens_per_doc,
                device=device,
                include_shared=args.include_shared_expert,
                activation_source=args.activation_source,
            )
            heldout = evaluate_docs(
                eval_docs,
                tokenizer,
                layer,
                None,
                max_tokens_per_doc=args.max_tokens_per_doc,
                device=device,
                include_shared=args.include_shared_expert,
                activation_source=args.activation_source,
            )
            results.append({"label": label, "status": "ok", "calibration": calib, "heldout": heldout})
        elif spec == "not_available":
            results.append({"label": label, "status": "not_available", "reason": "not implemented in this prototype slice"})
        else:
            results.append(
                run_ablation(
                    label,
                    layer,
                    tokenizer,
                    calib_docs,
                    eval_docs,
                    max_tokens_per_doc=args.max_tokens_per_doc,
                    device=device,
                    include_shared=args.include_shared_expert,
                    activation_source=args.activation_source,
                    **spec,
                )
            )
            _write_json(
                args.output_dir / "ablation_metrics.partial.json",
                {"status": "running", "completed": len(results), "results": results, "elapsed_sec": time.time() - started},
            )

    _write_json(
        args.output_dir / "ablation_metrics.json",
        {
            "status": "ok",
            "elapsed_sec": time.time() - started,
            "doc_policy": {
                "calibration": f"first {args.calib_docs} streamed docs",
                "heldout": f"next {args.eval_docs} streamed docs",
                "max_tokens_per_doc": args.max_tokens_per_doc,
                "text_postprocessing": "none; raw dataset text is passed directly to tokenizer with truncation only",
            },
            "activation_source": manifest["activation_source"],
            "activation_source_detail": manifest["activation_source_detail"],
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
