"""Reference PyTorch implementation of the RCQ-MoE prototype pipeline."""

from .codebooks import lloyd_max_codebook
from .correction import OnlineChannelRegression
from .artifact import load_official_qwen_rcq_artifact, save_official_qwen_rcq_artifact
from .decomposition import SharedDecomposition, choose_rank, decompose_shared_subspace
from .hadamard import block_hadamard_matrix, deterministic_block_signs
from .harness import OfficialQwenAblationResult, OfficialQwenHarnessResult, run_official_qwen_ablation, run_official_qwen_harness
from .metrics import kl_divergence_summary
from .official_qwen import (
    OfficialQwen35MoeRCQExperts,
    OfficialQwen35MoeRCQSparseMoeBlock,
    collect_official_qwen_mlp_inputs,
    convert_official_qwen35_moe_to_rcq,
    make_tiny_official_qwen35_moe,
)
from .quantization import RescueConfig, QuantizedResidual, quantize_residuals
from .storage import StorageBreakdown, expert_bpw
from .toy_moe import QuantizedToyMoeLayer, ToyMoeLayer, fit_toy_moe_output_correction, quantize_toy_moe_layer

__all__ = [
    "OnlineChannelRegression",
    "OfficialQwen35MoeRCQExperts",
    "OfficialQwen35MoeRCQSparseMoeBlock",
    "OfficialQwenAblationResult",
    "OfficialQwenHarnessResult",
    "QuantizedToyMoeLayer",
    "RescueConfig",
    "QuantizedResidual",
    "SharedDecomposition",
    "StorageBreakdown",
    "ToyMoeLayer",
    "block_hadamard_matrix",
    "choose_rank",
    "collect_official_qwen_mlp_inputs",
    "convert_official_qwen35_moe_to_rcq",
    "decompose_shared_subspace",
    "deterministic_block_signs",
    "expert_bpw",
    "fit_toy_moe_output_correction",
    "kl_divergence_summary",
    "load_official_qwen_rcq_artifact",
    "lloyd_max_codebook",
    "make_tiny_official_qwen35_moe",
    "quantize_residuals",
    "quantize_toy_moe_layer",
    "run_official_qwen_ablation",
    "run_official_qwen_harness",
    "save_official_qwen_rcq_artifact",
]
