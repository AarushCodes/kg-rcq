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
from .text_data import (
    TextBatchConfig,
    TextFixtureConfig,
    TextTokenBatch,
    encode_texts_as_toy_byte_tokens,
    read_text_fixture,
    texts_to_hf_token_batch,
    texts_to_input_ids,
    texts_to_toy_token_batch,
)
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
    "TextBatchConfig",
    "TextFixtureConfig",
    "TextTokenBatch",
    "ToyMoeLayer",
    "block_hadamard_matrix",
    "choose_rank",
    "collect_official_qwen_mlp_inputs",
    "convert_official_qwen35_moe_to_rcq",
    "decompose_shared_subspace",
    "deterministic_block_signs",
    "encode_texts_as_toy_byte_tokens",
    "expert_bpw",
    "fit_toy_moe_output_correction",
    "kl_divergence_summary",
    "load_official_qwen_rcq_artifact",
    "lloyd_max_codebook",
    "make_tiny_official_qwen35_moe",
    "quantize_residuals",
    "quantize_toy_moe_layer",
    "read_text_fixture",
    "run_official_qwen_ablation",
    "run_official_qwen_harness",
    "save_official_qwen_rcq_artifact",
    "texts_to_hf_token_batch",
    "texts_to_input_ids",
    "texts_to_toy_token_batch",
]
