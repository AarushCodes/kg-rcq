"""Reference PyTorch implementation of the RCQ-MoE prototype pipeline."""

from .codebooks import lloyd_max_codebook
from .correction import OnlineChannelRegression
from .decomposition import SharedDecomposition, choose_rank, decompose_shared_subspace
from .hadamard import block_hadamard_matrix, deterministic_block_signs
from .metrics import kl_divergence_summary
from .quantization import RescueConfig, QuantizedResidual, quantize_residuals
from .storage import StorageBreakdown, expert_bpw
from .toy_moe import QuantizedToyMoeLayer, ToyMoeLayer, fit_toy_moe_output_correction, quantize_toy_moe_layer

__all__ = [
    "OnlineChannelRegression",
    "QuantizedToyMoeLayer",
    "RescueConfig",
    "QuantizedResidual",
    "SharedDecomposition",
    "StorageBreakdown",
    "ToyMoeLayer",
    "block_hadamard_matrix",
    "choose_rank",
    "decompose_shared_subspace",
    "deterministic_block_signs",
    "expert_bpw",
    "fit_toy_moe_output_correction",
    "kl_divergence_summary",
    "lloyd_max_codebook",
    "quantize_residuals",
    "quantize_toy_moe_layer",
]
