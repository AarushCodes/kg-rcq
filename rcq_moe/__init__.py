"""Reference PyTorch implementation of the RCQ-MoE prototype pipeline."""

from .codebooks import lloyd_max_codebook
from .decomposition import SharedDecomposition, choose_rank, decompose_shared_subspace
from .hadamard import block_hadamard_matrix, deterministic_block_signs
from .quantization import RescueConfig, QuantizedResidual, quantize_residuals
from .storage import StorageBreakdown, expert_bpw

__all__ = [
    "RescueConfig",
    "QuantizedResidual",
    "SharedDecomposition",
    "StorageBreakdown",
    "block_hadamard_matrix",
    "choose_rank",
    "decompose_shared_subspace",
    "deterministic_block_signs",
    "expert_bpw",
    "lloyd_max_codebook",
    "quantize_residuals",
]
