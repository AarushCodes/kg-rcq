"""Reference PyTorch implementation of the RCQ-MoE prototype pipeline."""

from .codebooks import lloyd_max_codebook
from .hadamard import block_hadamard_matrix, deterministic_block_signs
from .quantization import RescueConfig, QuantizedResidual, quantize_residuals
from .storage import StorageBreakdown, expert_bpw

__all__ = [
    "RescueConfig",
    "QuantizedResidual",
    "StorageBreakdown",
    "block_hadamard_matrix",
    "deterministic_block_signs",
    "expert_bpw",
    "lloyd_max_codebook",
    "quantize_residuals",
]

