import torch

from rcq_moe.codebooks import lloyd_max_codebook
from rcq_moe.hadamard import block_hadamard_matrix, signed_hadamard_q
from rcq_moe.quantization import RescueConfig, binary_quantize_block, lloyd_quantize_block, quantize_residuals
from rcq_moe.storage import expert_bpw


def test_hadamard_is_orthonormal_and_signed_rotation_preserves_dot_product():
    torch.manual_seed(0)
    h = block_hadamard_matrix(8)
    eye = torch.eye(8)
    assert torch.allclose(h @ h.T, eye, atol=1e-6)

    q = signed_hadamard_q("tiny", 2, "up", 1, 8)
    row = torch.randn(8)
    act = torch.randn(8)
    rotated_row = row @ q
    rotated_act = act @ q
    assert torch.allclose(row @ act, rotated_row @ rotated_act, atol=1e-6)


def test_lloyd_max_two_bit_codebook_matches_expected_symmetry_and_scale():
    codebook = lloyd_max_codebook(2)
    assert torch.allclose(codebook, -torch.flip(codebook, dims=[0]), atol=1e-5)
    expected = torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104])
    assert torch.allclose(codebook, expected, atol=8e-4)


def test_binary_quantizer_uses_weighted_mean_absolute_scale():
    z = torch.tensor([[1.0, -3.0, 2.0, -4.0]])
    moments = torch.tensor([[1.0, 2.0, 1.0, 0.0]])
    dequant, scale, score = binary_quantize_block(z, moments)
    expected_scale = torch.tensor([(1.0 * 1.0 + 2.0 * 3.0 + 1.0 * 2.0) / 4.0])
    assert torch.allclose(scale, expected_scale)
    assert torch.equal(torch.sign(dequant), torch.sign(z))
    assert score.item() >= 0.0


def test_rescue_quantizer_reduces_weighted_error_against_binary_on_outlier_block():
    z = torch.tensor([[0.1, -0.2, 0.3, -5.0, 0.2, -0.1, 0.05, 4.0]])
    moments = torch.ones_like(z)
    binary, _, binary_score = binary_quantize_block(z, moments)
    rescued, _ = lloyd_quantize_block(z, moments, bits=4)
    rescue_score = (moments * (z - rescued).square()).sum(dim=-1)
    assert rescue_score.item() < binary_score.item()
    assert rescued.shape == binary.shape


def test_quantize_residuals_selects_global_widths_and_dequantizes_shape():
    torch.manual_seed(1)
    residuals = [torch.randn(5, 10), torch.randn(5, 10)]
    moments = torch.ones(2, 8)
    config = RescueConfig.rcq_1p75(block_size=8)
    quantized = quantize_residuals(residuals, moments, config, model_name="tiny", layer_id=0, linear_type="gate")

    widths = torch.stack([q.widths for q in quantized], dim=0)
    assert int((widths == 4).sum()) == round(widths.numel() * 0.05)
    assert int((widths == 2).sum()) == round(widths.numel() * 0.20)
    for original, q in zip(residuals, quantized):
        assert q.dequantize().shape == original.shape


def test_storage_bpw_counts_padding_excluded_indices_and_shared_overhead():
    widths = torch.tensor([[[1, 4], [2, 1]], [[1, 1], [4, 2]]])
    report = expert_bpw(num_experts=2, rows=2, cols=10, rank=1, widths=widths, block_size=8)
    assert report.index_bits == int((widths[:, :, 0] * 8).sum() + (widths[:, :, 1] * 2).sum())
    assert report.shared_bits == 16 * (1 * 10 + 2 * 2 * 1)
    assert report.total_bits == report.shared_bits + report.index_bits + report.scale_bits + report.metadata_bits
    assert report.bpw == report.total_bits / 40

