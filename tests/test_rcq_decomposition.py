import torch

from rcq_moe.decomposition import choose_rank, decompose_shared_output_subspace, decompose_shared_subspace, klt_transform


def test_choose_rank_follows_spec_but_caps_to_tiny_dimension():
    assert choose_rank(4) == 4
    assert choose_rank(2048) == 16
    assert choose_rank(16384) == 64


def test_klt_transform_is_inverse_after_eigenvalue_flooring():
    cov = torch.diag(torch.tensor([4.0, 1.0, 0.0], dtype=torch.float64))
    eigvals, eigvecs, t, t_inv = klt_transform(cov)
    assert torch.all(eigvals > 0)
    assert torch.allclose(t @ t_inv, torch.eye(3, dtype=torch.float64), atol=1e-8)
    assert torch.allclose(eigvecs.T @ eigvecs, torch.eye(3, dtype=torch.float64), atol=1e-8)


def test_shared_decomposition_reconstructs_weight_as_shared_plus_residual():
    torch.manual_seed(2)
    weights = [torch.randn(6, 5, dtype=torch.float64) for _ in range(3)]
    a = torch.randn(32, 5, dtype=torch.float64)
    covariance = a.T @ a / a.shape[0]
    importance = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)

    decomp = decompose_shared_subspace(weights, covariance, importance, rank=3)
    assert len(decomp.a_factors) == 3
    assert decomp.b_shared.shape == (3, 5)
    assert decomp.v_r.shape == (5, 3)
    assert 0.0 <= decomp.captured_energy <= 1.0

    for weight, a_factor, residual in zip(weights, decomp.a_factors, decomp.residuals):
        shared = a_factor @ decomp.b_shared
        assert torch.allclose(shared + residual, weight, atol=1e-10)


def test_full_rank_shared_decomposition_has_near_zero_residual():
    torch.manual_seed(3)
    weights = [torch.randn(4, 4, dtype=torch.float64) for _ in range(2)]
    covariance = torch.eye(4, dtype=torch.float64)
    importance = torch.ones(2, dtype=torch.float64)

    decomp = decompose_shared_subspace(weights, covariance, importance, rank=4)
    for residual in decomp.residuals:
        assert residual.norm().item() < 1e-10
    assert decomp.captured_energy > 0.999999


def test_left_output_decomposition_full_rank_reconstructs_weight():
    torch.manual_seed(4)
    weights = [torch.randn(5, 7, dtype=torch.float64) for _ in range(3)]
    output_covariance = torch.randn(5, 5, dtype=torch.float64)
    output_covariance = output_covariance @ output_covariance.T

    decomp = decompose_shared_output_subspace(weights, output_covariance, rank=5)

    assert decomp.u_shared.shape == (5, 5)
    assert torch.allclose(decomp.u_shared.T @ decomp.u_shared, torch.eye(5, dtype=torch.float64), atol=1e-8)
    for weight, c_factor, residual in zip(weights, decomp.c_factors, decomp.residuals):
        assert c_factor.shape == (5, 7)
        assert torch.allclose(decomp.u_shared @ c_factor + residual, weight, atol=1e-10)
        assert residual.norm().item() < 1e-10


def test_left_output_decomposition_low_rank_shapes_and_rejects_rank_zero():
    torch.manual_seed(5)
    weights = [torch.randn(6, 4) for _ in range(2)]
    output_covariance = torch.eye(6)

    decomp = decompose_shared_output_subspace(weights, output_covariance, rank=3)

    assert decomp.u_shared.shape == (6, 3)
    assert len(decomp.c_factors) == 2
    assert decomp.c_factors[0].shape == (3, 4)
    assert decomp.residuals[0].shape == (6, 4)
    assert 0.0 <= decomp.captured_energy <= 1.0
    try:
        decompose_shared_output_subspace(weights, output_covariance, rank=0)
    except ValueError:
        pass
    else:
        raise AssertionError("rank 0 should be rejected for left_output")
