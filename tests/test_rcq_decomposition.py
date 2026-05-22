import torch

from rcq_moe.decomposition import choose_rank, decompose_shared_subspace, klt_transform


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

