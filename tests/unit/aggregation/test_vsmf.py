"""Tests for :class:`~torchjd.aggregation.VSMF` and :class:`~torchjd.aggregation.VSMFWeighting`."""

from __future__ import annotations

import torch
import torch.nn as nn
from pytest import skip
from torch.testing import assert_close

from torchjd.aggregation import MGDA, VSMF, Mean, VSMFWeighting
from torchjd.autojac import backward, jac_to_grad


def test_output_shape() -> None:
    M, D, K = 8, 20, 3
    aggregator = VSMF(K=K, num_inner_iters=5, seed=42)
    X = torch.randn(M, D)
    result = aggregator(X)
    assert result.shape == (D,)


def test_gpu_if_available() -> None:
    if not torch.cuda.is_available():
        skip("CUDA not available")
    M, D, K = 8, 20, 3
    aggregator = VSMF(K=K, seed=0)
    X = torch.randn(M, D, device="cuda")
    result = aggregator(X)
    assert result.device.type == "cuda"
    assert result.shape == (D,)


def test_single_objective_matches_downstream() -> None:
    M, D, K = 1, 10, 1
    aggregator = VSMF(K=K, num_inner_iters=5, seed=0)
    X = torch.randn(M, D)
    result = aggregator(X)
    assert result.shape == (D,)
    expected = MGDA()(X)
    assert_close(result, expected)


def test_k_equals_m() -> None:
    M, D, K = 4, 15, 4
    aggregator = VSMF(K=K, num_inner_iters=5, seed=0)
    X = torch.randn(M, D)
    result = aggregator(X)
    assert result.shape == (D,)


def test_no_nan_inf() -> None:
    M, D, K = 10, 50, 4
    aggregator = VSMF(K=K, num_inner_iters=10, seed=1)
    X = torch.randn(M, D)
    result = aggregator(X)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


def test_warm_start_state_stored() -> None:
    M, D, K = 6, 20, 3
    aggregator = VSMF(K=K, warm_start=True, seed=0)
    X = torch.randn(M, D)
    aggregator(X)
    assert aggregator._W is not None
    assert aggregator._H is not None
    assert aggregator._W.shape == (M, K)
    assert aggregator._H.shape == (K, D)


def test_warm_start_second_call() -> None:
    M, D, K = 6, 20, 3
    aggregator = VSMF(K=K, warm_start=True, num_inner_iters=3, seed=0)
    X1 = torch.randn(M, D)
    X2 = torch.randn(M, D)
    r1 = aggregator(X1)
    r2 = aggregator(X2)
    assert r1.shape == (D,)
    assert r2.shape == (D,)


def test_reset_state() -> None:
    M, D, K = 6, 20, 3
    aggregator = VSMF(K=K, warm_start=True, seed=0)
    aggregator(torch.randn(M, D))
    aggregator.reset_state()
    assert aggregator._W is None
    assert aggregator._H is None
    assert aggregator._W_momentum is None


def test_soft_assignment() -> None:
    M, D, K = 8, 20, 3
    aggregator = VSMF(K=K, use_binary=False, seed=0)
    result = aggregator(torch.randn(M, D))
    assert result.shape == (D,)
    assert not torch.isnan(result).any()


def test_preference_weights() -> None:
    M, D, K = 8, 20, 3
    lam = torch.rand(M)
    aggregator = VSMF(K=K, preference_weights=lam, seed=0)
    result = aggregator(torch.randn(M, D))
    assert result.shape == (D,)


def test_momentum() -> None:
    M, D, K = 6, 20, 3
    aggregator = VSMF(K=K, momentum_beta=0.9, warm_start=True, seed=0)
    for _ in range(5):
        aggregator(torch.randn(M, D))


def test_different_downstream_aggregators() -> None:
    M, D, K = 8, 20, 3
    for agg in (Mean(), MGDA()):
        aggregator = VSMF(K=K, downstream_aggregator=agg, seed=0)
        result = aggregator(torch.randn(M, D))
        assert result.shape == (D,)


def test_large_m_small_k() -> None:
    M, D, K = 64, 100, 4
    aggregator = VSMF(K=K, num_inner_iters=5, seed=0)
    result = aggregator(torch.randn(M, D))
    assert result.shape == (D,)
    assert not torch.isnan(result).any()


def test_vsmf_weighting_output_shape() -> None:
    M, K_dim = 8, 3
    weighting = VSMFWeighting(K=K_dim, seed=0)
    gramian = torch.randn(M, M)
    gramian = gramian @ gramian.T
    result = weighting(gramian)
    assert result.shape == (M,)


def test_vsmf_weighting_weights_sum_to_one() -> None:
    M, K_dim = 8, 3
    weighting = VSMFWeighting(K=K_dim, seed=0)
    gramian = torch.eye(M)
    result = weighting(gramian)
    assert abs(result.sum().item() - 1.0) < 1e-4


def test_vsmf_weighting_no_nan() -> None:
    M, K_dim = 8, 3
    weighting = VSMFWeighting(K=K_dim, seed=0)
    gramian = torch.eye(M)
    result = weighting(gramian)
    assert not torch.isnan(result).any()


def test_autojac_backward() -> None:
    model = nn.Sequential(nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 1))
    aggregator = VSMF(K=2, num_inner_iters=5, seed=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    inputs = torch.randn(4, 5)
    targets = torch.randn(4, 1)
    loss_fn = nn.MSELoss(reduction="none")

    outputs = model(inputs)
    losses = loss_fn(outputs, targets).squeeze()

    params = list(model.parameters())
    backward(losses, inputs=params, parallel_chunk_size=1)
    jac_to_grad(params, aggregator)
    optimizer.step()
    optimizer.zero_grad()
