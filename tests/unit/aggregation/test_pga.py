from pytest import mark, raises
from torch import Tensor
from torch.testing import assert_close
from utils.tensors import ones_, tensor_, zeros_

from torchjd.aggregation import OSNMFPGA, SVDPGA, Aggregator

from ._asserts import (
    assert_expected_structure,
    assert_non_differentiable,
)
from ._inputs import scaled_matrices, typical_matrices

aggregators: list[Aggregator] = [
    SVDPGA(),
    SVDPGA(rank=1),
    OSNMFPGA(),
    OSNMFPGA(rank=1),
]
scaled_pairs = [(aggregator, matrix) for aggregator in aggregators for matrix in scaled_matrices]
typical_pairs = [(aggregator, matrix) for aggregator in aggregators for matrix in typical_matrices]
# There is no permutation invariance test: SVDPGA and OSNMFPGA keep eigenvalues down to 1e-10 times
# the largest one, below the rounding noise of a float32 Gramian, so their result can depend on the
# order of nearly dependent gradients.
requires_grad_pairs = [(aggregator, ones_(3, 5, requires_grad=True)) for aggregator in aggregators]


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: Aggregator, matrix: Tensor) -> None:
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], requires_grad_pairs)
def test_non_differentiable(aggregator: Aggregator, matrix: Tensor) -> None:
    assert_non_differentiable(aggregator, matrix)


@mark.parametrize("aggregator", aggregators)
def test_zero(aggregator: Aggregator) -> None:
    """Tests that the zero matrix is aggregated into the zero vector."""

    assert_close(aggregator(zeros_(2, 3)), zeros_(3))


@mark.parametrize("aggregator", aggregators)
def test_single_gradient(aggregator: Aggregator) -> None:
    """Tests that a single gradient is aggregated into its normalized direction."""

    J = tensor_([[3.0, 0.0, 4.0]])
    assert_close(aggregator(J), tensor_([0.6, 0.0, 0.8]))


@mark.parametrize("aggregator", [SVDPGA(), OSNMFPGA()])
def test_orthogonal_gradients(aggregator: Aggregator) -> None:
    """
    Tests that orthogonal gradients of different norms are aggregated into the average of their
    normalized directions when all directions are kept.
    """

    J = tensor_([[3.0, 0.0], [0.0, 4.0]])
    assert_close(aggregator(J), tensor_([0.5, 0.5]))


def test_representations() -> None:
    assert repr(SVDPGA()) == "SVDPGA(rank=None, eigenvalue_tol=1e-10)"
    assert str(SVDPGA()) == "SVDPGA"
    assert (
        repr(OSNMFPGA(rank=2)) == "OSNMFPGA(rank=2, max_iters=30, tol=1e-09, roundoff_scale=16.0, "
        "eigenvalue_tol=1e-10)"
    )
    assert str(OSNMFPGA()) == "OSNMFPGA"


@mark.parametrize("make", [lambda: SVDPGA(rank=0), lambda: OSNMFPGA(rank=0)])
def test_invalid_rank(make) -> None:  # noqa: ANN001
    with raises(ValueError, match="rank"):
        make()


def test_invalid_osnmf_parameters() -> None:
    cls = OSNMFPGA
    with raises(ValueError, match="max_iters"):
        cls(max_iters=0)
    with raises(ValueError, match="tol"):
        cls(tol=0.0)
    with raises(ValueError, match="roundoff_scale"):
        cls(roundoff_scale=-1.0)


@mark.parametrize("cls", [SVDPGA, OSNMFPGA])
def test_invalid_eigenvalue_tol(cls: type[SVDPGA] | type[OSNMFPGA]) -> None:
    with raises(ValueError, match="eigenvalue_tol"):
        cls(eigenvalue_tol=-1.0)
