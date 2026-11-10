import torch
from torch import Tensor

from torchjd.linalg import PSDMatrix

from ._aggregator_bases import GramianWeightedAggregator
from ._mixins import _NonDifferentiable
from ._weighting_bases import _GramianWeighting


def _sorted_spectrum(gramian: Tensor) -> tuple[Tensor, Tensor]:
    """Eigenvalues (clamped to be nonnegative) and eigenvectors, by decreasing eigenvalue."""

    eigenvalues, eigenvectors = torch.linalg.eigh(gramian)
    eigenvalues = eigenvalues.clamp_min(0)
    order = eigenvalues.argsort(descending=True)
    return eigenvalues[order], eigenvectors[:, order]


def _normalize(gramian: Tensor) -> tuple[Tensor, float]:
    """
    Symmetric Gramian divided by its largest diagonal element, and that element. The PGA directions
    do not depend on the scale of the gradients (their weights just scale inversely with it), so
    working on the normalized Gramian avoids overflow and underflow in low precision.
    """

    # Averaging with the transpose removes the asymmetry due to rounding, as eigh expects a symmetric
    # matrix.
    symmetric = 0.5 * (gramian + gramian.T)
    scale = symmetric.diagonal().max().item()
    if scale <= 0:
        return symmetric, 0.0
    return symmetric / scale, scale


def _select_rank(available_rank: int, rank: int | None) -> int:
    return available_rank if rank is None else min(available_rank, rank)


def _check_rank(rank: int | None) -> None:
    if rank is not None and rank < 1:
        raise ValueError(f"Parameter `rank` should be a positive integer or None. Found {rank=}.")


def _check_eigenvalue_tol(eigenvalue_tol: float) -> None:
    if eigenvalue_tol < 0:
        raise ValueError(
            f"Parameter `eigenvalue_tol` should be nonnegative. Found {eigenvalue_tol=}."
        )


def _check_iterations(max_iters: int, tol: float) -> None:
    if max_iters < 1:
        raise ValueError(f"Parameter `max_iters` should be a positive integer. Found {max_iters=}.")
    if tol <= 0:
        raise ValueError(f"Parameter `tol` should be positive. Found {tol=}.")


def _orient_by_mass(vectors: Tensor, dim: int) -> Tensor:
    """Flips the sign of each vector (along ``dim``) so that most of its squared mass is positive."""

    positive_mass = vectors.clamp_min(0).square().sum(dim=dim, keepdim=True)
    negative_mass = vectors.clamp_max(0).square().sum(dim=dim, keepdim=True)
    return vectors * torch.where(negative_mass > positive_mass, -1.0, 1.0)


def _initial_basis(
    gramian: Tensor, rank: int | None, eigenvalue_tol: float
) -> tuple[Tensor, Tensor] | None:
    """
    Weights of the oriented, unit-norm principal directions of the gradients (the basis is
    ``G @ basis_weights``), and the coordinates of the gradients in the principal basis. Directions
    whose eigenvalue is below ``eigenvalue_tol`` times the largest one are dropped.
    Returns ``None`` if no direction is kept.
    """

    eigenvalues, eigenvectors = _sorted_spectrum(gramian)
    positive = eigenvalues > eigenvalue_tol * eigenvalues[0]
    n_positive = int(positive.sum().item())
    t = _select_rank(n_positive, rank)
    if t == 0:
        return None
    roots = eigenvalues[:n_positive].sqrt()
    vectors = eigenvectors[:, :n_positive]
    basis_weights = _orient_by_mass(vectors[:, :t], dim=0) / roots[:t]
    coordinates = roots.unsqueeze(1) * vectors.T
    return basis_weights, coordinates


class SVDPGAWeighting(_GramianWeighting, _NonDifferentiable):
    """
    :class:`~torchjd.aggregation.Weighting` [:class:`~torchjd.linalg.PSDMatrix`]
    giving the weights of :class:`~torchjd.aggregation.SVDPGA`.

    :param rank: Maximum number of principal directions to average. If ``None``, all directions
        with a non-negligible eigenvalue are used.
    :param eigenvalue_tol: Eigenvalues of the Gramian smaller than this fraction of the largest
        one are considered zero.
    """

    def __init__(self, rank: int | None = None, eigenvalue_tol: float = 1e-10) -> None:
        super().__init__()
        _check_rank(rank)
        _check_eigenvalue_tol(eigenvalue_tol)
        self.rank = rank
        self.eigenvalue_tol = eigenvalue_tol

    def forward(self, gramian: PSDMatrix, /) -> Tensor:
        gram, scale = _normalize(gramian)
        basis = _initial_basis(gram, self.rank, self.eigenvalue_tol) if scale > 0 else None
        if basis is None:
            return torch.zeros(gramian.shape[0], dtype=gramian.dtype, device=gramian.device)
        basis_weights, _ = basis
        return basis_weights.mean(dim=1) / scale**0.5


class SVDPGA(GramianWeightedAggregator, _NonDifferentiable):
    """
    :class:`~torchjd.aggregation.GramianWeightedAggregator` that averages the principal directions
    (left singular vectors) of the gradients, each oriented towards the side where most of the
    gradients lie.

    :param rank: Maximum number of principal directions to average. If ``None``, all directions
        with a non-negligible eigenvalue are used.
    :param eigenvalue_tol: Eigenvalues of the Gramian smaller than this fraction of the largest
        one are considered zero.
    """

    gramian_weighting: SVDPGAWeighting

    def __init__(self, rank: int | None = None, eigenvalue_tol: float = 1e-10) -> None:
        super().__init__(SVDPGAWeighting(rank, eigenvalue_tol))

    def __repr__(self) -> str:
        weighting = self.gramian_weighting
        return (
            f"{self.__class__.__name__}(rank={weighting.rank}, "
            f"eigenvalue_tol={weighting.eigenvalue_tol})"
        )


class OSNMFPGAWeighting(_GramianWeighting, _NonDifferentiable):
    """
    :class:`~torchjd.aggregation.Weighting` [:class:`~torchjd.linalg.PSDMatrix`]
    giving the weights of :class:`~torchjd.aggregation.OSNMFPGA`.

    :param rank: Maximum number of basis directions. If ``None``, all directions with a
        non-negligible eigenvalue are used.
    :param max_iters: Maximum number of alternating updates of the factorization.
    :param tol: Relative change of the reconstruction loss below which the updates stop.
    :param roundoff_scale: Absolute change of the reconstruction loss attributed to rounding, in
        units of the machine epsilon times the squared norm of the gradients. Changes of the loss
        smaller than ``tol`` times the loss plus this amount also stop the updates.
    :param eigenvalue_tol: Eigenvalues of the Gramian smaller than this fraction of the largest
        one are considered zero.
    """

    def __init__(
        self,
        rank: int | None = None,
        max_iters: int = 30,
        tol: float = 1e-9,
        roundoff_scale: float = 16.0,
        eigenvalue_tol: float = 1e-10,
    ) -> None:
        super().__init__()
        _check_rank(rank)
        _check_iterations(max_iters, tol)
        _check_eigenvalue_tol(eigenvalue_tol)
        if roundoff_scale < 0:
            raise ValueError(
                f"Parameter `roundoff_scale` should be nonnegative. Found {roundoff_scale=}."
            )
        self.rank = rank
        self.max_iters = max_iters
        self.tol = tol
        self.roundoff_scale = roundoff_scale
        self.eigenvalue_tol = eigenvalue_tol

    def forward(self, gramian: PSDMatrix, /) -> Tensor:
        gram, scale = _normalize(gramian)
        basis = _initial_basis(gram, self.rank, self.eigenvalue_tol) if scale > 0 else None
        if basis is None:
            return torch.zeros(gram.shape[0], dtype=gram.dtype, device=gram.device)
        basis_weights, coordinates = basis

        energy = gram.trace().item()
        roundoff = self.roundoff_scale * torch.finfo(gram.dtype).eps * energy
        factor_h = (basis_weights.T @ gram).clamp_min(0)
        previous = _gram_loss(gram, basis_weights, factor_h)
        for _ in range(self.max_iters):
            basis_weights = _gram_procrustes_step(gram, factor_h, coordinates, self.eigenvalue_tol)
            loss = _gram_loss(gram, basis_weights, factor_h)
            factor_h = (basis_weights.T @ gram).clamp_min(0)
            if abs(previous - loss) <= self.tol * abs(previous) + roundoff:
                break
            previous = loss

        return basis_weights.mean(dim=1) / scale**0.5


def _gram_procrustes_step(
    gram: Tensor, factor_h: Tensor, coordinates: Tensor, eigenvalue_tol: float
) -> Tensor:
    """
    Weights B of the orthonormal basis W = G @ B closest to G @ H.T. Uses the Gramian of G @ H.T,
    and completes the solution with a reduced SVD of the gradients' coordinates when that Gramian is
    ill-conditioned.
    """

    small_gram = factor_h @ gram @ factor_h.T
    small_gram = 0.5 * (small_gram + small_gram.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(small_gram)
    if eigenvalues[0] <= eigenvalue_tol * eigenvalues[-1]:
        u, _, v_t = torch.linalg.svd(coordinates @ factor_h.T, full_matrices=False)
        return (coordinates / coordinates.square().sum(dim=1, keepdim=True)).T @ (u @ v_t)
    return (factor_h.T @ (eigenvectors / eigenvalues.sqrt())) @ eigenvectors.T


def _gram_loss(gram: Tensor, basis_weights: Tensor, factor_h: Tensor) -> float:
    """Value of 0.5 * ||G - W @ H||^2 for the basis W = G @ B, computed from the Gramian of G."""

    gram_b = gram @ basis_weights
    cross = (factor_h * gram_b.T).sum()
    quadratic = (factor_h * ((basis_weights.T @ gram_b) @ factor_h)).sum()
    return (0.5 * (gram.trace() - 2 * cross + quadratic)).item()


class OSNMFPGA(GramianWeightedAggregator, _NonDifferentiable):
    """
    :class:`~torchjd.aggregation.GramianWeightedAggregator` that approximates the gradients by
    :math:`WH`, with orthonormal columns in :math:`W` and nonnegative entries in :math:`H`
    (orthogonal semi-nonnegative matrix factorization), and averages the columns of :math:`W`.

    :param rank: Maximum number of basis directions. If ``None``, all directions with a
        non-negligible eigenvalue are used.
    :param max_iters: Maximum number of alternating updates of the factorization.
    :param tol: Relative change of the reconstruction loss below which the updates stop.
    :param roundoff_scale: Absolute change of the reconstruction loss attributed to rounding, in
        units of the machine epsilon times the squared norm of the gradients. Changes of the loss
        smaller than ``tol`` times the loss plus this amount also stop the updates.
    :param eigenvalue_tol: Eigenvalues of the Gramian smaller than this fraction of the largest
        one are considered zero.
    """

    gramian_weighting: OSNMFPGAWeighting

    def __init__(
        self,
        rank: int | None = None,
        max_iters: int = 30,
        tol: float = 1e-9,
        roundoff_scale: float = 16.0,
        eigenvalue_tol: float = 1e-10,
    ) -> None:
        super().__init__(OSNMFPGAWeighting(rank, max_iters, tol, roundoff_scale, eigenvalue_tol))

    def __repr__(self) -> str:
        weighting = self.gramian_weighting
        return (
            f"{self.__class__.__name__}(rank={weighting.rank}, "
            f"max_iters={weighting.max_iters}, tol={weighting.tol}, "
            f"roundoff_scale={weighting.roundoff_scale}, "
            f"eigenvalue_tol={weighting.eigenvalue_tol})"
        )
