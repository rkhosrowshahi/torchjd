r"""
VSMF (Versatile Sparse Matrix Factorization) aggregation.

:class:`VSMF` is the main entry point for the autojac path (full Jacobian).
:class:`VSMFWeighting` operates on the objective Gramian :math:`X X^T` (autogram path).

Module-level helpers support factorization diagnostics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from torchjd._linalg import Matrix, PSDMatrix

from ._aggregator_bases import Aggregator
from ._mgda import MGDA, MGDAWeighting
from ._weighting_bases import Weighting


class VSMF(Aggregator):
    r"""
    :class:`~torchjd.aggregation._aggregator_bases.Aggregator` that clusters objective gradients with
    Versatile Sparse Matrix Factorization (VSMF), then applies a downstream aggregator (default:
    :class:`~torchjd.aggregation.MGDA`) on the :math:`K` group representatives.

    At each call, the Jacobian :math:`X` of shape :math:`[M, D]` is factorized as :math:`X \approx W
    H` with alternating explicit gradient steps on :math:`W` and :math:`H`. Rows of :math:`W` are
    turned into group weights (hard or soft), producing a grouped Jacobian :math:`G` of shape
    :math:`[K, D]`, which is then aggregated.

    :param K: Number of groups / factorization rank. Should satisfy :math:`K \ll M`.
    :param alpha1: :math:`\ell_1` penalty on :math:`W` (sparsity of group membership).
    :param alpha2: :math:`\ell_2` penalty on :math:`W`.
    :param lambda1: :math:`\ell_1` penalty on :math:`H`.
    :param lambda2: :math:`\ell_2` penalty on :math:`H`.
    :param num_inner_iters: Alternating :math:`W` / :math:`H` steps per forward call.
    :param lr_inner: Step size for inner updates.
    :param use_binary: If ``True``, hard-assign each objective via row-wise argmax on :math:`W`. If
        ``False``, use a row-normalized soft weighting from :math:`|W|`.
    :param preference_weights: Optional per-objective weights of shape :math:`[M]`.
    :param downstream_aggregator: Aggregator applied to :math:`G` of shape :math:`[K, D]`.
    :param warm_start: If ``True``, reuse previous :math:`W, H` when shapes match.
    :param momentum_beta: EMA on :math:`W` after inner iterations (``0`` disables).
    :param residual_reinit_threshold: Relative residual :math:`\|X - WH\|_F / (\|X\|_F + \epsilon)`
        above which warm-start state is cleared. Use ``inf`` to disable.
    :param device: Reserved for future use; tensors follow the input ``matrix`` device.
    :param seed: RNG seed for cold-start initialization.
    """

    def __init__(
        self,
        K: int = 4,
        alpha1: float = 0.01,
        alpha2: float = 0.01,
        lambda1: float = 0.0,
        lambda2: float = 0.01,
        num_inner_iters: int = 5,
        lr_inner: float = 0.01,
        use_binary: bool = True,
        preference_weights: Tensor | None = None,
        downstream_aggregator: Aggregator | None = None,
        warm_start: bool = True,
        momentum_beta: float = 0.0,
        residual_reinit_threshold: float = 2.0,
        device: torch.device | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.K = K
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.num_inner_iters = num_inner_iters
        self.lr_inner = lr_inner
        self.use_binary = use_binary
        self.preference_weights = preference_weights
        self.downstream_aggregator = downstream_aggregator or MGDA()
        self.warm_start = warm_start
        self.momentum_beta = momentum_beta
        self.residual_reinit_threshold = residual_reinit_threshold
        self.device = device
        self.seed = seed

        self._W: Tensor | None = None
        self._H: Tensor | None = None
        self._W_momentum: Tensor | None = None

    def __repr__(self) -> str:
        parts = [
            f"K={self.K}",
            f"alpha1={self.alpha1}",
            f"alpha2={self.alpha2}",
            f"lambda1={self.lambda1}",
            f"lambda2={self.lambda2}",
            f"num_inner_iters={self.num_inner_iters}",
            f"lr_inner={self.lr_inner}",
            f"use_binary={self.use_binary}",
            f"warm_start={self.warm_start}",
            f"momentum_beta={self.momentum_beta}",
            f"residual_reinit_threshold={self.residual_reinit_threshold}",
        ]
        if self.device is not None:
            parts.append(f"device={self.device!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def forward(self, matrix: Matrix, /) -> Tensor:
        """
        :param matrix: Jacobian of shape :math:`[M, D]`.
        """
        X = matrix
        M, _D = X.shape
        K = self.K
        device = X.device
        dtype = X.dtype

        W, H = self._initialize_or_warmstart(M, _D, K, device, dtype)

        for _ in range(self.num_inner_iters):
            W, H = self._update_step(X, W, H)

        if (
            self.momentum_beta > 0.0
            and self._W_momentum is not None
            and self._W_momentum.shape == W.shape
        ):
            W = self.momentum_beta * self._W_momentum + (1.0 - self.momentum_beta) * W
        self._W_momentum = W.detach().clone()

        with torch.no_grad():
            residual = (X - W @ H).norm() / (X.norm() + 1e-8)
            force_reinit = residual.item() > self.residual_reinit_threshold

        if self.warm_start and not force_reinit:
            self._W = W.detach().clone()
            self._H = H.detach().clone()
        else:
            self._W = None
            self._H = None

        membership_norm, empty_groups = self._group_membership_matrix(W, M, K, device, dtype)
        G = membership_norm.T @ X
        if empty_groups.any():
            mask = (~empty_groups).to(dtype=dtype).unsqueeze(1)
            G = G * mask

        return self.downstream_aggregator(G)

    def _initialize_or_warmstart(
        self,
        M: int,
        D: int,
        K: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        if (
            self.warm_start
            and self._W is not None
            and self._H is not None
            and self._W.shape == (M, K)
            and self._H.shape == (K, D)
        ):
            W = self._W.to(device=device, dtype=dtype).requires_grad_(False)
            H = self._H.to(device=device, dtype=dtype).requires_grad_(False)
        else:
            generator = torch.Generator(device=device)
            if self.seed is not None:
                generator.manual_seed(self.seed)
            W = torch.randn(M, K, device=device, dtype=dtype, generator=generator)
            H = torch.randn(K, D, device=device, dtype=dtype, generator=generator)
            W = F.normalize(W, dim=0)
            H = F.normalize(H, dim=1)
        return W.clone(), H.clone()

    def _update_step(self, X: Tensor, W: Tensor, H: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            WtW = W.T @ W
            WtX = W.T @ X
            HHt = H @ H.T
            XHt = X @ H.T

            grad_H = WtW @ H - WtX
            if self.lambda2 > 0:
                grad_H = grad_H + self.lambda2 * H
            if self.lambda1 > 0:
                grad_H = grad_H + self.lambda1 * torch.sign(H)
            H = H - self.lr_inner * grad_H

            grad_W = W @ HHt - XHt
            if self.alpha2 > 0:
                grad_W = grad_W + self.alpha2 * W
            if self.alpha1 > 0:
                grad_W = grad_W + self.alpha1 * torch.sign(W)
            W = W - self.lr_inner * grad_W

        return W, H

    def _group_membership_matrix(
        self,
        W: Tensor,
        M: int,
        K: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """
        Returns column-normalized membership weights (detached) and an empty-group mask [K].
        """
        with torch.no_grad():
            if self.use_binary:
                assignments = W.argmax(dim=1)
                B = torch.zeros(M, K, device=device, dtype=dtype)
                B[torch.arange(M, device=device), assignments] = 1.0
                membership = B
            else:
                W_pos = W.abs()
                row_sums = W_pos.sum(dim=1, keepdim=True).clamp(min=1e-8)
                membership = W_pos / row_sums

            if self.preference_weights is not None:
                lam = self.preference_weights.to(device=device, dtype=dtype)
                membership = membership * lam.unsqueeze(1)

            col_sums = membership.sum(dim=0, keepdim=True).clamp(min=1e-8)
            membership_norm = membership / col_sums
            empty_groups = membership.sum(dim=0) < 1e-8

        return membership_norm, empty_groups

    def reset_state(self) -> None:
        """Clears warm-start and momentum buffers."""
        self._W = None
        self._H = None
        self._W_momentum = None

    def get_residual(self, matrix: Tensor) -> float:
        """
        Relative Frobenius residual using stored :math:`W, H`. Returns ``inf`` if none.
        """
        if self._W is None or self._H is None:
            return float("inf")
        X = matrix
        W = self._W.to(device=X.device, dtype=X.dtype)
        H = self._H.to(device=X.device, dtype=X.dtype)
        return ((X - W @ H).norm() / (X.norm() + 1e-8)).item()


class VSMFWeighting(Weighting[PSDMatrix]):
    r"""
    :class:`~torchjd.aggregation._weighting_bases.Weighting` that clusters objectives using a
    Gramian-only factorization step, then maps weights from a downstream weighting on the
    :math:`K \times K` grouped Gramian.

    The primary Jacobian-descent interface is :class:`VSMF`.

    :param K: Number of groups.
    :param alpha1: :math:`\ell_1` penalty on :math:`W`.
    :param alpha2: :math:`\ell_2` penalty on :math:`W`.
    :param num_inner_iters: Inner gradient steps on :math:`W` per call.
    :param lr_inner: Step size for inner updates.
    :param downstream_weighting: Weighting applied to the grouped Gramian of shape :math:`[K, K]`.
    :param warm_start: If ``True``, reuse previous :math:`W` when shapes match.
    :param seed: RNG seed for cold-start initialization.
    """

    def __init__(
        self,
        K: int = 4,
        alpha1: float = 0.01,
        alpha2: float = 0.01,
        num_inner_iters: int = 5,
        lr_inner: float = 0.01,
        downstream_weighting: Weighting | None = None,
        warm_start: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.K = K
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.num_inner_iters = num_inner_iters
        self.lr_inner = lr_inner
        self.downstream_weighting = downstream_weighting or MGDAWeighting()
        self.warm_start = warm_start
        self.seed = seed
        self._W: Tensor | None = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(K={self.K}, alpha1={self.alpha1}, alpha2={self.alpha2}, "
            f"num_inner_iters={self.num_inner_iters}, lr_inner={self.lr_inner}, "
            f"warm_start={self.warm_start})"
        )

    def forward(self, gramian: PSDMatrix, /) -> Tensor:
        """
        :param gramian: Matrix :math:`X X^T` of shape :math:`[M, M]`.
        """
        G = gramian
        M = G.shape[0]
        K = self.K
        device = G.device
        dtype = G.dtype

        if self.warm_start and self._W is not None and self._W.shape == (M, K):
            W = self._W.to(device=device, dtype=dtype).clone()
        else:
            generator = torch.Generator(device=device)
            if self.seed is not None:
                generator.manual_seed(self.seed)
            W = torch.randn(M, K, device=device, dtype=dtype, generator=generator)

        with torch.no_grad():
            for _ in range(self.num_inner_iters):
                WtW = W.T @ W
                GW = G @ W
                grad_W = W @ WtW - GW
                if self.alpha2 > 0:
                    grad_W = grad_W + self.alpha2 * W
                if self.alpha1 > 0:
                    grad_W = grad_W + self.alpha1 * torch.sign(W)
                W = W - self.lr_inner * grad_W

        if self.warm_start:
            self._W = W.detach().clone()

        assignments = W.argmax(dim=1)
        B = torch.zeros(M, K, device=device, dtype=dtype)
        B[torch.arange(M, device=device), assignments] = 1.0
        group_sizes = B.sum(dim=0).clamp(min=1.0)
        BtGB = B.T @ G @ B
        norm = group_sizes.unsqueeze(1) * group_sizes.unsqueeze(0)
        grouped_gramian = BtGB / norm

        k_weights = self.downstream_weighting(grouped_gramian)

        m_weights = torch.zeros(M, device=device, dtype=dtype)
        for k in range(K):
            mask = assignments == k
            if mask.any():
                m_weights[mask] = k_weights[k] / group_sizes[k]

        return m_weights / m_weights.sum().clamp(min=1e-8)


def compute_factorization_residual(X: Tensor, W: Tensor, H: Tensor) -> float:
    r"""Relative Frobenius residual :math:`\|X - WH\|_F / \|X\|_F`."""
    return ((X - W @ H).norm() / (X.norm() + 1e-8)).item()


def compute_group_conflict_matrix(X: Tensor, assignments: Tensor, K: int) -> Tensor:
    """
    :math:`K \times K` cosine similarities between group-mean gradients.
    """
    _, D = X.shape
    device = X.device
    dtype = X.dtype
    group_means = torch.zeros(K, D, device=device, dtype=dtype)
    for k in range(K):
        mask = assignments == k
        if mask.any():
            group_means[k] = X[mask].mean(dim=0)
    norms = group_means.norm(dim=1, keepdim=True).clamp(min=1e-8)
    group_means_normalized = group_means / norms
    return group_means_normalized @ group_means_normalized.T


def compute_intra_group_coherence(X: Tensor, assignments: Tensor, K: int) -> Tensor:
    """
    Per-group mean cosine similarity between distinct pairs in the group.
    """
    coherences = torch.zeros(K, device=X.device, dtype=X.dtype)
    for k in range(K):
        mask = assignments == k
        Xk = X[mask]
        if Xk.shape[0] < 2:
            coherences[k] = 1.0
            continue
        norms = Xk.norm(dim=1, keepdim=True).clamp(min=1e-8)
        Xk_n = Xk / norms
        cos_sim = Xk_n @ Xk_n.T
        n_k = Xk.shape[0]
        coherences[k] = (cos_sim.sum() - n_k) / (n_k * (n_k - 1))
    return coherences
