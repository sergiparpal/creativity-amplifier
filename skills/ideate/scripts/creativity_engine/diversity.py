"""Diverse selection via DPP, plus diversity metrics.

A Determinantal Point Process favours subsets whose items are mutually
dissimilar: the probability of a set is proportional to the determinant of the
kernel submatrix, which is the squared volume they span. We use the standard
**fast greedy MAP** inference (Chen et al., 2018) to pick a diverse slate, with a
farthest-point fallback if anything degenerates.

Metrics here (mean pairwise distance, Vendi score) are used by the value gate to
prove the diverse slate beats a non-diverse baseline.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Kernel
# --------------------------------------------------------------------------- #
def build_kernel(
    vecs: np.ndarray,
    quality: Optional[np.ndarray] = None,
    jitter: float = 1e-6,
) -> np.ndarray:
    """Build a PSD DPP kernel ``L = diag(q) (X Xᵀ) diag(q) + jitter·I``.

    ``X Xᵀ`` is a Gram matrix (always PSD). ``quality`` weights items so a DPP
    trades off diversity against per-item quality; default is uniform. The jitter
    makes it strictly PD so greedy log-det inference is well-conditioned.
    """
    vecs = np.asarray(vecs, dtype=np.float64)
    n = vecs.shape[0]
    sim = vecs @ vecs.T
    if quality is not None:
        q = np.asarray(quality, dtype=np.float64).reshape(-1)
        sim = (q[:, None] * q[None, :]) * sim
    sim = sim + jitter * np.eye(n)
    return sim


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def greedy_map_dpp(kernel: np.ndarray, k: int, epsilon: float = 1e-10) -> List[int]:
    """Fast greedy MAP inference for a DPP (Chen et al., 2018).

    Returns up to ``k`` item indices, greedily maximizing the marginal gain in
    log-determinant.

    Variable names follow the paper's notation (kept terse so the algorithm
    lines up with it):

    * ``di2s``   — d_i², the squared marginal gain of adding each item next.
    * ``cis``    — the incremental Cholesky factors built row by row.
    * ``di_opt`` / ``ci_opt`` — the chosen item's d and its Cholesky column.
    * ``eis``    — the new Cholesky row contributed by the chosen item.
    """
    kernel = np.asarray(kernel, dtype=np.float64)
    n = kernel.shape[0]
    k = min(k, n)
    if k <= 0:
        return []
    cis = np.zeros((k, n))
    di2s = np.diag(kernel).astype(np.float64).copy()
    selected: List[int] = []
    j = int(np.argmax(di2s))
    selected.append(j)
    while len(selected) < k:
        t = len(selected) - 1
        ci_opt = cis[:t, j]
        di_opt = float(np.sqrt(max(di2s[j], 1e-300)))
        row = kernel[j, :]
        eis = (row - ci_opt @ cis[:t, :]) / di_opt
        cis[t, :] = eis
        di2s = di2s - np.square(eis)
        di2s[j] = -np.inf
        j = int(np.argmax(di2s))
        if di2s[j] < epsilon:
            break
        selected.append(j)
    return selected


def farthest_point_sampling(
    vecs: np.ndarray,
    k: int,
    start: int = 0,
    seeds: Optional[Sequence[int]] = None,
) -> List[int]:
    """Max-min cosine-distance greedy selection (also the DPP fallback).

    Returns up to ``k`` indices into ``vecs`` that are mutually far apart, each
    chosen to maximize its minimum cosine distance to everything picked so far.

    ``seeds`` pre-selects indices (kept, order preserved) and seeds the frontier
    — use it to extend an existing selection. When ``seeds`` is None the walk
    starts from ``start``. The returned list always begins with the seeds.
    """
    vecs = np.asarray(vecs, dtype=np.float64)
    n = vecs.shape[0]
    k = min(k, n)
    if k <= 0:
        return []
    dist = 1.0 - vecs @ vecs.T
    if seeds:
        selected = [int(s) for s in seeds]
        # min cosine distance from each row to its nearest already-selected seed
        min_d = dist[:, selected].min(axis=1)
    else:
        selected = [int(start)]
        min_d = dist[start].copy()
    while len(selected) < k:
        min_d[selected] = -np.inf
        j = int(np.argmax(min_d))
        selected.append(j)
        min_d = np.minimum(min_d, dist[j])
    return selected


def select_diverse(
    vecs: np.ndarray,
    k: int,
    quality: Optional[np.ndarray] = None,
    seed: int = 0,
) -> List[int]:
    """Pick ``k`` diverse item indices. Greedy DPP, farthest-point fallback."""
    vecs = np.asarray(vecs, dtype=np.float64)
    n = vecs.shape[0]
    if n == 0:
        return []
    if k >= n:
        return list(range(n))
    try:
        kernel = build_kernel(vecs, quality=quality)
        sel = greedy_map_dpp(kernel, k)
        if len(sel) >= min(k, n):
            return sel
        # top up with farthest-point if greedy stopped early (rank-deficient)
        fallback = farthest_point_sampling(vecs, k)
        for idx in fallback:
            if idx not in sel:
                sel.append(idx)
            if len(sel) >= k:
                break
        return sel[:k]
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        # Expected ways the kernel can degenerate (singular/ill-conditioned,
        # ragged input). Fall back to farthest-point, but re-raise under
        # CREATIVITY_DEBUG so a genuine bug isn't masked by the fallback.
        if os.environ.get("CREATIVITY_DEBUG"):
            raise
        return farthest_point_sampling(vecs, k)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def pairwise_cosine_sims(vecs: np.ndarray) -> np.ndarray:
    """Flat array of cosine similarities over all unordered pairs ``(i < j)``.

    Empty when there are fewer than two rows. Assumes L2-normalized rows, so a
    dot product is the cosine. Shared by ``mean_pairwise_distance`` here and the
    monitor's ``mean_pairwise_cosine`` so the two can never drift apart.
    """
    vecs = np.asarray(vecs, dtype=np.float64)
    n = vecs.shape[0]
    if n < 2:
        return np.zeros((0,), dtype=np.float64)
    sims = vecs @ vecs.T
    return sims[np.triu_indices(n, k=1)]


def mean_pairwise_distance(vecs: np.ndarray) -> float:
    """Average cosine distance over all unordered pairs (higher == more diverse)."""
    pairs = pairwise_cosine_sims(vecs)
    if pairs.size == 0:
        return 0.0
    return float(np.mean(1.0 - pairs))


def vendi_score(vecs: np.ndarray) -> float:
    """Vendi score: effective number of distinct items (exp of von-Neumann
    entropy of the normalized similarity matrix). 1 == all identical, up to n."""
    vecs = np.asarray(vecs, dtype=np.float64)
    n = vecs.shape[0]
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    K = (vecs @ vecs.T) / n  # eigenvalues >= 0, sum to 1
    w = np.linalg.eigvalsh(K)
    w = w[w > 1e-12]
    if w.size == 0:
        return 1.0
    w = w / w.sum()
    entropy = -np.sum(w * np.log(w))
    return float(np.exp(entropy))
