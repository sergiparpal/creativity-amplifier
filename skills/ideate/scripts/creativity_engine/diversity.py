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

from typing import List, Optional

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
    vecs: np.ndarray, k: int, start: int = 0
) -> List[int]:
    """Max-min cosine-distance greedy selection (DPP fallback)."""
    vecs = np.asarray(vecs, dtype=np.float64)
    n = vecs.shape[0]
    k = min(k, n)
    if k <= 0:
        return []
    sims = vecs @ vecs.T
    dist = 1.0 - sims
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
    except Exception:
        return farthest_point_sampling(vecs, k)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def mean_pairwise_distance(vecs: np.ndarray) -> float:
    """Average cosine distance over all unordered pairs (higher == more diverse)."""
    vecs = np.asarray(vecs, dtype=np.float64)
    n = vecs.shape[0]
    if n < 2:
        return 0.0
    sims = vecs @ vecs.T
    iu = np.triu_indices(n, k=1)
    return float(np.mean(1.0 - sims[iu]))


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
