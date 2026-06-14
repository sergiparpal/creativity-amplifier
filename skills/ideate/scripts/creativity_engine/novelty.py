"""Geometric novelty: k-NN mean distance in embedding space.

Novelty is **decoupled from the judge** — it is a pure property of where a point
sits relative to others. A point far from its neighbors is novel; a point in a
crowd is not. This is the only thing that owns "is this new?".
"""

from __future__ import annotations

import numpy as np


def cosine_distance_matrix(vecs: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """``(n, m)`` cosine distances (``1 - cos``) between rows of the two sets.

    Both are assumed L2-normalized, so cosine similarity is a dot product.
    """
    vecs = np.asarray(vecs, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    sims = vecs @ reference.T
    return 1.0 - sims


def knn_novelty(
    vecs: np.ndarray,
    reference: np.ndarray,
    k: int = 5,
    exclude_self: bool = False,
) -> np.ndarray:
    """Mean cosine distance from each row of ``vecs`` to its ``k`` nearest
    neighbours in ``reference``.

    Higher == more novel. If ``reference`` is empty, everything is maximally
    novel (1.0). When ``reference is vecs`` (self-novelty), pass
    ``exclude_self=True`` so a point isn't its own neighbour.
    """
    vecs = np.asarray(vecs, dtype=np.float32)
    n = vecs.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    m = reference.shape[0]
    if m == 0:
        return np.ones((n,), dtype=np.float32)

    dist = cosine_distance_matrix(vecs, reference)  # (n, m)
    if exclude_self and m == n:
        # Assume row i of vecs corresponds to row i of reference.
        np.fill_diagonal(dist, np.inf)
        m_eff = m - 1
    else:
        m_eff = m
    if m_eff <= 0:
        return np.ones((n,), dtype=np.float32)

    kk = min(k, m_eff)
    # k smallest distances per row.
    part = np.partition(dist, kk - 1, axis=1)[:, :kk]
    out = part.mean(axis=1)
    out = np.clip(out, 0.0, 2.0)
    return out.astype(np.float32)
