"""Geometric novelty: k-NN mean distance in embedding space.

Novelty is **decoupled from the judge** — it is a pure property of where a point
sits relative to others. A point far from its neighbors is novel; a point in a
crowd is not. This is the only thing that owns "is this new?".

Scope, stated honestly: ``novelty`` = mean k-NN distance to this session's own
elites + batch; a **variety proxy**, NOT originality vs. prior art. The reference
set is only the points generated so far — there is no external/world referent
here, so a high ``novelty`` means "unlike the other ideas in this run", not "novel
to the world".
"""

from __future__ import annotations

import numpy as np

# Cosine distance (1 - cos) maxes out at 2.0 for antipodal vectors; novelty is
# clipped to this so a degenerate distance can't blow past the natural ceiling.
MAX_COSINE_DISTANCE = 2.0


def cosine_distance_matrix(vecs: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """``(n, m)`` cosine distances (``1 - cos``) between rows of the two sets.

    Both are assumed L2-normalized, so cosine similarity is a dot product.
    """
    vecs = np.asarray(vecs, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    sims = vecs @ reference.T
    return 1.0 - sims


def mean_knn_distance(dist: np.ndarray, k: int, n_neighbors: int) -> np.ndarray:
    """Mean of the ``k`` smallest distances per row of a ``(n, m)`` matrix.

    The shared kernel behind both :func:`knn_novelty` and the pipeline's
    survivor-novelty pass. Callers mask any row's own neighbour to ``inf`` before
    calling and pass ``n_neighbors`` = the count of valid (non-masked) columns.
    Returns 1.0 (maximally novel) per row when there are no neighbours.
    """
    n = dist.shape[0]
    if n_neighbors <= 0:
        return np.ones((n,), dtype=np.float32)
    kk = min(k, n_neighbors)
    part = np.partition(dist, kk - 1, axis=1)[:, :kk]
    return np.clip(part.mean(axis=1), 0.0, MAX_COSINE_DISTANCE).astype(np.float32)


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
    return mean_knn_distance(dist, k, n_neighbors=m_eff)
