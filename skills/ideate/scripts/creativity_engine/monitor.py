"""Anti-collapse monitor.

Two complementary signals decide whether the search is converging:

* **Shannon entropy over niche occupancy** — are ideas spreading across many
  niches, or piling into a few? Low normalized entropy == collapse.
* **Mean pairwise cosine** of the current generation — are the raw candidates
  getting samey? High mean cosine == collapse.

The monitor only *reports*; the skill reacts by raising diversity pressure. This
machinery is never removed or bypassed — it is the whole point.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

# Defaults chosen so a near-duplicate stream trips the flag and a diverse one
# does not. Tunable per call.
DEFAULT_COS_THRESHOLD = 0.55
DEFAULT_ENTROPY_THRESHOLD = 0.50


def shannon_entropy(counts: Sequence[float]) -> float:
    """Shannon entropy (nats) of a count distribution."""
    arr = np.asarray([c for c in counts if c > 0], dtype=np.float64)
    if arr.size == 0:
        return 0.0
    p = arr / arr.sum()
    return float(-np.sum(p * np.log(p)))


def normalized_entropy(counts: Sequence[float]) -> float:
    """Entropy normalized to [0, 1] by the max possible (log of #occupied)."""
    nonzero = [c for c in counts if c > 0]
    if len(nonzero) <= 1:
        return 0.0
    h = shannon_entropy(nonzero)
    return float(h / np.log(len(nonzero)))


def mean_pairwise_cosine(vecs: np.ndarray) -> float:
    """Average cosine similarity over all unordered pairs."""
    vecs = np.asarray(vecs, dtype=np.float64)
    n = vecs.shape[0]
    if n < 2:
        return 0.0
    sims = vecs @ vecs.T
    iu = np.triu_indices(n, k=1)
    return float(np.mean(sims[iu]))


def evaluate(
    generation_vecs: np.ndarray,
    niche_counts: Sequence[float],
    cos_threshold: float = DEFAULT_COS_THRESHOLD,
    entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
) -> Dict[str, object]:
    """Compute monitor metrics and the ``collapsing`` flag.

    ``collapsing`` trips when the generation is too similar (mean cosine high) OR
    occupancy has concentrated (normalized entropy low while there are enough
    niches to spread across).
    """
    vecs = np.asarray(generation_vecs, dtype=np.float64)
    n = vecs.shape[0]
    mean_cos = mean_pairwise_cosine(vecs) if n >= 2 else 0.0
    norm_ent = normalized_entropy(niche_counts)
    occupied = int(sum(1 for c in niche_counts if c > 0))

    reasons: List[str] = []
    too_similar = n >= 2 and mean_cos > cos_threshold
    # Only treat low entropy as collapse once there's something to spread over.
    too_concentrated = occupied >= 3 and norm_ent < entropy_threshold
    if too_similar:
        reasons.append(
            f"mean pairwise cosine {mean_cos:.2f} > {cos_threshold:.2f}"
        )
    if too_concentrated:
        reasons.append(
            f"normalized niche entropy {norm_ent:.2f} < {entropy_threshold:.2f}"
        )

    return {
        "collapsing": bool(too_similar or too_concentrated),
        "mean_cosine": round(mean_cos, 4),
        "entropy": round(shannon_entropy(niche_counts), 4),
        "normalized_entropy": round(norm_ent, 4),
        "coverage": occupied,
        "n": n,
        "reasons": reasons,
    }
