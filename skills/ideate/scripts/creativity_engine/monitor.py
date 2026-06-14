"""Anti-collapse monitor.

Two complementary signals decide whether the search is converging:

* **Shannon entropy over niche occupancy** — are ideas spreading across many
  niches, or piling into a few? Low normalized entropy == collapse.
* **Mean pairwise cosine** of the current generation — are the raw candidates
  getting samey? The similarity signal is **calibrated to the project**: a
  rolling baseline of recent generations' mean cosine is kept, and a generation
  trips the flag when it is meaningfully *more* similar than that baseline
  (``baseline + margin``) or breaches an absolute safety ceiling. Before a
  baseline exists it falls back to a fixed absolute threshold. This keeps the
  monitor from misfiring when the embedder or domain shifts the natural scale of
  cosine similarity.

The monitor only *reports*; the skill reacts by raising diversity pressure. This
machinery is never removed or bypassed — it is the whole point.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .diversity import pairwise_cosine_sims

# Absolute fallback (used until a rolling baseline has enough samples) and the
# normalized-entropy collapse threshold. Tunable per call.
DEFAULT_COS_THRESHOLD = 0.55
DEFAULT_ENTROPY_THRESHOLD = 0.50
# Calibration: a generation is "too similar" when its mean cosine exceeds the
# rolling baseline by more than ``MARGIN`` or crosses the absolute ``CEILING``.
DEFAULT_MARGIN = 0.15
DEFAULT_COS_CEILING = 0.80
# How many prior generations must be in the baseline before the relative rule is
# trusted; below this we use the absolute threshold.
DEFAULT_MIN_BASELINE = 2


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
    pairs = pairwise_cosine_sims(vecs)
    if pairs.size == 0:
        return 0.0
    return float(np.mean(pairs))


def _similarity_limit(
    baseline: Optional[Sequence[float]],
    cos_threshold: float,
    margin: float,
    cos_ceiling: float,
    min_baseline: int,
) -> float:
    """The effective cosine ceiling above which a generation is "too similar".

    With enough baseline samples the limit is calibrated to the project
    (``min(baseline_mean + margin, cos_ceiling)``); otherwise the fixed absolute
    threshold is used.
    """
    vals = [float(b) for b in (baseline or []) if b is not None]
    if len(vals) >= min_baseline:
        return min(float(np.mean(vals)) + margin, cos_ceiling)
    return cos_threshold


def evaluate(
    generation_vecs: np.ndarray,
    niche_counts: Sequence[float],
    cos_threshold: float = DEFAULT_COS_THRESHOLD,
    entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
    baseline: Optional[Sequence[float]] = None,
    margin: float = DEFAULT_MARGIN,
    cos_ceiling: float = DEFAULT_COS_CEILING,
    min_baseline: int = DEFAULT_MIN_BASELINE,
) -> Dict[str, object]:
    """Compute monitor metrics and the ``collapsing`` flag.

    ``collapsing`` trips when the generation is too similar (mean cosine above the
    calibrated limit — see :func:`_similarity_limit`) OR occupancy has
    concentrated (normalized entropy low while there are enough niches to spread
    across). ``baseline`` is the rolling window of recent generations' mean
    cosine; pass it to enable the relative rule.
    """
    vecs = np.asarray(generation_vecs, dtype=np.float64)
    n = vecs.shape[0]
    mean_cos = mean_pairwise_cosine(vecs) if n >= 2 else 0.0
    norm_ent = normalized_entropy(niche_counts)
    occupied = int(sum(1 for c in niche_counts if c > 0))

    cos_limit = _similarity_limit(
        baseline, cos_threshold, margin, cos_ceiling, min_baseline
    )
    base_n = len([b for b in (baseline or []) if b is not None])
    calibrated = base_n >= min_baseline

    reasons: List[str] = []
    too_similar = n >= 2 and mean_cos > cos_limit
    # Only treat low entropy as collapse once there's something to spread over.
    too_concentrated = occupied >= 3 and norm_ent < entropy_threshold
    if too_similar:
        how = (
            f"baseline {float(np.mean([b for b in baseline if b is not None])):.2f} + "
            f"margin {margin:.2f}" if calibrated else "absolute threshold"
        )
        reasons.append(
            f"mean pairwise cosine {mean_cos:.2f} > {cos_limit:.2f} ({how})"
        )
    if too_concentrated:
        reasons.append(
            f"normalized niche entropy {norm_ent:.2f} < {entropy_threshold:.2f}"
        )

    return {
        "collapsing": bool(too_similar or too_concentrated),
        # ``too_similar`` is the similarity signal alone (vs. the combined flag),
        # and ``calibrated`` says whether it used the relative rule or the absolute
        # fallback. Callers use the pair to decide whether a generation may train
        # the calibration baseline (only healthy, relatively-judged ones may).
        "too_similar": bool(too_similar),
        "calibrated": bool(calibrated),
        "mean_cosine": round(mean_cos, 4),
        "cos_limit": round(cos_limit, 4),
        "baseline_n": base_n,
        "entropy": round(shannon_entropy(niche_counts), 4),
        "normalized_entropy": round(norm_ent, 4),
        "coverage": occupied,
        "n": n,
        "reasons": reasons,
    }
