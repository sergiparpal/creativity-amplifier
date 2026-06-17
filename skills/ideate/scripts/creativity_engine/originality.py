"""Advisory originality measurement: distance from an "obvious-set".

This module is **measurement only**. Unlike geometric novelty (which has no
external referent — see :mod:`novelty`), originality here is scored against a set
of *obvious* reference vectors (clichés / would-be prior-art anchors): how far an
idea sits from the nearest obvious one.

Hard boundary — load-bearing: originality is **never** wired into the selection
geometry (the DPP ``q`` / kernel in :mod:`diversity`), **never** gates the
self-test (it is excluded from ``ok``), and **never** steers the engine. The skill
uses the same idea at *generation* time (repel clichés); the engine only ever
*measures* it. Keeping it out of selection is what stops a cliché signal from
pruning variety.

Vectors are assumed **L2-normalized** (cosine similarity == dot product),
consistent with every embedder in :mod:`embed`.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def originality_scores(idea_vecs, obvious_vecs) -> Dict[str, Any]:
    """Distance-to-obvious for each idea vector.

    For each idea, ``originality = 1 - max cosine similarity to any vector in
    ``obvious_vecs``. Higher == further from the obvious set. Both inputs are
    assumed L2-normalized so cosine is a plain dot product.

    Returns ``{"per_idea": [...], "slate_mean": float}``. Degenerate inputs are
    handled gracefully (no exception, no NaN): an empty ``obvious_vecs`` (no
    referent to be far from) yields ``0.0`` for every idea and a ``0.0`` mean; an
    empty ``idea_vecs`` yields an empty ``per_idea`` and a ``0.0`` mean.
    """
    ideas = np.asarray(idea_vecs, dtype=np.float32)
    obvious = np.asarray(obvious_vecs, dtype=np.float32)
    n = ideas.shape[0] if ideas.ndim == 2 else 0
    # No ideas, or no obvious referent (or a malformed referent): zero result.
    if n == 0 or obvious.ndim != 2 or obvious.shape[0] == 0:
        return {"per_idea": [0.0] * n, "slate_mean": 0.0}
    sims = ideas @ obvious.T  # (n, m) cosine similarities (rows L2-normalized)
    max_sims = sims.max(axis=1)
    per_idea = 1.0 - max_sims
    return {
        "per_idea": [round(float(x), 4) for x in per_idea],
        "slate_mean": round(float(per_idea.mean()), 4),
    }
