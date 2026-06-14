"""MAP-Elites archive over the resolved axes.

Each candidate is placed into a **niche** — a discrete cell of descriptor space —
and the archive keeps at most **one elite per niche**. Niche keys combine a
bucket per axis:

* ``categorical`` → the value itself (slugified),
* ``continuous``  → the bin index over its ``range``,
* ``open``        → a CVT (Voronoi) cell over the embedding of the axis value.

The within-niche elite is the higher-``fitness`` candidate (fitness comes from
the judge, upstream); ties break toward higher geometric novelty. The judge is
never called here — geometry owns diversity, and quality only ranks *within* an
already-diverse niche.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import Axis, AxesSpec, Niche

_SAFE = re.compile(r"[^a-z0-9]+")


def _slug(value: Any) -> str:
    s = _SAFE.sub("-", str(value).strip().lower()).strip("-")
    return s or "none"


class CVTNicher:
    """Deterministic CVT tessellation of the unit sphere for an open axis.

    Centroids are fixed random unit directions seeded by ``seed`` (so cell ids
    are stable across cycles without persisting anything). Assignment is by
    maximum cosine similarity.
    """

    def __init__(self, dim: int, k: int = 16, seed: int = 0):
        self.dim = int(dim)
        self.k = int(k)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        c = rng.standard_normal((self.k, self.dim))
        norms = np.linalg.norm(c, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.centroids = (c / norms).astype(np.float32)

    def cell(self, vec: np.ndarray) -> int:
        vec = np.asarray(vec, dtype=np.float32)
        return int(np.argmax(self.centroids @ vec))

    def cells(self, vecs: np.ndarray) -> List[int]:
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.shape[0] == 0:
            return []
        sims = vecs @ self.centroids.T  # (n, k)
        return [int(i) for i in np.argmax(sims, axis=1)]


def continuous_bin(axis: Axis, value: Any) -> int:
    lo, hi = axis.range  # type: ignore[misc]
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0
    if hi <= lo:
        return 0
    frac = (v - lo) / (hi - lo)
    idx = int(frac * axis.bins)
    return max(0, min(axis.bins - 1, idx))


def axis_bucket(axis: Axis, value: Any, open_cell: Optional[int] = None) -> str:
    """Bucket label for one axis (used both for the niche key and display)."""
    if axis.type == "categorical":
        return _slug(value if value is not None else "none")
    if axis.type == "continuous":
        return f"b{continuous_bin(axis, value)}"
    # open
    if open_cell is None:
        return "cell?"
    return f"cell{open_cell}"


def compute_niche(
    descriptor: Dict[str, Any],
    spec: AxesSpec,
    open_cells: Optional[Dict[str, int]] = None,
) -> Tuple[str, Dict[str, str]]:
    """Return ``(niche_id, coords)`` for a candidate's descriptor."""
    open_cells = open_cells or {}
    coords: Dict[str, str] = {}
    parts: List[str] = []
    for axis in spec.axes:
        val = descriptor.get(axis.name)
        cell = open_cells.get(axis.name) if axis.type == "open" else None
        bucket = axis_bucket(axis, val, open_cell=cell)
        coords[axis.name] = bucket
        parts.append(f"{axis.name}={bucket}")
    return "|".join(parts), coords


class Archive:
    """One elite per niche, persisted as a plain dict."""

    def __init__(self, spec: AxesSpec):
        self.spec = spec
        self.niches: Dict[str, Niche] = {}
        self.counts: Dict[str, int] = {}

    @classmethod
    def from_dict(cls, spec: AxesSpec, data: Dict[str, Any]) -> "Archive":
        arc = cls(spec)
        for nid, rec in (data.get("niches", {}) or {}).items():
            arc.niches[nid] = Niche.from_dict(rec)
        arc.counts = dict(data.get("counts", {}) or {})
        return arc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "niches": {nid: n.to_dict() for nid, n in self.niches.items()},
            "counts": dict(self.counts),
        }

    def place(
        self,
        candidate_id: str,
        niche_id: str,
        coords: Dict[str, str],
        fitness: float,
        novelty: float,
    ) -> bool:
        """Insert a candidate; return True if it (newly) became the niche elite.

        Elite selection: higher fitness wins; ties break toward higher novelty.
        """
        self.counts[niche_id] = self.counts.get(niche_id, 0) + 1
        cur = self.niches.get(niche_id)
        if cur is None:
            self.niches[niche_id] = Niche(
                id=niche_id, coords=coords, elite_id=candidate_id,
                fitness=fitness, novelty=novelty,
            )
            return True
        better = (fitness > cur.fitness) or (
            fitness == cur.fitness and novelty > cur.novelty
        )
        if better:
            cur.elite_id = candidate_id
            cur.fitness = fitness
            cur.novelty = novelty
            cur.coords = coords
            return True
        return False

    def elite_ids(self) -> List[str]:
        return [n.elite_id for n in self.niches.values() if n.elite_id]

    def niche_counts(self) -> List[int]:
        return [self.counts.get(nid, 0) for nid in self.niches]

    def __len__(self) -> int:
        return len(self.niches)
