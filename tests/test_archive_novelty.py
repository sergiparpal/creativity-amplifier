"""Phase 3: geometric novelty + MAP-Elites placement invariants.

Domain-neutral fixtures only. Asserts novelty monotonicity, one-elite-per-niche,
placement across categorical/continuous/open axes, and that no judge lives in
these modules (diversity is decoupled from the judge by construction)."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from creativity_engine import archive as archive_mod
from creativity_engine import novelty
from creativity_engine.archive import Archive, CVTNicher, compute_niche, continuous_bin
from creativity_engine.config import axes_spec_from_dict


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


# --------------------------------------------------------------------------- #
# Novelty
# --------------------------------------------------------------------------- #
def test_novelty_higher_for_far_point():
    # reference clustered around axis e0
    rng = np.random.default_rng(0)
    base = np.zeros(8)
    base[0] = 1.0
    ref = np.array([_unit(base + 0.02 * rng.standard_normal(8)) for _ in range(10)])

    near = _unit(base + 0.02 * rng.standard_normal(8))  # inside the cluster
    far = _unit(np.eye(8)[3])                            # orthogonal direction

    nov = novelty.knn_novelty(np.array([near, far]), ref, k=3)
    assert nov[1] > nov[0]
    assert nov[1] > 0.5  # far point is clearly novel


def test_novelty_empty_reference_is_max():
    nov = novelty.knn_novelty(np.array([_unit([1, 0, 0])]), np.zeros((0, 3)), k=3)
    assert nov[0] == 1.0


def test_novelty_exclude_self():
    pts = np.array([_unit(v) for v in np.eye(5)])
    nov = novelty.knn_novelty(pts, pts, k=2, exclude_self=True)
    # all orthogonal => each point's neighbours are at distance 1
    assert np.allclose(nov, 1.0, atol=1e-5)


# --------------------------------------------------------------------------- #
# Niching / placement across axis types
# --------------------------------------------------------------------------- #
SPEC = axes_spec_from_dict(
    {
        "domain": "t",
        "axes": [
            {"name": "audience", "type": "categorical"},
            {"name": "edginess", "type": "continuous", "range": [0, 1]},
            {"name": "mechanism", "type": "open", "primary_novelty": True},
        ],
    }
)


def test_continuous_bin_clamps():
    edg = SPEC.axis("edginess")
    assert continuous_bin(edg, 0.0) == 0
    assert continuous_bin(edg, 0.7) == 3
    assert continuous_bin(edg, 1.0) == 4  # clamped to bins-1
    assert continuous_bin(edg, 5.0) == 4
    assert continuous_bin(edg, -1.0) == 0


def test_compute_niche_all_axis_types():
    desc = {"audience": "Young Adults", "edginess": 0.7, "mechanism": "x"}
    nid, coords = compute_niche(desc, SPEC, {"mechanism": 7})
    assert coords["audience"] == "young-adults"
    assert coords["edginess"] == "b3"
    assert coords["mechanism"] == "cell7"
    assert nid == "audience=young-adults|edginess=b3|mechanism=cell7"


def test_distinct_descriptors_distinct_niches():
    a, _ = compute_niche({"audience": "kids", "edginess": 0.1, "mechanism": "m"}, SPEC, {"mechanism": 1})
    b, _ = compute_niche({"audience": "kids", "edginess": 0.9, "mechanism": "m"}, SPEC, {"mechanism": 1})
    c, _ = compute_niche({"audience": "kids", "edginess": 0.1, "mechanism": "m"}, SPEC, {"mechanism": 4})
    assert a != b  # different continuous bucket
    assert a != c  # different open cell
    # same descriptor + same cell => same niche (stable)
    a2, _ = compute_niche({"audience": "kids", "edginess": 0.1, "mechanism": "m"}, SPEC, {"mechanism": 1})
    assert a == a2


def test_cvt_nicher_deterministic_and_stable():
    n1 = CVTNicher(dim=16, k=8, seed=3)
    n2 = CVTNicher(dim=16, k=8, seed=3)
    assert np.array_equal(n1.centroids, n2.centroids)
    rng = np.random.default_rng(1)
    vecs = np.array([_unit(rng.standard_normal(16)) for _ in range(20)])
    assert n1.cells(vecs) == n2.cells(vecs)
    # each assignment is the argmax-cosine centroid
    for i, c in enumerate(n1.cells(vecs)):
        sims = n1.centroids @ vecs[i]
        assert c == int(np.argmax(sims))


# --------------------------------------------------------------------------- #
# One elite per niche
# --------------------------------------------------------------------------- #
def test_one_elite_per_niche_fitness_then_novelty():
    arc = Archive(SPEC)
    assert arc.place("c1", "n1", {}, fitness=0.5, novelty=0.2) is True
    # lower fitness does not replace
    assert arc.place("c2", "n1", {}, fitness=0.4, novelty=0.9) is False
    assert arc.niches["n1"].elite_id == "c1"
    # tie fitness, higher novelty wins
    assert arc.place("c3", "n1", {}, fitness=0.5, novelty=0.5) is True
    assert arc.niches["n1"].elite_id == "c3"
    # a new niche
    assert arc.place("c4", "n2", {}, fitness=0.1, novelty=0.1) is True
    assert len(arc) == 2
    assert set(arc.elite_ids()) == {"c3", "c4"}
    # occupancy counts: n1 saw 3 placements, n2 saw 1
    assert sorted(arc.niche_counts(), reverse=True) == [3, 1]


def test_archive_round_trips_through_dict():
    arc = Archive(SPEC)
    arc.place("c1", "n1", {"audience": "kids"}, fitness=0.7, novelty=0.3)
    again = Archive.from_dict(SPEC, arc.to_dict())
    assert again.elite_ids() == ["c1"]
    assert again.niches["n1"].fitness == 0.7
    assert again.counts == arc.counts


# --------------------------------------------------------------------------- #
# Judge independence (by construction)
# --------------------------------------------------------------------------- #
def test_no_judge_module_in_engine():
    # Judging is done by the agent, never in Python. There is no judge module.
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("creativity_engine.judge")


def test_geometry_modules_expose_no_judge_symbol():
    for mod in (archive_mod, novelty):
        public = [n for n in dir(mod) if not n.startswith("_")]
        assert not any("judge" in n.lower() for n in public)
