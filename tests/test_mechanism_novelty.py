"""S4 — mechanism-space novelty (CI check).

Mechanism novelty reuses the surface k-NN kernel on mechanism embeddings; it is persisted in a
parallel store, surfaced on the slate / ingest result / metrics, and is ADVISORY — it never
enters dedup, the DPP slate, or the surface `novelty` values.
"""

from __future__ import annotations

import numpy as np

from creativity_engine import config, novelty, pipeline, selftest
from creativity_engine.state import State


def _unit(rows):
    a = np.asarray(rows, dtype=np.float64)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_kernel_is_shared_with_surface_novelty():
    # Same machinery as surface novelty: orthogonal mechanisms are maximally novel,
    # identical ones are not.
    vecs = _unit([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    far = novelty.knn_novelty(vecs, vecs, k=1, exclude_self=True)
    same = novelty.knn_novelty(_unit([[1, 0, 0], [1, 0, 0]]), _unit([[1, 0, 0], [1, 0, 0]]),
                               k=1, exclude_self=True)
    assert float(far.mean()) > float(same.mean())


def _axes():
    return config.load_generic_axes().to_dict()


def test_mechanism_embeddings_persisted_and_subset_of_elites(home):
    axes = _axes()
    pipeline.init_project("p", axes, seed=0, home=home)
    target = int(State("p", home=home).read_meta()["candidates_per_generation"])
    res = pipeline.ingest("p", selftest.diverse_candidates(target), axes, seed=0, home=home)
    mech = State("p", home=home).read_mech_embeddings()
    assert mech                                   # populated
    elite_ids = set(State("p", home=home).read_candidates().keys())
    assert set(mech).issubset(elite_ids)
    # slate items + result carry the advisory signal
    assert any(s.get("mechanism_novelty") is not None for s in res["slate"])
    assert res["slate_mechanism_novelty"] is not None
    # surface novelty still present and untouched
    assert all("novelty" in s for s in res["slate"])
    assert res["monitor"]["collapsing"] is False  # advisory, never drives collapse


def test_metrics_reports_mechanism_spread(home):
    axes = _axes()
    pipeline.init_project("p", axes, seed=0, home=home)
    target = int(State("p", home=home).read_meta()["candidates_per_generation"])
    pipeline.ingest("p", selftest.diverse_candidates(target, gen=0), axes, seed=0, home=home)
    pipeline.ingest("p", selftest.diverse_candidates(target, gen=1), axes, seed=0, home=home)
    m = pipeline.metrics("p", home=home)
    assert "mechanism_spread" in m and "mechanism_n" in m
    assert m["mechanism_spread"] is None or isinstance(m["mechanism_spread"], float)
    assert m["mechanism_n"] >= 2


def test_mechanism_novelty_never_gates_selftest():
    rep = selftest.run()  # hermetic temp home, hash embedder
    assert rep["ok"] is True
    # advisory: not a variety_gate check
    assert "mechanism_novelty" not in rep["variety_gate"]["checks"]
