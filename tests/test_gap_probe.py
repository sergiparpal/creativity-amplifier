"""Advisory surface/mechanism gap probe (CI check).

The probe is measurement only: the helper math is correct, the per-cycle emission is OFF by
default (and the off-path output is unchanged, ``ask_policy`` included) and emits + persists
when on, and it never enters the self-test's `ok` or any gate.
"""

from __future__ import annotations

import numpy as np

from cambrian_engine import config, gap, pipeline, selftest
from cambrian_engine.state import State


def _unit(rows):
    a = np.asarray(rows, dtype=np.float64)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_helper_zero_gap_on_identical_spaces():
    v = _unit([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    out = gap.surface_mechanism_gap(v, v)
    assert out["n"] == 3
    assert abs(out["gap"]) < 1e-9
    assert out["surface_spread"] == out["mechanism_spread"]


def test_helper_monotone_mechanism_has_larger_gap():
    # Same surface; mechanism collapsed to one point -> mechanism_spread ~ 0 -> bigger gap.
    surf = _unit([[1, 0], [0, 1], [0.7, 0.7], [-1, 0]])
    mech_varied = surf.copy()
    mech_mono = np.tile([1.0, 0.0], (4, 1))
    g_varied = gap.surface_mechanism_gap(surf, mech_varied)
    g_mono = gap.surface_mechanism_gap(surf, mech_mono)
    assert g_mono["mechanism_spread"] < g_varied["mechanism_spread"]
    assert g_mono["gap"] > g_varied["gap"]


def test_helper_graceful_on_too_few_or_misaligned():
    assert gap.surface_mechanism_gap(np.zeros((1, 3)), np.zeros((1, 3)))["corr"] is None
    assert gap.surface_mechanism_gap(np.zeros((3, 3)), np.zeros((2, 3)))["n"] == 2
    assert gap.surface_mechanism_gap(np.zeros((3, 3)), np.zeros((2, 3)))["gap"] == 0.0


def _axes(engine=None):
    a = config.load_generic_axes().to_dict()
    if engine:
        a["engine"] = engine
    return a


def test_gap_probe_off_by_default_is_zero_cost(home):
    axes = _axes()
    pipeline.init_project("p", axes, seed=0, home=home)
    target = int(State("p", home=home).read_meta()["candidates_per_generation"])
    res = pipeline.ingest("p", selftest.diverse_candidates(target), axes, seed=0, home=home)
    assert "surface_mechanism_gap" not in res          # off -> output unchanged
    assert "ask_policy" in res                          # S3 key must survive the off-path
    assert "gap_log" not in State("p", home=home).read_meta()


def test_gap_probe_emits_and_persists_when_on(home):
    axes = _axes({"gap_probe": True})
    pipeline.init_project("p", axes, seed=0, home=home)
    target = int(State("p", home=home).read_meta()["candidates_per_generation"])
    res = pipeline.ingest("p", selftest.diverse_candidates(target), axes, seed=0, home=home)
    g = res["surface_mechanism_gap"]
    assert {"surface_spread", "mechanism_spread", "gap"} <= set(g)
    assert res["monitor"]["collapsing"] is False        # advisory, never drives collapse
    assert "ask_policy" in res                           # other keys unaffected
    log = State("p", home=home).read_meta().get("gap_log", [])
    assert len(log) == 1 and "gap" in log[0]


def test_metrics_surfaces_gap_log_only_when_present(home):
    off = _axes()
    pipeline.init_project("off", off, seed=0, home=home)
    t = int(State("off", home=home).read_meta()["candidates_per_generation"])
    pipeline.ingest("off", selftest.diverse_candidates(t), off, seed=0, home=home)
    assert "gap_log" not in pipeline.metrics("off", home=home)   # off -> not surfaced

    on = _axes({"gap_probe": True})
    pipeline.init_project("on", on, seed=0, home=home)
    pipeline.ingest("on", selftest.diverse_candidates(t), on, seed=0, home=home)
    m = pipeline.metrics("on", home=home)
    assert m["gap_log"] and "gap" in m["gap_log"][0]             # on -> the persisted series


def test_selftest_reports_gap_probe_and_never_gates_ok():
    rep = selftest.run()  # hermetic temp home, hash embedder
    assert "gap_probe" in rep
    gp = rep["gap_probe"]
    if gp.get("ran"):
        assert "gap" in gp["diverse_slate"]
        assert gp["sanity_monotone_gap_larger"] is True
    assert rep["ok"] is True                             # probe never affects ok
    assert "gap_probe" not in rep["variety_gate"]["checks"]
