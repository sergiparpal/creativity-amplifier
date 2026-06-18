"""Pipeline robustness: engine-knob pinning, null-engine guard, empty-cycle schema."""

from __future__ import annotations

from creativity_engine import config, pipeline, selftest
from creativity_engine.state import State


def _generic():
    return config.load_generic_axes().to_dict()


def test_open_niching_knobs_pinned_to_init_snapshot(home):
    # The open-axis NICHING knobs (open_niches / open_niche_freeze_factor) are
    # pinned at init like the axes geometry: a later cycle passing a DIFFERENT
    # value must be ignored, so the CVT partition can't be refit with a different k
    # than the cells already in the archive. (Other engine knobs stay per-cycle
    # overridable — see test_state_pruning.)
    axes = _generic()
    axes["engine"] = {"open_niches": 10}
    pipeline.init_project("pin", axes, seed=0, home=home)
    target = int(State("pin", home=home).read_meta()["candidates_per_generation"])

    axes2 = _generic()
    axes2["engine"] = {"open_niches": 99}  # operator edits the file mid-session
    pipeline.ingest("pin", selftest.diverse_candidates(target), axes2, seed=0, home=home)

    assert State("pin", home=home).read_meta()["engine"]["open_niches"] == 10


def test_metrics_survives_null_engine_block(home):
    # An older/hand-edited meta may carry "engine": null; metrics must not crash.
    pipeline.init_project("m", _generic(), seed=0, home=home)
    st = State("m", home=home)
    meta = st.read_meta()
    meta["engine"] = None
    st.write_meta(meta)
    res = pipeline.metrics("m", home=home)  # would AttributeError on None.get(...)
    assert "open_axis" in res


def test_empty_cycle_returns_full_schema(home):
    # An empty-candidate generation must return the same response shape as a normal
    # one, so consumers never KeyError on the advisory keys.
    pipeline.init_project("e", _generic(), seed=0, home=home)
    res = pipeline.ingest("e", [], _generic(), seed=0, home=home)
    for key in ("slate", "ask_pairs", "ask_policy", "monitor", "parents", "open_axis"):
        assert key in res, key
    assert res["ask_policy"]["phase"] in ("explore", "refine")
    assert res["monitor"]["under_generation"] is False
    assert res["monitor"]["variety_eroding"] is False
    assert res["monitor"]["collapsing"] is False
