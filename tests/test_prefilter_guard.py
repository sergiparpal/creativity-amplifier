"""Prefilter guard: a soft signal when the agent submits far fewer candidates to
``ingest`` than the per-generation target — the one stage the monitor can't see
(the agent over-prefiltering and cutting variety under cover of "off-brief").

The guard is advisory only: it must never flip ``collapsing`` or train/suppress
the monitor's calibration window.
"""

from __future__ import annotations

from cambrian_engine import config, pipeline, selftest
from cambrian_engine.state import State


def _axes():
    return config.load_generic_axes().to_dict()


def _target(home) -> int:
    return int(State("p", home=home).read_meta()["candidates_per_generation"])


def test_under_generation_flag_set_when_submitting_few(home):
    pipeline.init_project("p", _axes(), seed=0, home=home)
    target = _target(home)  # generic default = 12
    # Submit well below 0.6 * target.
    few = max(1, int(0.6 * target) - 2)
    res = pipeline.ingest("p", selftest.diverse_candidates(few), _axes(), seed=0, home=home)
    mon = res["monitor"]
    assert mon["submitted"] == few
    assert mon["target_candidates"] == target
    assert mon["under_generation"] is True
    assert "under_generation_note" in mon


def test_under_generation_flag_clear_at_target(home):
    pipeline.init_project("p", _axes(), seed=0, home=home)
    target = _target(home)
    res = pipeline.ingest("p", selftest.diverse_candidates(target), _axes(), seed=0, home=home)
    mon = res["monitor"]
    assert mon["submitted"] == target
    assert mon["under_generation"] is False
    assert "under_generation_note" not in mon


def test_guard_does_not_affect_collapsing(home):
    # A small but DIVERSE batch is under target, yet must not be reported as
    # collapsing — the guard is orthogonal to the collapse signals.
    pipeline.init_project("p", _axes(), seed=0, home=home)
    res = pipeline.ingest("p", selftest.diverse_candidates(4), _axes(), seed=0, home=home)
    mon = res["monitor"]
    assert mon["under_generation"] is True
    assert mon["collapsing"] is False


def test_guard_does_not_train_or_suppress_calibration_window(home):
    # The under-generation flag must not touch the rolling cos_window: a healthy
    # (not too_similar) under-target generation still trains it like any other.
    pipeline.init_project("p", _axes(), seed=0, home=home)
    pipeline.ingest("p", selftest.diverse_candidates(4, gen=0), _axes(), seed=0, home=home)
    window = State("p", home=home).read_meta().get("cos_window", [])
    # A diverse generation of >=2 candidates rolls exactly one value into the window.
    assert len(window) == 1


def test_under_generation_ratio_is_configurable(home):
    # Lowering the ratio makes the same submission count pass the guard.
    axes = _axes()
    axes["engine"] = {"under_generation_ratio": 0.1}
    pipeline.init_project("p", axes, seed=0, home=home)
    target = _target(home)
    few = max(1, int(0.6 * target) - 2)  # would trip the default 0.6, but not 0.1
    res = pipeline.ingest("p", selftest.diverse_candidates(few), axes, seed=0, home=home)
    assert res["monitor"]["under_generation"] is False
