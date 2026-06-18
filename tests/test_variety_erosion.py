"""S2 — variety-erosion sensor (CI check).

Two halves:
* the pure assessor (`monitor.assess_variety_erosion`) — accelerating decay fires after
  K; a long, HEALTHY decelerating decay never false-positives; an unhealthy submit count
  suppresses it;
* the wiring — the flag is present, advisory (never drives `collapsing`), and the sensor
  keeps its OWN window so the monitor's calibration window (`cos_window`) is untouched.
"""

from __future__ import annotations

from creativity_engine import config, monitor, pipeline, selftest
from creativity_engine.state import State

W, RHO, K = 5, 0.5, 2  # the fixed defaults


def _feed(series, *, submitted_healthy=True):
    win, streak, flags = [], 0, []
    for v in series:
        out = monitor.assess_variety_erosion(
            win, streak, v, submitted_healthy,
            window=W, accel_ratio=RHO, persist=K,
        )
        win, streak = out["novelty_window"], out["erosion_streak"]
        flags.append(out["variety_eroding"])
    return flags


def test_accelerating_decay_fires_after_persist():
    # novelty falling faster and faster: a generator regressing to the mode
    series = [0.80, 0.62, 0.50, 0.43, 0.38, 0.30, 0.20, 0.08]
    flags = _feed(series)
    assert any(flags)            # it fires
    assert flags[-1] is True     # still firing at the end
    assert not any(flags[:W])    # never before a full window exists


def test_decelerating_decay_never_false_positives_on_long_healthy_session():
    # natural archive-fill decay: monotone but DECELERATING -> must never flag
    series = [0.80, 0.62, 0.50, 0.43, 0.40, 0.385, 0.378, 0.375, 0.374, 0.373]
    assert not any(_feed(series))


def test_unhealthy_submit_count_suppresses_flag():
    series = [0.80, 0.62, 0.50, 0.43, 0.38, 0.30, 0.20, 0.08]
    assert not any(_feed(series, submitted_healthy=False))


def _axes():
    return config.load_generic_axes().to_dict()


def test_flag_present_and_advisory_on_healthy_run(home):
    pipeline.init_project("p", _axes(), seed=0, home=home)
    target = int(State("p", home=home).read_meta()["candidates_per_generation"])
    res = pipeline.ingest("p", selftest.diverse_candidates(target), _axes(), seed=0, home=home)
    mon = res["monitor"]
    assert "variety_eroding" in mon          # wired
    assert mon["variety_eroding"] is False    # one healthy generation cannot erode
    assert mon["collapsing"] is False         # erosion never drives collapse


def test_sensor_keeps_its_own_window_and_leaves_calibration_untouched(home):
    # The sensor maintains meta["novelty_window"]; the monitor's cos_window
    # (calibration / boiling-frog guard) must roll exactly one value per healthy
    # generation, independent of erosion state.
    pipeline.init_project("p", _axes(), seed=0, home=home)
    pipeline.ingest("p", selftest.diverse_candidates(12, gen=0), _axes(), seed=0, home=home)
    meta = State("p", home=home).read_meta()
    assert len(meta.get("cos_window", [])) == 1   # calibration untouched
    assert "novelty_window" in meta                # erosion has its own series
