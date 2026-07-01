"""User discards: the engine drops a discarded idea from future slates and from the
next-generation parents, persistently and namespaced by domain. A discard is a human
veto applied only to the presented/parent pool — never wired into novelty/DPP/monitor.
"""

from __future__ import annotations

from cambrian_engine import config, pipeline, selftest
from cambrian_engine.state import State


def _generic():
    return config.load_generic_axes().to_dict()


def _slate_ids(res):
    return [item["id"] for item in res["slate"]]


def _target(project, home):
    return int(State(project, home=home).read_meta()["candidates_per_generation"])


def test_discard_removes_elite_from_future_slate(home):
    # Two projects fed identical candidates + seed have identical archives, so the
    # ONLY difference in the gen-1 slate is the discard. (A slate is built only when
    # a generation has candidates, so we compare a real second generation.)
    pipeline.init_project("ctl", _generic(), seed=0, home=home)
    pipeline.init_project("trt", _generic(), seed=0, home=home)
    n = _target("ctl", home)

    for proj in ("ctl", "trt"):
        pipeline.ingest(proj, selftest.diverse_candidates(n, gen=0), _generic(),
                        seed=0, home=home)

    # Discard, in the treatment project only, an id that the control's gen-1 slate
    # actually surfaces — so the assertion can't false-pass on an id DPP never picks.
    ctl_gen1 = pipeline.ingest("ctl", selftest.diverse_candidates(n, gen=1),
                               _generic(), seed=0, home=home)
    victim = _slate_ids(ctl_gen1)[0]
    assert victim in _slate_ids(ctl_gen1)  # control: present

    pipeline.remember("trt", {"type": "discard", "id": victim}, home=home)
    trt_gen1 = pipeline.ingest("trt", selftest.diverse_candidates(n, gen=1),
                               _generic(), seed=0, home=home)
    assert victim not in _slate_ids(trt_gen1)  # treatment: vetoed away


def test_discard_excludes_from_parents_and_repin_restores(home):
    pipeline.init_project("p", _generic(), seed=0, home=home)
    n = _target("p", home)
    res = pipeline.ingest("p", selftest.diverse_candidates(n, gen=0), _generic(),
                          seed=0, home=home)
    victim = _slate_ids(res)[0]

    # k large enough to return every elite, so an elite absent from parents is a
    # deliberate exclusion, not just an unfilled slot.
    control = pipeline.parents("p", k=100, seed=0, home=home)["parents"]
    assert any(p["id"] == victim for p in control)  # present before the veto

    pipeline.remember("p", {"type": "discard", "id": victim}, home=home)
    after = pipeline.parents("p", k=100, seed=0, home=home)["parents"]
    assert all(p["id"] != victim for p in after)  # excluded after the veto

    # Re-pinning un-discards it: pins are always kept as parents.
    pipeline.remember("p", {"type": "pin", "id": victim}, home=home)
    repin = pipeline.parents("p", k=100, seed=0, home=home)["parents"]
    assert any(p["id"] == victim and p["pinned"] for p in repin)


def test_discard_persists_in_recall_and_is_mutually_exclusive(home):
    pipeline.init_project("r", _generic(), seed=0, home=home)
    pipeline.remember("r", {"type": "discard", "id": "x"}, home=home)
    rec = pipeline.recall("r", home=home)
    assert rec["discards"] == ["x"] and rec["pins"] == []

    # pin the same id -> moves to pins, leaves discards (latest action wins)
    pipeline.remember("r", {"type": "pin", "id": "x"}, home=home)
    rec = pipeline.recall("r", home=home)
    assert rec["pins"] == ["x"] and rec["discards"] == []
