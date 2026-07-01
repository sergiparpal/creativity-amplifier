"""S3 — generation-aware ask-policy schedule (CI check).

Deterministic plumbing only: the schedule resolves the right weight per 0-indexed
generation, the schedule is OFF by default (no silent flip), and the sign of the
similarity weight maps to the pair framing emitted by ``memory.select_ask_pairs``
(w_sim <= 0 -> region-separating / explore; > 0 -> similar / boundary refine).
This asserts no value improvement — that is a hypothesis validated offline, not in CI.
"""

from __future__ import annotations

from cambrian_engine import config, memory, pipeline, selftest
from cambrian_engine.config import EngineConfig
from cambrian_engine.state import State


def test_schedule_off_by_default_is_flat():
    c = EngineConfig()
    assert c.explore_until_generation == 0
    for g in range(4):
        assert c.ask_sim_weight_for_generation(g) == c.ask_sim_weight


def test_schedule_explores_early_refines_later():
    c = EngineConfig(explore_until_generation=1)  # explore on gen 0, refine after
    assert c.ask_sim_weight_for_generation(0) <= 0
    assert c.ask_sim_weight_for_generation(1) > 0
    assert c.ask_sim_weight_for_generation(2) > 0


def test_sign_maps_to_pair_framing():
    # §G contract: a non-positive sim weight surfaces region-separating pairs;
    # a positive one surfaces similar (boundary) pairs.
    slate = [
        {"id": "a", "fitness": 1.0, "novelty": 0.5},
        {"id": "b", "fitness": 1.0, "novelty": 0.5},
        {"id": "c", "fitness": 1.0, "novelty": 0.5},
    ]
    emb = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.7, 0.7]}
    explore = memory.select_ask_pairs(slate, emb, [], weights=(-0.5, 0.3, 0.2))
    refine = memory.select_ask_pairs(slate, emb, [], weights=(0.5, 0.3, 0.2))
    assert explore and "region-separating" in explore[0][2]
    assert refine and "similar" in refine[0][2]


def test_ingest_applies_schedule_per_generation(home):
    axes = config.load_generic_axes().to_dict()
    axes["engine"] = {"explore_until_generation": 1}
    pipeline.init_project("p", axes, seed=0, home=home)
    target = int(State("p", home=home).read_meta()["candidates_per_generation"])

    r0 = pipeline.ingest(
        "p", selftest.diverse_candidates(target, gen=0), axes, seed=0, home=home
    )
    r1 = pipeline.ingest(
        "p", selftest.diverse_candidates(target, gen=1), axes, seed=0, home=home
    )
    assert r0["ask_policy"]["generation"] == 0
    assert r0["ask_policy"]["phase"] == "explore"
    assert r0["ask_policy"]["ask_sim_weight_effective"] <= 0
    assert r1["ask_policy"]["generation"] == 1
    assert r1["ask_policy"]["phase"] == "refine"
    assert r1["ask_policy"]["ask_sim_weight_effective"] > 0


def test_ingest_default_project_stays_flat(home):
    axes = config.load_generic_axes().to_dict()  # no engine override -> schedule off
    pipeline.init_project("q", axes, seed=0, home=home)
    target = int(State("q", home=home).read_meta()["candidates_per_generation"])
    r = pipeline.ingest("q", selftest.diverse_candidates(target), axes, seed=0, home=home)
    assert r["ask_policy"]["phase"] == "refine"
    assert r["ask_policy"]["ask_sim_weight_effective"] == EngineConfig().ask_sim_weight
