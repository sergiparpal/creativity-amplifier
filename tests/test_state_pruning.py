"""State hygiene (item 4c): long sessions prune candidate records + embeddings
that nothing reads again, bounding the per-cycle whole-file rewrite cost.

The keep set is exactly what the engine still consumes — archive elites
(dedup/novelty/slate), pins (parents), and comparison ids (recall's learned
preferences) — so pruning must never change engine output.
"""

from __future__ import annotations

from creativity_engine import config, pipeline, selftest
from creativity_engine.archive import Archive
from creativity_engine.state import State


# --------------------------------------------------------------------------- #
# unit: the keep-set logic
# --------------------------------------------------------------------------- #
def test_maybe_prune_keeps_only_keepset():
    cand = {f"c{i}": {"text": str(i)} for i in range(10)}
    emb = {f"c{i}": [float(i)] for i in range(10)}
    keep = {"c1", "c3", "c5"}
    n = pipeline._maybe_prune_state(cand, emb, keep, threshold=5)
    assert n == 7
    assert set(cand) == keep
    assert set(emb) == keep


def test_maybe_prune_is_noop_below_threshold():
    cand = {f"c{i}": {} for i in range(4)}
    emb = {f"c{i}": [0.0] for i in range(4)}
    assert pipeline._maybe_prune_state(cand, emb, {"c0"}, threshold=5) == 0
    assert len(cand) == 4 and len(emb) == 4


def test_maybe_prune_disabled_when_threshold_zero():
    cand = {f"c{i}": {} for i in range(100)}
    emb = {f"c{i}": [0.0] for i in range(100)}
    assert pipeline._maybe_prune_state(cand, emb, set(), threshold=0) == 0
    assert len(cand) == 100


def test_maybe_prune_sweeps_orphan_embeddings():
    # When pruning triggers, an embedding with no cand_store record and not in the
    # keep set is swept too, so the two stores stay aligned.
    cand = {f"c{i}": {} for i in range(5)}
    emb = {f"c{i}": [0.0] for i in range(5)}
    emb["orphan"] = [1.0]  # not in cand_store and not kept
    n = pipeline._maybe_prune_state(cand, emb, {"c0"}, threshold=3)  # len(cand)=5 > 3
    assert n == 4  # the four dropped cand_store records (orphan isn't counted)
    assert set(cand) == {"c0"}
    assert set(emb) == {"c0"}


# --------------------------------------------------------------------------- #
# integration: pruning is wired into ingest and preserves output
# --------------------------------------------------------------------------- #
def _axes(prune_threshold: int):
    axes = config.load_generic_axes().to_dict()
    axes["engine"] = {"state_prune_threshold": prune_threshold}
    return axes


def test_ingest_prune_keeps_referenced_and_drops_rest(home):
    # ingest re-resolves the engine config each call, so cycle 1 runs with pruning
    # OFF (build non-elite history) and cycle 2 with it ON (the prune fires).
    spec = config.load_generic_axes()
    domain = spec.domain

    axes_off = _axes(prune_threshold=0)
    pipeline.init_project("p", axes_off, seed=0, home=home)
    pipeline.ingest("p", selftest.single_shot_candidates(12, prefix="a"), axes_off, seed=0, home=home)

    st = State("p", home=home)
    arc = Archive.from_dict(spec, st.read_archive())
    elites = set(arc.elite_ids())
    all_ids = set(st.read_candidates())
    non_elites = list(all_ids - elites)
    assert non_elites, "need a non-elite to prove the keep-set isn't just elites"

    # Pin a NON-elite (so its survival can only be due to the pin) and record a
    # comparison referencing another id.
    pinned = non_elites[0]
    win = non_elites[1] if len(non_elites) > 1 else non_elites[0]
    lose = next(iter(elites))
    pipeline.remember("p", {"type": "pin", "id": pinned}, home=home)
    pipeline.remember("p", {"type": "comparison", "winner": win, "loser": lose}, home=home)

    # Cycle 2 with pruning ON triggers the prune.
    res = pipeline.ingest("p", selftest.single_shot_candidates(12, prefix="b"), axes_off | {"engine": {"state_prune_threshold": 3}}, seed=0, home=home)

    st = State("p", home=home)
    cand = set(st.read_candidates())
    emb = set(st.read_embeddings())
    arc2 = Archive.from_dict(spec, st.read_archive())
    elites2 = set(arc2.elite_ids())
    pins = set(st.read_pins(domain))
    comp_ids = {win, lose}
    keep = elites2 | pins | comp_ids

    # Nothing outside the keep set survives, and everything in it is retained.
    assert cand <= keep, f"pruned too little: {cand - keep}"
    assert emb <= keep
    assert elites2 <= cand, "an elite was wrongly pruned"
    assert pinned in cand, "the non-elite pin was wrongly pruned"
    assert comp_ids <= cand, "a comparison id was wrongly pruned"

    # The engine still produces output after pruning.
    assert res["slate"]
    assert pipeline.parents("p", home=home)["parents"]


def test_default_threshold_does_not_prune_small_session(home):
    # The default (2000) never triggers in a normal session: all history retained.
    axes = config.load_generic_axes().to_dict()  # default engine config
    pipeline.init_project("p", axes, seed=0, home=home)
    for g in range(3):
        pipeline.ingest("p", selftest.diverse_candidates(12, gen=g), axes, seed=0, home=home)
    n_cands = len(State("p", home=home).read_candidates())
    # 3 generations of distinct candidates: well under 2000, nothing dropped.
    assert n_cands >= 24
