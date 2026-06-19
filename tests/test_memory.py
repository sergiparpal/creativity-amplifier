"""Phase 5: preference memory round-trips & stays namespaced; the active learner
finds the most-informative pair; parents never drop a pinned stepping stone."""

from __future__ import annotations

import json

import numpy as np
import pytest

from creativity_engine import memory
from creativity_engine.__main__ import main
from creativity_engine.state import State


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return (v / (n if n else 1.0)).tolist()


# --------------------------------------------------------------------------- #
# remember / recall
# --------------------------------------------------------------------------- #
def test_comparison_round_trip(home):
    st = State("p").ensure()
    memory.remember(st, "marketing", {"type": "comparison", "winner": "a", "loser": "b"})
    memory.remember(st, "marketing", {"type": "comparison", "winner": "a", "loser": "c"})
    rec = memory.recall(st, "marketing")
    assert len(rec["preferences"]) == 2
    assert rec["summary"]["win_counts"]["a"] == 2
    assert rec["summary"]["n_comparisons"] == 2


def test_pin_round_trip_and_idempotent(home):
    st = State("p").ensure()
    memory.remember(st, "marketing", {"type": "pin", "id": "c1"})
    memory.remember(st, "marketing", {"type": "pin", "id": "c1"})  # idempotent
    memory.remember(st, "marketing", {"type": "pin", "id": "c2"})
    assert memory.recall(st, "marketing")["pins"] == ["c1", "c2"]


def test_discard_round_trip_and_idempotent(home):
    st = State("p").ensure()
    memory.remember(st, "marketing", {"type": "discard", "id": "c1"})
    memory.remember(st, "marketing", {"type": "discard", "id": "c1"})  # idempotent
    memory.remember(st, "marketing", {"type": "discard", "id": "c2"})
    rec = memory.recall(st, "marketing")
    assert rec["discards"] == ["c1", "c2"]
    assert rec["pins"] == []  # discards never bleed into pins


def test_pin_and_discard_are_mutually_exclusive_latest_wins(home):
    st = State("p").ensure()
    # pin then discard the same id -> in discards, gone from pins
    memory.remember(st, "m", {"type": "pin", "id": "x"})
    memory.remember(st, "m", {"type": "discard", "id": "x"})
    rec = memory.recall(st, "m")
    assert rec["pins"] == [] and rec["discards"] == ["x"]
    # re-pin the same id -> back to pins, gone from discards (un-discarded)
    memory.remember(st, "m", {"type": "pin", "id": "x"})
    rec = memory.recall(st, "m")
    assert rec["pins"] == ["x"] and rec["discards"] == []


def test_preferred_values_from_winners(home):
    st = State("p").ensure()
    st.write_candidates(
        {
            "a": {"id": "a", "descriptor": {"register": "deadpan", "edginess": 0.8}},
            "b": {"id": "b", "descriptor": {"register": "earnest"}},
        }
    )
    memory.remember(st, "m", {"type": "comparison", "winner": "a", "loser": "b"})
    rec = memory.recall(st, "m")
    assert rec["summary"]["preferred_values"].get("register=deadpan") == 1


@pytest.mark.parametrize(
    "event",
    [
        {"winner": "a", "loser": "b"},  # missing type
        {"type": "comparison", "winner": "a"},  # missing loser
        {"type": "pin"},  # missing id
        {"type": "discard"},  # missing id
        {"type": "bogus"},  # unknown type
    ],
)
def test_remember_validation(home, event):
    st = State("p").ensure()
    with pytest.raises(ValueError):
        memory.remember(st, "m", event)


def test_memory_namespaced_per_domain(home):
    st = State("p").ensure()
    memory.remember(st, "marketing", {"type": "comparison", "winner": "a", "loser": "b"})
    memory.remember(st, "research", {"type": "comparison", "winner": "z", "loser": "y"})
    memory.remember(st, "marketing", {"type": "pin", "id": "p1"})
    m = memory.recall(st, "marketing")
    r = memory.recall(st, "research")
    assert m["summary"]["win_counts"] == {"a": 1}
    assert r["summary"]["win_counts"] == {"z": 1}
    assert m["pins"] == ["p1"]
    assert r["pins"] == []


# --------------------------------------------------------------------------- #
# active learning: most informative pair
# --------------------------------------------------------------------------- #
def _slate():
    # A,B: nearly identical embeddings, equal fitness, high novelty -> most informative
    # C,D: orthogonal, low novelty
    return (
        [
            {"id": "A", "fitness": 1.0, "novelty": 0.9},
            {"id": "B", "fitness": 1.0, "novelty": 0.9},
            {"id": "C", "fitness": 1.0, "novelty": 0.1},
            {"id": "D", "fitness": 1.0, "novelty": 0.1},
        ],
        {
            "A": _unit([1, 0, 0, 0]),
            "B": _unit([0.99, 0.01, 0, 0]),
            "C": _unit([0, 1, 0, 0]),
            "D": _unit([0, 0, 1, 0]),
        },
    )


def test_active_learner_picks_most_informative_pair():
    slate, emb = _slate()
    pairs = memory.select_ask_pairs(slate, emb, comparisons=[], max_pairs=2)
    assert pairs, "expected at least one pair"
    assert set(pairs[0][:2]) == {"A", "B"}


def test_active_learner_skips_decided_pairs():
    slate, emb = _slate()
    decided = [{"type": "comparison", "winner": "A", "loser": "B"}]
    pairs = memory.select_ask_pairs(slate, emb, comparisons=decided, max_pairs=2)
    # the otherwise-top (A,B) pair must not be asked again
    assert all(set(p[:2]) != {"A", "B"} for p in pairs)


def test_ask_weights_can_flip_policy_to_explore():
    # Default (positive sim weight) asks about the SIMILAR near-twins (learn the
    # preference function); a non-positive ask_sim_weight flips to region-separating
    # pairs (explore). Both readings are legitimate, so the weighting is tunable.
    slate, emb = _slate()
    default_top = memory.select_ask_pairs(slate, emb, [], max_pairs=2)[0]
    assert set(default_top[:2]) == {"A", "B"}

    explore_top = memory.select_ask_pairs(
        slate, emb, [], max_pairs=2, weights=(-0.5, 0.3, 0.2)
    )[0]
    assert set(explore_top[:2]) != {"A", "B"}
    a, b = explore_top[:2]
    sim = float(np.dot(np.asarray(emb[a]), np.asarray(emb[b])))
    assert sim < 0.5  # the chosen pair is embedding-dissimilar
    assert "region-separating" in explore_top[2]  # reason reflects the policy


def test_active_learner_handles_tiny_slate():
    assert memory.select_ask_pairs([{"id": "x"}], {}, []) == []


# --------------------------------------------------------------------------- #
# parents honor pins
# --------------------------------------------------------------------------- #
def _elite_pool():
    # five distinct directions
    ids = [f"e{i}" for i in range(5)]
    emb = {
        "e0": _unit([1, 0, 0, 0, 0]),
        "e1": _unit([0, 1, 0, 0, 0]),
        "e2": _unit([0, 0, 1, 0, 0]),
        "e3": _unit([0, 0, 0, 1, 0]),
        "e4": _unit([0, 0, 0, 0, 1]),
    }
    return ids, emb


def test_parents_includes_pin_and_is_diverse():
    ids, emb = _elite_pool()
    out = memory.select_parents(ids, emb, pins=["e0"], k=3)
    assert "e0" in out
    assert len(out) == 3
    assert len(set(out)) == 3  # no repeats


def test_parents_never_drops_pins_even_above_k():
    ids, emb = _elite_pool()
    out = memory.select_parents(ids, emb, pins=["e0", "e1", "e2"], k=1)
    # all three pins survive despite k=1
    for p in ("e0", "e1", "e2"):
        assert p in out


def test_parents_keeps_pin_without_embedding():
    ids, emb = _elite_pool()
    out = memory.select_parents(ids, emb, pins=["ghost"], k=3)
    assert out[0] == "ghost"  # pin with no embedding still kept, first
    assert len(out) == 3


def test_parents_empty_pool_returns_pins():
    out = memory.select_parents([], {}, pins=["e0"], k=4)
    assert out == ["e0"]


def test_parents_excludes_discarded_elites():
    ids, emb = _elite_pool()
    out = memory.select_parents(ids, emb, pins=[], k=5, discards=["e1", "e3"])
    assert "e1" not in out and "e3" not in out
    # the surviving elites can still fill the parent set
    assert set(out) <= {"e0", "e2", "e4"}


# --------------------------------------------------------------------------- #
# pipeline remember via CLI
# --------------------------------------------------------------------------- #
def test_cli_remember_round_trip(tmp_path, capsys, home):
    axes = tmp_path / "axes.json"
    axes.write_text(
        json.dumps(
            {"domain": "mk", "axes": [{"name": "a", "type": "categorical"}]}
        ),
        encoding="utf-8",
    )
    assert main(["init-project", "--project", "p", "--axes", str(axes)]) == 0
    capsys.readouterr()

    event = tmp_path / "ev.json"
    event.write_text(json.dumps({"type": "pin", "id": "c1"}), encoding="utf-8")
    assert main(["remember", "--project", "p", "--event", str(event)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True and out["id"] == "c1"

    assert main(["recall", "--project", "p"]) == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["pins"] == ["c1"]
    assert rec["domain"] == "mk"


def test_cli_discard_round_trip(tmp_path, capsys, home):
    axes = tmp_path / "axes.json"
    axes.write_text(
        json.dumps(
            {"domain": "mk", "axes": [{"name": "a", "type": "categorical"}]}
        ),
        encoding="utf-8",
    )
    assert main(["init-project", "--project", "p", "--axes", str(axes)]) == 0
    capsys.readouterr()

    event = tmp_path / "ev.json"
    event.write_text(json.dumps({"type": "discard", "id": "c9"}), encoding="utf-8")
    assert main(["remember", "--project", "p", "--event", str(event)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True and out["type"] == "discard" and out["id"] == "c9"

    assert main(["recall", "--project", "p"]) == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["discards"] == ["c9"]
