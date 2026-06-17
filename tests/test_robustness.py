"""Regression + coverage tests for the robustness fixes.

Covers: corrupt/torn persistence files, non-finite agent input, concurrent pins,
the CREATIVITY_DEBUG truthiness contract, the embedding-dimension guard, the
open-axis re-key merge, and flat-vector originality input.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from creativity_engine import pipeline
from creativity_engine.archive import Archive, continuous_bin
from creativity_engine.config import (
    Axis,
    Candidate,
    ConfigError,
    Niche,
    axes_spec_from_dict,
    debug_enabled,
)
from creativity_engine.originality import originality_scores
from creativity_engine.state import State


# --------------------------------------------------------------------------- #
# H1: a torn/corrupt comparisons line never poisons the whole history
# --------------------------------------------------------------------------- #
def test_read_comparisons_skips_torn_line(home):
    st = State("p").ensure()
    st.append_comparison("d", {"type": "comparison", "winner": "a", "loser": "b"})
    # Simulate an interrupted append: a partial, unterminated JSON line.
    with st.comparisons_path("d").open("a", encoding="utf-8") as fh:
        fh.write('{"type": "comparison", "winner":')
    out = st.read_comparisons("d")
    assert out == [{"type": "comparison", "winner": "a", "loser": "b", "context": ""}] \
        or out == [{"type": "comparison", "winner": "a", "loser": "b"}]


# --------------------------------------------------------------------------- #
# M5: empty vs corrupt JSON state files
# --------------------------------------------------------------------------- #
def test_read_json_empty_file_returns_default(home):
    st = State("p").ensure()
    st.meta_path.write_text("", encoding="utf-8")
    assert st.read_meta() == {}  # empty file treated as absent, not a crash


def test_read_json_corrupt_file_raises_configerror(home):
    st = State("p").ensure()
    st.meta_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        st.read_meta()


# --------------------------------------------------------------------------- #
# M1: the file lock provides mutual exclusion, and concurrent add_pin keeps every
# pin. The mutual-exclusion test is deterministic (no timing); the threaded test
# is the realistic end-to-end check.
# --------------------------------------------------------------------------- #
def test_file_lock_is_mutually_exclusive(home):
    from creativity_engine.state import _file_lock

    st = State("p").ensure()
    target = st.pins_path("d")
    target.parent.mkdir(parents=True, exist_ok=True)
    lockdir = target.parent / (target.name + ".lock")
    with _file_lock(target):
        assert lockdir.exists()
        # A second acquirer cannot create the same lock dir while it's held.
        with pytest.raises(FileExistsError):
            lockdir.mkdir()
    assert not lockdir.exists()  # released and cleaned up on exit


def test_add_pin_concurrent_no_loss(home):
    State("p").ensure()
    n = 16
    barrier = threading.Barrier(n)
    errors: list = []

    def worker(i: int) -> None:
        try:
            barrier.wait()  # release all workers together to maximize contention
            State("p").add_pin("d", f"pin{i}")
        except Exception as exc:  # surface a worker failure as a test failure
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"workers raised: {errors}"
    pins = State("p").read_pins("d")
    assert sorted(pins) == sorted(f"pin{i}" for i in range(n))


# --------------------------------------------------------------------------- #
# H2: non-finite numbers from the agent never crash placement / selection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", float("nan"), float("inf")])
def test_continuous_bin_non_finite_is_bin_zero(value):
    ax = Axis(name="x", type="continuous", range=(0.0, 10.0), bins=5)
    assert continuous_bin(ax, value) == 0


def test_ingest_tolerates_non_finite_descriptor(home):
    axes = {
        "domain": "d",
        "unit_of_generation": "idea",
        "axes": [
            {"name": "intensity", "type": "continuous", "range": [0, 10], "bins": 5},
            {"name": "mech", "type": "open", "primary_novelty": True},
        ],
    }
    pipeline.init_project("p", axes, seed=0)
    cands = {
        "candidates": [
            {"id": "c1", "text": "an idea about widgets",
             "descriptor": {"intensity": "nan", "mech": "lever"}},
            {"id": "c2", "text": "another idea on gadgets",
             "descriptor": {"intensity": 5, "mech": "pulley"}},
        ]
    }
    result = pipeline.ingest("p", cands, axes, seed=0)  # must not raise
    assert len(result["slate"]) >= 1


def test_candidate_rejects_non_finite_fitness():
    with pytest.raises(ConfigError):
        Candidate.from_dict({"id": "c", "text": "t", "fitness": float("nan")})
    with pytest.raises(ConfigError):
        Candidate.from_dict({"id": "c", "text": "t", "fitness": "inf"})


def test_axis_non_integer_bins_raises_configerror():
    with pytest.raises(ConfigError):
        axes_spec_from_dict(
            {"axes": [{"name": "x", "type": "continuous",
                       "range": [0, 1], "bins": "lots"}]}
        )


# --------------------------------------------------------------------------- #
# M7: CREATIVITY_DEBUG truthiness contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True), ("true", True), ("yes", True), ("on", True), ("anything", True),
        ("0", False), ("false", False), ("no", False), ("off", False), ("", False),
        ("  0  ", False), ("FALSE", False),
    ],
)
def test_debug_enabled(monkeypatch, value, expected):
    monkeypatch.setenv("CREATIVITY_DEBUG", value)
    assert debug_enabled() is expected


def test_debug_enabled_unset(monkeypatch):
    monkeypatch.delenv("CREATIVITY_DEBUG", raising=False)
    assert debug_enabled() is False


# --------------------------------------------------------------------------- #
# M8: load-bearing branches that were untested
# --------------------------------------------------------------------------- #
def test_guard_embedding_dim_rejects_mixed_widths():
    class _Emb:
        name = "hash"

    stored = {"a": [0.0] * 256}
    vecs = np.zeros((1, 512), dtype=np.float32)
    with pytest.raises(ConfigError):
        pipeline._guard_embedding_dim(stored, vecs, _Emb(), "proj")


def test_rekey_open_axis_merges_collision_by_elite():
    spec = axes_spec_from_dict(
        {
            "domain": "d",
            "unit_of_generation": "idea",
            "axes": [
                {"name": "type", "type": "categorical"},
                {"name": "mech", "type": "open", "primary_novelty": True},
            ],
        }
    )
    arc = Archive(spec)
    a_id = "type=cata|mech=cell0"
    b_id = "type=cata|mech=cell1"
    arc.niches[a_id] = Niche(
        id=a_id, coords={"type": "cata", "mech": "cell0"},
        elite_id="x", fitness=0.4, novelty=0.1,
    )
    arc.niches[b_id] = Niche(
        id=b_id, coords={"type": "cata", "mech": "cell1"},
        elite_id="y", fitness=0.9, novelty=0.2,
    )
    arc.counts = {a_id: 2, b_id: 3}

    # Both old niches re-key onto frozen cell 0 -> they collide and must merge.
    arc.rekey_open_axis(spec, "mech", {a_id: 0, b_id: 0})

    assert len(arc.niches) == 1
    merged = next(iter(arc.niches.values()))
    assert merged.elite_id == "y"          # higher fitness wins the merge
    assert merged.fitness == 0.9
    assert arc.counts[merged.id] == 5       # occupancy counts summed


# --------------------------------------------------------------------------- #
# originality: a single vector passed flat is scored, not silently dropped
# --------------------------------------------------------------------------- #
def test_originality_handles_flat_single_vectors():
    idea = np.array([1.0, 0.0, 0.0], dtype=np.float32)       # 1-D, one idea
    obvious = np.array([0.0, 1.0, 0.0], dtype=np.float32)     # 1-D, one referent
    out = originality_scores(idea, obvious)
    # orthogonal -> max cosine 0 -> originality 1.0 (not the dropped "0 ideas" case)
    assert len(out["per_idea"]) == 1
    assert out["per_idea"][0] == pytest.approx(1.0, abs=1e-4)


def test_originality_clamps_to_non_negative():
    # antipodal normalized vectors: 1 - (-1) = 2.0, clamped within [0, 2]
    idea = np.array([[1.0, 0.0]], dtype=np.float32)
    obvious = np.array([[-1.0, 0.0]], dtype=np.float32)
    out = originality_scores(idea, obvious)
    assert 0.0 <= out["per_idea"][0] <= 2.0
