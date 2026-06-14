"""Phase 1: state round-trips and stays namespaced per domain."""

from __future__ import annotations

import json

from creativity_engine.state import State, base_dir


def test_base_dir_respects_env(home):
    assert base_dir() == home


def test_ensure_and_paths(home):
    st = State("My Project!").ensure()
    assert st.root.exists()
    # project name is slugified onto the filesystem
    assert st.root.parent == home
    assert "My" in st.root.name and "/" not in st.root.name
    paths = st.paths()
    assert paths["root"] == str(st.root)


def test_json_round_trip(home):
    st = State("proj").ensure()
    axes = {"domain": "d", "axes": [{"name": "a", "type": "categorical"}]}
    st.write_axes(axes)
    assert st.read_axes() == axes

    archive = {"n1": {"id": "n1", "elite_id": "c1"}}
    st.write_archive(archive)
    assert st.read_archive() == archive

    embeddings = {"c1": [0.1, 0.2, 0.3]}
    st.write_embeddings(embeddings)
    assert st.read_embeddings() == embeddings


def test_missing_files_return_defaults(home):
    st = State("empty")
    assert st.read_archive() == {}
    assert st.read_candidates() == {}
    assert st.read_axes() is None
    assert st.exists() is False


def test_atomic_write_leaves_no_temp(home):
    st = State("proj").ensure()
    st.write_meta({"x": 1})
    leftovers = [p.name for p in st.root.iterdir() if p.name.startswith(".tmp-")]
    assert leftovers == []


def test_comparisons_append_and_read(home):
    st = State("proj").ensure()
    st.append_comparison("marketing", {"winner": "a", "loser": "b"})
    st.append_comparison("marketing", {"winner": "c", "loser": "d"})
    events = st.read_comparisons("marketing")
    assert len(events) == 2
    assert events[0]["winner"] == "a"


def test_memory_namespaced_per_domain(home):
    st = State("proj").ensure()
    st.append_comparison("marketing", {"winner": "a"})
    st.append_comparison("research", {"winner": "z"})
    assert len(st.read_comparisons("marketing")) == 1
    assert len(st.read_comparisons("research")) == 1
    assert st.read_comparisons("marketing")[0]["winner"] == "a"
    assert st.read_comparisons("research")[0]["winner"] == "z"
    # separate files on disk
    assert st.comparisons_path("marketing") != st.comparisons_path("research")


def test_pins_round_trip_and_dedup(home):
    st = State("proj").ensure()
    st.add_pin("marketing", "c1")
    st.add_pin("marketing", "c1")  # idempotent
    st.add_pin("marketing", "c2")
    assert st.read_pins("marketing") == ["c1", "c2"]


def test_write_json_is_valid_json(home):
    st = State("proj").ensure()
    st.write_meta({"b": 2, "a": 1})
    raw = st.meta_path.read_text(encoding="utf-8")
    assert json.loads(raw) == {"a": 1, "b": 2}
