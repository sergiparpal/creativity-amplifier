"""init-project resets stale geometry when the axes change, preserving memory.

Re-initing an existing project with a DIFFERENT axes geometry used to leave the
old niche keys in the archive, mixed with new ones on the next ingest. init-project
now detects an incompatible re-init and resets the geometric state (archive /
candidates / embeddings / frozen open-nicher) while keeping per-domain preference
memory. An identical re-init (or a settings-only change like slate_size) is a no-op.
"""

from __future__ import annotations

from creativity_engine import pipeline, selftest
from creativity_engine.state import State


def _axes(open_name="mechanism", cat="form"):
    return {
        "domain": "d",
        "unit_of_generation": "idea",
        "axes": [
            {"name": cat, "type": "categorical"},
            {"name": open_name, "type": "open", "primary_novelty": True},
        ],
        "slate_size": 4,
        "candidates_per_generation": 6,
    }


def _cands(n, open_name="mechanism", cat="form"):
    return [
        {"id": f"c{i}", "text": f"distinct idea number {i} about topic {i}",
         "descriptor": {cat: f"v{i}", open_name: f"approach {i}"}}
        for i in range(n)
    ]


def test_reinit_same_axes_preserves_archive(home):
    axes = _axes()
    pipeline.init_project("p", axes, seed=1, home=home)
    pipeline.ingest("p", _cands(4), axes, seed=1, home=home)
    before = State("p", home=home).read_archive()
    res = pipeline.init_project("p", axes, seed=1, home=home)  # identical axes
    assert res["reset"] is False
    assert State("p", home=home).read_archive() == before  # untouched


def test_reinit_changed_axes_resets_geometry_keeps_memory(home):
    axes = _axes()
    pipeline.init_project("p", axes, seed=1, home=home)
    pipeline.ingest("p", _cands(4), axes, seed=1, home=home)
    assert State("p", home=home).read_archive().get("niches")
    pipeline.remember("p", {"type": "pin", "id": "c0"}, home=home)

    axes2 = _axes(open_name="approach", cat="colour")  # renamed axes -> new geometry
    res = pipeline.init_project("p", axes2, seed=1, home=home)
    assert res["reset"] is True

    st = State("p", home=home)
    assert st.read_archive() == {}            # geometry wiped
    assert st.read_embeddings() == {}
    assert st.read_candidates() == {}
    assert st.read_open_nicher() is None
    assert "c0" in st.read_pins("d")          # preference memory preserved (same domain)
    meta = st.read_meta()
    assert "cos_window" not in meta and meta.get("cycles", 0) == 0  # series dropped


def test_reinit_slate_size_only_is_not_a_geometry_change(home):
    axes = _axes()
    pipeline.init_project("p", axes, seed=1, home=home)
    pipeline.ingest("p", _cands(4), axes, seed=1, home=home)
    axes2 = dict(axes, slate_size=2)  # slate_size does not affect niche placement
    res = pipeline.init_project("p", axes2, seed=1, home=home)
    assert res["reset"] is False
    assert State("p", home=home).read_archive().get("niches")


def test_post_reset_ingest_has_only_new_axis_keys(home):
    axes = _axes()
    pipeline.init_project("p", axes, seed=1, home=home)
    pipeline.ingest("p", _cands(4), axes, seed=1, home=home)
    axes2 = _axes(open_name="approach", cat="colour")
    pipeline.init_project("p", axes2, seed=1, home=home)
    pipeline.ingest(
        "p", _cands(4, open_name="approach", cat="colour"), axes2, seed=1, home=home
    )
    keys = list(State("p", home=home).read_archive()["niches"].keys())
    assert keys, "expected niches after the post-reset ingest"
    # No stale form=/mechanism= keys survive — every key is on the new axes.
    assert all(k.startswith("colour=") for k in keys)


def test_reinit_empty_project_does_not_reset(home):
    # init -> init (no ingest yet): there is no geometry to reset, even if axes differ.
    pipeline.init_project("p", _axes(), seed=1, home=home)
    res = pipeline.init_project("p", _axes(open_name="approach", cat="colour"),
                                seed=1, home=home)
    assert res["reset"] is False
