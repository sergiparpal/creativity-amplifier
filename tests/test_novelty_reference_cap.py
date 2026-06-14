"""Bounded novelty/dedup reference growth.

The dedup and k-NN novelty passes run against the archive elites every cycle.
Below the cap the reference is exactly the embedded elites (so small projects are
unchanged); above it, only the most-novel ``NOVELTY_REF_CAP`` elites are used.
"""

from __future__ import annotations

import numpy as np

from creativity_engine import pipeline
from creativity_engine.archive import Archive
from creativity_engine.config import axes_spec_from_dict

SPEC = axes_spec_from_dict(
    {"domain": "t", "axes": [{"name": "m", "type": "open", "primary_novelty": True}]}
)


def _archive_with(n, embedded=None):
    """Archive with ``n`` niches; niche i gets novelty i/n. ``embedded`` selects
    which elites have stored embeddings (default: all)."""
    embedded = set(range(n)) if embedded is None else set(embedded)
    arc = Archive(SPEC)
    emb = {}
    for i in range(n):
        cid = f"c{i}"
        arc.place(cid, f"m=cell{i}", {"m": f"cell{i}"}, fitness=0.5, novelty=i / n)
        if i in embedded:
            emb[cid] = [1.0, 0.0]
    return arc, emb


def test_below_cap_is_identical_to_naive_list():
    arc, emb = _archive_with(10)
    naive = [eid for eid in arc.elite_ids() if eid in emb]
    assert pipeline._novelty_reference_ids(arc, emb, cap=100) == naive
    # the default cap (500) also leaves a small project untouched
    assert pipeline._novelty_reference_ids(arc, emb) == naive


def test_above_cap_keeps_most_novel():
    arc, emb = _archive_with(10)
    ref = pipeline._novelty_reference_ids(arc, emb, cap=3)
    assert len(ref) == 3
    # niche i has novelty i/10, so the three most-novel are c9, c8, c7
    assert set(ref) == {"c9", "c8", "c7"}
    assert all(r in emb for r in ref)


def test_unembedded_elites_excluded():
    # elites 0..4 lack embeddings; only 5..9 are eligible references
    arc, emb = _archive_with(10, embedded=range(5, 10))
    ref = pipeline._novelty_reference_ids(arc, emb, cap=100)
    assert set(ref) == {f"c{i}" for i in range(5, 10)}


def test_ingest_runs_with_tiny_cap(home, monkeypatch):
    # With a tiny cap the loop still ingests across cycles and yields a slate,
    # computing novelty against at most `cap` references.
    monkeypatch.setattr(pipeline, "NOVELTY_REF_CAP", 2)
    axes = {"domain": "cap", "axes": [{"name": "m", "type": "open", "primary_novelty": True}]}

    def gen(g):
        return [
            {"id": f"g{g}-{i}", "text": f"approach {g}-{i} via {w}",
             "descriptor": {"m": f"approach {g}-{i} via {w}"}}
            for i, w in enumerate(["alpha", "beta", "gamma", "delta", "epsilon"])
        ]

    pipeline.ingest("capproj", gen(0), axes, seed=0, home=home)
    res = pipeline.ingest("capproj", gen(1), axes, seed=0, home=home)
    assert res["slate"]  # still produces a slate
    # the reference the next cycle would use is bounded by the cap
    from creativity_engine.state import State

    arc = Archive.from_dict(SPEC, State("capproj", home=home).read_archive())
    emb = State("capproj", home=home).read_embeddings()
    assert len(pipeline._novelty_reference_ids(arc, emb)) <= 2
