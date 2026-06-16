"""Data-adaptive open-axis niching: cold-start separation + fit-once-then-freeze.

Asserts (a) the deterministic cold-start partition separates clearly-distinct
mechanisms more often than near-identical ones, and (b) once enough mechanism
embeddings accumulate, the partition is fitted once, frozen, persisted, and the
archive is re-keyed onto the frozen cells WITHOUT being scrambled — niche ids
stay stable across later cycles and the centroids are never refit.
"""

from __future__ import annotations

import numpy as np

from creativity_engine import pipeline
from creativity_engine.archive import Archive, CVTNicher
from creativity_engine.config import axes_spec_from_dict
from creativity_engine.embed import get_embedder
from creativity_engine.state import State

SPEC = {
    "domain": "freeze-test",
    "unit_of_generation": "idea",
    "axes": [{"name": "mechanism", "type": "open", "primary_novelty": True}],
}

_VERBS = ["spin", "weave", "forge", "graft", "echo", "sift", "braid", "kindle"]
_ADJS = ["amber", "glassy", "tidal", "mossy", "stark", "feral", "civic", "lunar"]
_NOUNS = ["lantern", "delta", "anvil", "orchard", "beacon", "prism", "ferry", "loom"]


def _candidates(n, gen=0, prefix="c"):
    """``n`` mechanism ideas spread across vocabulary so they don't dedup."""
    out = []
    for i in range(n):
        v = _VERBS[i % len(_VERBS)]
        a = _ADJS[(i // len(_VERBS)) % len(_ADJS)]
        no = _NOUNS[(i // (len(_VERBS) * len(_ADJS))) % len(_NOUNS)]
        mech = f"{v} the {a} {no} via approach {gen}-{i}"
        out.append(
            {
                "id": f"{prefix}{gen}-{i}",
                "text": mech,
                "descriptor": {"mechanism": mech},
            }
        )
    return out


def _cell_of(nid):
    # niche_id is "mechanism=cellN" for a single open axis
    return int(nid.split("cell")[1])


# --------------------------------------------------------------------------- #
# Cold start: distinct mechanisms separate more than near-identical ones
# --------------------------------------------------------------------------- #
def test_cold_start_separates_distinct_more_than_near_identical():
    emb = get_embedder("hash")
    nicher = CVTNicher(dim=emb.dim, k=24, seed=0)  # cold start, fixed centroids

    distinct = [
        ("referral incentives for loyal customers", "underwater midnight sculpture exhibit"),
        ("gamified streak with daily rewards", "mycelial relay across rooftops"),
        ("reverse auction for surplus seats", "ambient signage that hums at dusk"),
        ("peer teaching circles in libraries", "time-limited scarcity drop"),
    ]
    near = [
        ("referral incentives for loyal customers", "referral incentives for loyal customers!"),
        ("gamified streak with daily rewards", "gamified streak with daily reward"),
        ("reverse auction for surplus seats", "reverse auction for surplus seat"),
        ("peer teaching circles in libraries", "peer teaching circles in library"),
    ]

    def diff_rate(pairs):
        diff = 0
        for x, y in pairs:
            cx = nicher.cell(emb.embed([x])[0])
            cy = nicher.cell(emb.embed([y])[0])
            diff += int(cx != cy)
        return diff / len(pairs)

    assert diff_rate(distinct) > diff_rate(near)


def test_cold_start_assignment_is_deterministic():
    emb = get_embedder("hash")
    n1 = CVTNicher(dim=emb.dim, k=24, seed=7)
    n2 = CVTNicher(dim=emb.dim, k=24, seed=7)
    vecs = emb.embed([f"mechanism number {i}" for i in range(15)])
    assert n1.cells(vecs) == n2.cells(vecs)


# --------------------------------------------------------------------------- #
# Freeze: fit once, persist, re-key, stay stable
# --------------------------------------------------------------------------- #
def test_freeze_persists_centroids_and_rekeys_archive(home):
    spec = axes_spec_from_dict(SPEC)
    emb = get_embedder("hash")
    project = "freeze"
    # One generation past the freeze threshold (4 * OPEN_NICHES) in a single cycle.
    threshold = pipeline.OPEN_NICHE_FREEZE_FACTOR * pipeline.OPEN_NICHES
    pipeline.ingest(project, _candidates(threshold + 4), SPEC, seed=0, home=home)

    state = State(project, home=home)
    on = state.read_open_nicher()
    assert on is not None and on["frozen"] is True
    centroids = np.asarray(on["centroids"], dtype=np.float32)
    assert centroids.shape == (pipeline.OPEN_NICHES, emb.dim)

    # The archive must be re-keyed onto the frozen cells, not scrambled: every
    # surviving niche's cell equals the frozen-cell of its own elite's mechanism.
    nicher = CVTNicher.from_dict(on)
    arc = Archive.from_dict(spec, state.read_archive())
    cand = state.read_candidates()
    assert len(arc.niches) >= 2  # the mechanisms really did spread
    for nid, niche in arc.niches.items():
        mech = cand[niche.elite_id]["descriptor"]["mechanism"]
        assert nicher.cell(emb.embed([mech])[0]) == _cell_of(nid)
        # the candidate record's niche_id was refreshed to the frozen id
        assert cand[niche.elite_id]["niche_id"] == nid


def test_niche_ids_stable_after_freeze(home):
    spec = axes_spec_from_dict(SPEC)
    emb = get_embedder("hash")
    project = "stable"
    threshold = pipeline.OPEN_NICHE_FREEZE_FACTOR * pipeline.OPEN_NICHES
    pipeline.ingest(project, _candidates(threshold + 4), SPEC, seed=0, home=home)

    state = State(project, home=home)
    on1 = state.read_open_nicher()
    arc1 = Archive.from_dict(spec, state.read_archive())
    ids1 = set(arc1.niches)
    centroids1 = np.asarray(on1["centroids"], dtype=np.float32)

    # A later cycle of fresh mechanisms must NOT refit and must NOT rename ids.
    pipeline.ingest(project, _candidates(12, gen=1, prefix="late"), SPEC, seed=0, home=home)
    on2 = state.read_open_nicher()
    arc2 = Archive.from_dict(spec, state.read_archive())

    assert on2["frozen"] is True
    assert np.array_equal(centroids1, np.asarray(on2["centroids"], dtype=np.float32))
    # every pre-existing niche id still exists (stable, not renamed)
    assert ids1 <= set(arc2.niches)
    # the new candidates landed in frozen cells consistent with the nicher
    nicher = CVTNicher.from_dict(on2)
    cand = state.read_candidates()
    for niche in arc2.niches.values():
        mech = cand[niche.elite_id]["descriptor"]["mechanism"]
        assert nicher.cell(emb.embed([mech])[0]) == _cell_of(niche.id)


def test_freeze_refreshes_cand_store_and_slate_consistency(home):
    """After a freeze re-keys the archive, the elite candidate records (the ones
    slates are built from) must carry the NEW niche id/coords, and every slate
    item must point at a niche that actually exists in the re-keyed archive."""
    spec = axes_spec_from_dict(SPEC)
    project = "consistency"
    threshold = pipeline.OPEN_NICHE_FREEZE_FACTOR * pipeline.OPEN_NICHES
    result = pipeline.ingest(project, _candidates(threshold + 4), SPEC, seed=0, home=home)

    state = State(project, home=home)
    on = state.read_open_nicher()
    assert on is not None and on["frozen"] is True  # the freeze really fired

    arc = Archive.from_dict(spec, state.read_archive())
    cand = state.read_candidates()
    assert len(arc.niches) >= 2

    # Every elite record's display fields were re-synced to the re-keyed archive.
    for niche in arc.niches.values():
        rec = cand[niche.elite_id]
        assert rec["niche_id"] == niche.id
        assert rec["coords"] == niche.coords

    # Every slate item points at a niche present in the post-freeze archive
    # (no stale niche_id survives the re-key).
    archive_ids = set(arc.niches)
    assert result["slate"]  # the freeze cycle still produced a slate
    for item in result["slate"]:
        assert item["niche_id"] in archive_ids


def test_freeze_slate_selection_is_deterministic(home):
    """The display-field refresh must not perturb selection: slate ids are driven
    by the archive elites + embeddings, not the cand_store display fields, so two
    identical seeded runs across the freeze must yield the same slate ids."""
    spec = axes_spec_from_dict(SPEC)
    threshold = pipeline.OPEN_NICHE_FREEZE_FACTOR * pipeline.OPEN_NICHES
    cands = _candidates(threshold + 4)

    r1 = pipeline.ingest("det-a", cands, SPEC, seed=0, home=home)
    r2 = pipeline.ingest("det-b", cands, SPEC, seed=0, home=home)

    ids1 = {item["id"] for item in r1["slate"]}
    ids2 = {item["id"] for item in r2["slate"]}
    assert ids1 and ids1 == ids2

    # Slate ids are a subset of the archive's elite ids (selection is the archive's
    # job, untouched by the niche_id/coords refresh).
    arc = Archive.from_dict(spec, State("det-a", home=home).read_archive())
    assert ids1 <= set(arc.elite_ids())


def test_cold_start_before_threshold_is_not_frozen(home):
    project = "cold"
    # Well below the freeze threshold -> still accumulating, no centroids yet.
    pipeline.ingest(project, _candidates(10), SPEC, seed=0, home=home)
    on = State(project, home=home).read_open_nicher()
    assert on is not None
    assert on.get("frozen") is False
    assert "centroids" not in on
    assert len(on["accum"]) == 10


# --------------------------------------------------------------------------- #
# Observability (item 4a): freeze progress is reported by ingest + metrics
# --------------------------------------------------------------------------- #
def test_ingest_reports_open_axis_progress_on_cold_start(home):
    threshold = pipeline.OPEN_NICHE_FREEZE_FACTOR * pipeline.OPEN_NICHES
    res = pipeline.ingest("prog", _candidates(10), SPEC, seed=0, home=home)
    oa = res["open_axis"]
    assert oa["present"] is True
    assert oa["frozen"] is False
    assert oa["partition"] == "cold_start"
    assert oa["accumulated"] == 10
    assert oa["freeze_threshold"] == threshold
    assert 0.0 < oa["progress"] < 1.0
    # metrics reports the same partition status.
    assert pipeline.metrics("prog", home=home)["open_axis"]["accumulated"] == 10


def test_open_axis_status_flips_to_frozen_after_freeze(home):
    threshold = pipeline.OPEN_NICHE_FREEZE_FACTOR * pipeline.OPEN_NICHES
    res = pipeline.ingest("frz", _candidates(threshold + 4), SPEC, seed=0, home=home)
    oa = res["open_axis"]
    assert oa["frozen"] is True
    assert oa["partition"] == "frozen"
    assert oa["progress"] == 1.0
    # ...and metrics agrees after the freeze.
    moa = pipeline.metrics("frz", home=home)["open_axis"]
    assert moa["frozen"] is True and moa["progress"] == 1.0


def test_open_axis_absent_when_no_open_axis(home):
    # A spec with only a categorical axis has no open/primary novelty axis.
    spec = {"domain": "noopen", "axes": [{"name": "kind", "type": "categorical"}]}
    res = pipeline.ingest(
        "noopen",
        [{"id": "c0", "text": "an idea", "descriptor": {"kind": "a"}}],
        spec, seed=0, home=home,
    )
    assert res["open_axis"] == {"present": False}
