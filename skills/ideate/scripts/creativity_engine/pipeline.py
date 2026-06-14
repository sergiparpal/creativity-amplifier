"""High-level orchestration for the CLI commands.

Each public function returns a JSON-serializable dict (or raises). The CLI in
``__main__`` is a thin wrapper that parses args, calls these, and prints JSON.

The ``ingest`` flow is the heart of one cycle:
embed → dedup → place (MAP-Elites over the resolved axes) → geometric novelty →
DPP diverse slate → anti-collapse monitor. The judge is never called here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:  # import for type hints only; avoids a runtime import cycle
    from .state import State

from . import (
    __version__,
    archive as archive_mod,
    config,
    diversity,
    memory,
    monitor,
    novelty,
)
from .config import AxesSpec, Candidate
from .embed import dedupe, default_dedup_tau
from .session import Session

# Default tuning knobs. These are the **fallback defaults** for direct/library
# callers and the self-test's placement helper; the real ``ingest`` path resolves
# every knob from :class:`config.EngineConfig` (per-domain overridable). The values
# here MUST mirror ``EngineConfig``'s defaults — ``test_engine_config`` guards that.
#
# The near-duplicate cosine threshold is per-embedder (see ``embed.default_dedup_tau``).
KNN_K = 5
# Open-axis (mechanism) niching. The partition is data-adaptive: cold-start fixed
# centroids until ``OPEN_NICHE_FREEZE_FACTOR * OPEN_NICHES`` mechanism embeddings
# have accumulated, then a one-time k-means fit freezes the cells (see
# ``_accumulate_and_maybe_freeze``).
OPEN_NICHES = 24
OPEN_NICHE_FREEZE_FACTOR = 4
MAX_DPP_POOL = 200
# Cap on the dedup/novelty reference set (the most-novel elites). Dedup and k-NN
# novelty run against the archive elites every cycle (O(n·m)); without a cap this
# grows unbounded. Below the cap behavior is identical to using every elite.
NOVELTY_REF_CAP = 500
# How much the judge's (bounded) fitness is allowed to weight the DPP slate.
# 0 -> pure diversity; 1 -> full quality-diversity. Kept low so geometry owns
# the slate and the judge can only nudge ordering within an already-diverse pool.
QUALITY_WEIGHT = 0.3


# --------------------------------------------------------------------------- #
# init-project
# --------------------------------------------------------------------------- #
def init_project(
    project: str,
    axes_source,
    seed: int = 0,
    home: Optional[Path] = None,
) -> Dict[str, Any]:
    """Create state dirs and snapshot the resolved axes for the session.

    The axes geometry goes to ``axes.json``; the agent-/session-level settings
    that ride alongside it (candidates-per-generation, judge rubric) are
    recorded in ``meta.json`` — kept out of the engine's :class:`AxesSpec`.
    """
    spec = config.load_axes(axes_source)
    settings = config.load_session_settings(axes_source)
    econfig = config.load_engine_config(axes_source)
    sess = Session(project, home=home, seed=seed).ensure()
    sess.adopt_spec(spec)
    meta = sess.state.read_meta()
    meta.update(
        {
            "project": project,
            "domain": spec.domain,
            "unit_of_generation": spec.unit_of_generation,
            "candidates_per_generation": settings.candidates_per_generation,
            "judge_rubric": settings.judge_rubric,
            "engine": econfig.to_dict(),
            "seed": int(seed),
            "version": __version__,
        }
    )
    sess.state.write_meta(meta)
    return {"ok": True, "domain": spec.domain, "paths": sess.state.paths()}


# --------------------------------------------------------------------------- #
# recall
# --------------------------------------------------------------------------- #
def recall(project: str, k: int = 10, home: Optional[Path] = None) -> Dict[str, Any]:
    """Return memory for in-context injection: recent choices, pins, win tallies."""
    sess = Session(project, home=home)
    return memory.recall(sess.state, sess.domain, k=k)


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #
def _parse_candidates(candidates) -> List[Candidate]:
    if isinstance(candidates, dict):
        candidates = candidates.get("candidates", [])
    if not isinstance(candidates, list):
        raise config.ConfigError("candidates must be a list (or {candidates: [...]})")
    return [Candidate.from_dict(c) for c in candidates]


def _survivor_novelty(
    surv_vecs: np.ndarray, existing_vecs: np.ndarray, k: int
) -> np.ndarray:
    """Mean k-NN distance of each survivor to (existing ∪ other survivors)."""
    n = surv_vecs.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    if existing_vecs.shape[0] > 0:
        ref = np.vstack([existing_vecs, surv_vecs])
        offset = existing_vecs.shape[0]
    else:
        ref = surv_vecs
        offset = 0
    dist = novelty.cosine_distance_matrix(surv_vecs, ref)
    # Mask each survivor's own row in the combined reference so it isn't its own
    # neighbour, then reuse the shared mean-k-NN kernel.
    dist[np.arange(n), offset + np.arange(n)] = np.inf
    return novelty.mean_knn_distance(dist, k, n_neighbors=ref.shape[0] - 1)


def _slate_item(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": record["id"],
        "text": record.get("text", ""),
        "descriptor": record.get("descriptor", {}),
        "genealogy": record.get("genealogy", {}),
        "niche_id": record.get("niche_id"),
        "coords": record.get("coords", {}),
        "novelty": round(float(record.get("novelty", 0.0)), 4),
        "fitness": round(float(record.get("fitness", 1.0)), 4),
        # The agent looks an embedding up by candidate id, so the ref IS the id;
        # kept as a distinct field so the contract survives if that ever changes.
        "embedding_ref": record["id"],
    }


def _open_axis_texts(
    open_axis: Any, descriptors: List[Dict[str, Any]], texts: List[str]
) -> List[str]:
    """The text to embed for the open axis: its descriptor value, else the idea."""
    return [str(d.get(open_axis.name) or t) for d, t in zip(descriptors, texts)]


def assign_open_cells(
    spec: AxesSpec,
    descriptors: List[Dict[str, Any]],
    texts: List[str],
    embedder,
    seed: int,
    nicher: Optional["archive_mod.CVTNicher"] = None,
    open_niches: int = OPEN_NICHES,
) -> Tuple[Optional[Any], List[Optional[int]], np.ndarray]:
    """Voronoi cell per item for the primary "open" axis.

    Returns ``(open_axis, cells, open_vecs)`` where ``cells[i]`` is the item's
    cell index (or ``None`` when there is no open axis) and ``open_vecs`` are the
    embedded open-axis texts (so callers can accumulate them without re-embedding).
    When ``nicher`` is given (a frozen, data-fitted partition) it is used as-is;
    otherwise a deterministic cold-start partition is built. Shared by
    :func:`ingest` and the self-test so both place candidates identically.
    """
    open_axis = spec.primary_axis
    n = len(texts)
    if open_axis is None or n == 0:
        return open_axis, [None] * n, np.zeros((0, 1), dtype=np.float32)
    open_texts = _open_axis_texts(open_axis, descriptors, texts)
    open_vecs = embedder.embed(open_texts)
    if nicher is None:
        nicher = archive_mod.CVTNicher(dim=open_vecs.shape[1], k=open_niches, seed=seed)
    return open_axis, nicher.cells(open_vecs), open_vecs


def _frozen_open_nicher(
    on_state: Optional[Dict[str, Any]], open_axis: Optional[Any]
) -> Optional["archive_mod.CVTNicher"]:
    """The persisted frozen nicher, or ``None`` (cold start / no open axis)."""
    if open_axis is None or not on_state:
        return None
    if on_state.get("frozen") and on_state.get("centroids"):
        return archive_mod.CVTNicher.from_dict(on_state)
    return None


def _elite_open_cells(
    arc: "archive_mod.Archive",
    cand_store: Dict[str, Any],
    open_axis: Any,
    embedder,
    nicher: "archive_mod.CVTNicher",
) -> Dict[str, int]:
    """Frozen-cell index for each niche, from its elite's open-axis embedding."""
    nids: List[str] = []
    texts: List[str] = []
    for nid, niche in arc.niches.items():
        rec = cand_store.get(niche.elite_id, {})
        # Falls back descriptor-value -> idea text -> "". An empty mechanism embeds
        # to a zero vector and lands in cell 0; such niches merge there on freeze.
        # Harmless in practice (candidates always carry text), just deterministic.
        mech = str(rec.get("descriptor", {}).get(open_axis.name) or rec.get("text") or "")
        nids.append(nid)
        texts.append(mech)
    if not texts:
        return {}
    cells = nicher.cells(embedder.embed(texts))
    return {nid: cell for nid, cell in zip(nids, cells)}


def _accumulate_and_maybe_freeze(
    state: State,
    on_state: Optional[Dict[str, Any]],
    open_axis: Optional[Any],
    open_vecs: np.ndarray,
    arc: "archive_mod.Archive",
    cand_store: Dict[str, Any],
    spec: AxesSpec,
    embedder,
    seed: int,
    open_niches: int = OPEN_NICHES,
    freeze_factor: int = OPEN_NICHE_FREEZE_FACTOR,
) -> None:
    """Grow the mechanism-embedding buffer; freeze the partition once it's full.

    Until ``freeze_factor * open_niches`` open-axis embeddings have accumulated we
    stay on the cold-start partition (only the buffer grows). On the cycle that
    crosses the threshold we fit k-means **once**, persist the frozen centroids,
    and re-key the archive onto them (:meth:`Archive.rekey_open_axis`) so existing
    niche ids migrate without being scrambled. After freezing we never refit.
    """
    if open_axis is None:
        return
    on_state = dict(on_state or {})
    if on_state.get("frozen"):
        return  # already frozen — niche ids are fixed

    accum: List[List[float]] = list(on_state.get("accum", []))
    if open_vecs.shape[0]:
        accum.extend([[float(x) for x in v] for v in open_vecs])

    threshold = freeze_factor * open_niches
    if len(accum) < threshold:
        state.write_open_nicher({"frozen": False, "accum": accum})
        return

    # Threshold reached: fit once, freeze, and re-bucket the archive.
    nicher = archive_mod.CVTNicher.fit(
        np.asarray(accum, dtype=np.float32), k=open_niches, seed=seed
    )
    cell_by_nid = _elite_open_cells(arc, cand_store, open_axis, embedder, nicher)
    arc.rekey_open_axis(spec, open_axis.name, cell_by_nid)
    # Surviving elites carry the old niche id in their candidate record — refresh.
    for nid, niche in arc.niches.items():
        rec = cand_store.get(niche.elite_id)
        if rec is not None:
            rec["niche_id"] = nid
            rec["coords"] = niche.coords

    frozen = nicher.to_dict()
    frozen["frozen"] = True
    state.write_open_nicher(frozen)


def _empty_cycle(arc: "archive_mod.Archive") -> Dict[str, Any]:
    """Result dict for a generation with no candidates to ingest."""
    return {
        "slate": [],
        "ask_pairs": [],
        "monitor": monitor.evaluate(np.zeros((0, 1)), arc.niche_counts()),
        "parents": [],
    }


def _guard_embedding_dim(
    stored_emb: Dict[str, List[float]], vecs: np.ndarray, embedder, project: str
) -> None:
    """Fail loudly if a prior embedder wrote incompatible-dimension vectors.

    Mixing dimensions within a project would give dedup/novelty ragged arrays.
    """
    if not stored_emb:
        return
    existing_dim = len(next(iter(stored_emb.values())))
    if existing_dim != vecs.shape[1]:
        raise config.ConfigError(
            f"project {project!r} has {existing_dim}-dim embeddings but the "
            f"current embedder ({embedder.name!r}) produces {vecs.shape[1]}-dim "
            f"vectors; reuse the original embedder ($CREATIVITY_EMBEDDER) or "
            f"start a fresh project."
        )


def _novelty_reference_ids(
    arc: "archive_mod.Archive",
    stored_emb: Dict[str, List[float]],
    cap: Optional[int] = None,
) -> List[str]:
    """Elite ids used as the dedup/novelty reference, capped to the most-novel.

    At or below ``cap`` this is exactly the embedded elites in archive order, so
    small-project behavior is unchanged. Above ``cap`` it keeps the ``cap``
    most-novel elites, bounding the O(n·m) dedup and k-NN novelty passes. ``cap``
    defaults to the module-level :data:`NOVELTY_REF_CAP`, read at call time so it
    stays overridable.
    """
    if cap is None:
        cap = NOVELTY_REF_CAP
    ids = [eid for eid in arc.elite_ids() if eid in stored_emb]
    if len(ids) <= cap:
        return ids
    novelty_by_elite = {n.elite_id: n.novelty for n in arc.niches.values()}
    ids.sort(key=lambda eid: novelty_by_elite.get(eid, 0.0), reverse=True)
    return ids[:cap]


def _stack_embeddings(
    ids: List[str], stored_emb: Dict[str, List[float]], dim: int
) -> np.ndarray:
    """``(len(ids), dim)`` float32 matrix of the given ids' vectors (empty if none)."""
    if ids:
        return np.asarray([stored_emb[i] for i in ids], dtype=np.float32)
    return np.zeros((0, dim), dtype=np.float32)


def _place_survivors(
    survivors: List[Candidate],
    surv_vecs: np.ndarray,
    cells: List[Optional[int]],
    novelties: np.ndarray,
    open_axis: Optional[Any],
    spec: AxesSpec,
    arc: "archive_mod.Archive",
    cand_store: Dict[str, Any],
    stored_emb: Dict[str, List[float]],
) -> None:
    """Insert each survivor into its niche; record its candidate + embedding."""
    for idx, c in enumerate(survivors):
        ocell = {}
        if open_axis is not None and cells[idx] is not None:
            ocell = {open_axis.name: cells[idx]}
        nid, coords = archive_mod.compute_niche(c.descriptor, spec, ocell)
        nov = float(novelties[idx])
        arc.place(c.id, nid, coords, fitness=c.fitness, novelty=nov)
        cand_store[c.id] = {
            **c.to_dict(),
            "niche_id": nid,
            "coords": coords,
            "novelty": nov,
        }
        stored_emb[c.id] = [float(x) for x in surv_vecs[idx]]


def _select_slate(
    arc: "archive_mod.Archive",
    stored_emb: Dict[str, List[float]],
    cand_store: Dict[str, Any],
    spec: AxesSpec,
    seed: int,
    max_dpp_pool: int = MAX_DPP_POOL,
    quality_weight: float = QUALITY_WEIGHT,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """DPP diverse slate over the current niche elites. Returns ``(slate, ids)``."""
    elites: List[Tuple[str, float, str]] = [
        (niche.elite_id, niche.fitness, nid)
        for nid, niche in arc.niches.items()
        if niche.elite_id and niche.elite_id in stored_emb
    ]
    # cap the pool by novelty for latency
    if len(elites) > max_dpp_pool:
        elites.sort(
            key=lambda e: cand_store.get(e[0], {}).get("novelty", 0.0), reverse=True
        )
        elites = elites[:max_dpp_pool]
    if not elites:
        return [], []

    elite_ids = [e[0] for e in elites]
    elite_vecs = np.asarray([stored_emb[i] for i in elite_ids], dtype=np.float32)
    quality = np.asarray([e[1] for e in elites], dtype=np.float64)
    sel = diversity.select_diverse(
        elite_vecs, k=spec.slate_size, quality=quality, seed=seed,
        quality_weight=quality_weight,
    )
    slate_ids = [elite_ids[i] for i in sel]
    slate = [_slate_item(cand_store[i]) for i in slate_ids]
    return slate, slate_ids


def _persist_cycle(
    state: State,
    arc: "archive_mod.Archive",
    stored_emb: Dict[str, List[float]],
    cand_store: Dict[str, Any],
    vecs: np.ndarray,
    embedder,
    mon: Dict[str, Any],
    econfig: "config.EngineConfig",
) -> None:
    """Write archive/embeddings/candidates, bump cycle metadata, roll the window."""
    state.write_archive(arc.to_dict())
    state.write_embeddings(stored_emb)
    state.write_candidates(cand_store)
    meta = state.read_meta()
    meta["cycles"] = int(meta.get("cycles", 0)) + 1
    meta["embedder"] = embedder.name
    meta["embedding_dim"] = int(vecs.shape[1])
    meta["engine"] = econfig.to_dict()  # keep the resolved knobs visible/auditable
    # Roll the monitor's calibration window with this generation's mean cosine.
    # The window must track the project's *normal* diversity scale, never the
    # collapse it exists to detect. So once a calibrated baseline exists we exclude
    # any generation the RELATIVE rule flags as too similar — otherwise a sustained
    # collapse trains the baseline up past itself and the flag goes quiet (a
    # "boiling-frog" blind spot). While still bootstrapping (no calibrated baseline
    # yet) we add every generation regardless, so the window can form even under an
    # embedder whose natural cosine scale trips the absolute fallback.
    suppress = mon.get("calibrated") and mon.get("too_similar")
    if int(mon.get("n", 0)) >= 2 and not suppress:
        cos_window = list(meta.get("cos_window", []))
        cos_window.append(float(mon["mean_cosine"]))
        meta["cos_window"] = cos_window[-econfig.monitor_window:]
    state.write_meta(meta)


def ingest(
    project: str,
    candidates,
    axes_source,
    seed: int = 0,
    home: Optional[Path] = None,
) -> Dict[str, Any]:
    """Embed → dedup → place → novelty → archive → DPP → monitor for one cycle."""
    spec = config.load_axes(axes_source)
    econfig = config.load_engine_config(axes_source)
    sess = Session(project, home=home, seed=seed).ensure()
    # The axes passed in are authoritative for this cycle; snapshot them only on
    # a fresh project so an existing project keeps its original resolved axes.
    if sess.state.read_axes() is None:
        sess.adopt_spec(spec)
    state = sess.state

    cand_list = _parse_candidates(candidates)
    arc = archive_mod.Archive.from_dict(spec, state.read_archive())
    if not cand_list:
        return _empty_cycle(arc)

    stored_emb: Dict[str, List[float]] = state.read_embeddings()
    cand_store: Dict[str, Any] = state.read_candidates()

    embedder = sess.embedder
    vecs = embedder.embed([c.text for c in cand_list])
    _guard_embedding_dim(stored_emb, vecs, embedder, project)

    # Existing archive elites seed both dedup and the novelty reference, capped to
    # the most-novel novelty_ref_cap so the per-cycle cost stays bounded.
    existing_ids = _novelty_reference_ids(arc, stored_emb, cap=econfig.novelty_ref_cap)
    existing_vecs = _stack_embeddings(existing_ids, stored_emb, vecs.shape[1])

    tau = econfig.dedup_tau if econfig.dedup_tau is not None else default_dedup_tau(
        embedder.name
    )
    keep, _drop = dedupe(
        vecs, tau=tau, existing=existing_vecs if existing_vecs.shape[0] else None
    )
    survivors = [cand_list[i] for i in keep]
    surv_vecs = vecs[keep] if keep else np.zeros((0, vecs.shape[1]), dtype=np.float32)

    # Open-axis niching is data-adaptive: use the frozen partition if one has been
    # fitted, else the deterministic cold-start partition.
    open_axis = spec.primary_axis
    on_state = state.read_open_nicher()
    frozen_nicher = _frozen_open_nicher(on_state, open_axis)
    open_axis, cells, open_vecs = assign_open_cells(
        spec, [c.descriptor for c in survivors], [c.text for c in survivors],
        embedder, seed, nicher=frozen_nicher, open_niches=econfig.open_niches,
    )
    novelties = _survivor_novelty(surv_vecs, existing_vecs, econfig.knn_k)
    _place_survivors(
        survivors, surv_vecs, cells, novelties, open_axis,
        spec, arc, cand_store, stored_emb,
    )

    # Accumulate the mechanism embeddings; once enough exist, fit + freeze the
    # partition once and re-key the archive onto the frozen cells.
    _accumulate_and_maybe_freeze(
        state, on_state, open_axis, open_vecs, arc, cand_store, spec,
        embedder, seed, open_niches=econfig.open_niches,
        freeze_factor=econfig.open_niche_freeze_factor,
    )

    slate, slate_ids = _select_slate(
        arc, stored_emb, cand_store, spec, seed,
        max_dpp_pool=econfig.max_dpp_pool, quality_weight=econfig.quality_weight,
    )

    # Monitor the RAW generation (pre-dedup) so a near-duplicate batch still
    # registers as collapsing — dedup would otherwise hide it behind survivors.
    # The baseline is the rolling window of prior generations' mean cosine, so the
    # similarity flag is calibrated to this project rather than a fixed constant.
    baseline = list(state.read_meta().get("cos_window", []))
    mon = monitor.evaluate(
        vecs, arc.niche_counts(), baseline=baseline,
        cos_threshold=econfig.monitor_cos_threshold,
        entropy_threshold=econfig.monitor_entropy_threshold,
        margin=econfig.monitor_margin,
        cos_ceiling=econfig.monitor_cos_ceiling,
        min_baseline=econfig.monitor_min_baseline,
    )

    # Namespace preference memory by the session domain so ingest is consistent
    # with remember/recall/parents (all share Session's snapshot resolution).
    domain = sess.domain
    comparisons = state.read_comparisons(domain)
    ask_pairs = memory.select_ask_pairs(slate, stored_emb, comparisons, max_pairs=2)

    _persist_cycle(state, arc, stored_emb, cand_store, vecs, embedder, mon, econfig)
    return {
        "slate": slate,
        "ask_pairs": ask_pairs,
        "monitor": mon,
        "parents": slate_ids,
    }


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def metrics(project: str, home: Optional[Path] = None) -> Dict[str, Any]:
    """Current archive health: entropy, mean cosine, coverage, n."""
    sess = Session(project, home=home)
    arc = archive_mod.Archive.from_dict(sess.spec, sess.state.read_archive())
    stored_emb = sess.state.read_embeddings()
    elite_ids = [i for i in arc.elite_ids() if i in stored_emb]
    # dim only matters in the empty case (the resulting (0, dim) array is never
    # used in arithmetic); take it from a stored vector when one exists.
    dim = len(next(iter(stored_emb.values()))) if stored_emb else 1
    elite_vecs = _stack_embeddings(elite_ids, stored_emb, dim=dim)
    mon = monitor.evaluate(elite_vecs, arc.niche_counts())
    return {
        "entropy": mon["entropy"],
        "mean_cosine": mon["mean_cosine"],
        "coverage": mon["coverage"],
        "n": len(elite_ids),
    }


# --------------------------------------------------------------------------- #
# remember / parents
# --------------------------------------------------------------------------- #
def remember(project: str, event: Dict[str, Any],
             home: Optional[Path] = None) -> Dict[str, Any]:
    """Append a comparison/pin to this domain's preference memory."""
    sess = Session(project, home=home).ensure()
    return memory.remember(sess.state, sess.domain, event)


def parents(project: str, k: int = 4, seed: int = 0,
            home: Optional[Path] = None) -> Dict[str, Any]:
    """Diverse parents for the next generation; pinned stepping stones kept."""
    sess = Session(project, home=home, seed=seed)
    arc = archive_mod.Archive.from_dict(sess.spec, sess.state.read_archive())
    stored_emb = sess.state.read_embeddings()
    cand_store = sess.state.read_candidates()
    # Session.domain is the shared snapshot-resolved namespace, so pins are read
    # from the namespace recall/ingest/remember wrote them to.
    pins = sess.state.read_pins(sess.domain)
    elite_ids = [i for i in arc.elite_ids() if i in stored_emb]
    chosen = memory.select_parents(elite_ids, stored_emb, pins, k)
    records = []
    for cid in chosen:
        rec = cand_store.get(cid, {})
        records.append(
            {
                "id": cid,
                "text": rec.get("text", ""),
                "coords": rec.get("coords", {}),
                "niche_id": rec.get("niche_id"),
                "novelty": round(float(rec.get("novelty", 0.0)), 4),
                "pinned": cid in pins,
            }
        )
    return {"parents": records}
