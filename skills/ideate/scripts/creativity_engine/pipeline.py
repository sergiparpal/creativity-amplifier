"""High-level orchestration for the CLI commands.

Each public function returns a JSON-serializable dict (or raises). The CLI in
``__main__`` is a thin wrapper that parses args, calls these, and prints JSON.

The ``ingest`` flow is the heart of one cycle:
embed → dedup → place (MAP-Elites over the resolved axes) → geometric novelty →
DPP diverse slate → anti-collapse monitor. The judge is never called here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
from .embed import dedupe
from .session import Session

# Tuning knobs (kept modest so first results stay quick — see plan §10 latency).
DEDUP_TAU = 0.92
KNN_K = 5
OPEN_NICHES = 16
MAX_DPP_POOL = 200


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


def assign_open_cells(
    spec: AxesSpec,
    descriptors: List[Dict[str, Any]],
    texts: List[str],
    embedder,
    seed: int,
) -> Tuple[Optional[Any], List[Optional[int]]]:
    """CVT (Voronoi) cell per item for the primary "open" axis.

    Returns ``(open_axis, cells)`` where ``cells[i]`` is the item's cell index
    (or ``None`` when there is no open axis). Shared by :func:`ingest` and the
    self-test so both place candidates through identical logic.
    """
    open_axis = spec.primary_axis
    n = len(texts)
    if open_axis is None or n == 0:
        return open_axis, [None] * n
    open_texts = [
        str(d.get(open_axis.name) or t) for d, t in zip(descriptors, texts)
    ]
    open_vecs = embedder.embed(open_texts)
    nicher = archive_mod.CVTNicher(dim=open_vecs.shape[1], k=OPEN_NICHES, seed=seed)
    return open_axis, nicher.cells(open_vecs)


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
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """DPP diverse slate over the current niche elites. Returns ``(slate, ids)``."""
    elites: List[Tuple[str, float, str]] = [
        (niche.elite_id, niche.fitness, nid)
        for nid, niche in arc.niches.items()
        if niche.elite_id and niche.elite_id in stored_emb
    ]
    # cap the pool by novelty for latency
    if len(elites) > MAX_DPP_POOL:
        elites.sort(
            key=lambda e: cand_store.get(e[0], {}).get("novelty", 0.0), reverse=True
        )
        elites = elites[:MAX_DPP_POOL]
    if not elites:
        return [], []

    elite_ids = [e[0] for e in elites]
    elite_vecs = np.asarray([stored_emb[i] for i in elite_ids], dtype=np.float32)
    quality = np.asarray([e[1] for e in elites], dtype=np.float64)
    sel = diversity.select_diverse(
        elite_vecs, k=spec.slate_size, quality=quality, seed=seed
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
) -> None:
    """Write archive/embeddings/candidates and bump the cycle metadata."""
    state.write_archive(arc.to_dict())
    state.write_embeddings(stored_emb)
    state.write_candidates(cand_store)
    meta = state.read_meta()
    meta["cycles"] = int(meta.get("cycles", 0)) + 1
    meta["embedder"] = embedder.name
    meta["embedding_dim"] = int(vecs.shape[1])
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

    # Existing archive elites seed both dedup and the novelty reference.
    existing_ids = [eid for eid in arc.elite_ids() if eid in stored_emb]
    existing_vecs = _stack_embeddings(existing_ids, stored_emb, vecs.shape[1])

    keep, _drop = dedupe(
        vecs, tau=DEDUP_TAU, existing=existing_vecs if existing_vecs.shape[0] else None
    )
    survivors = [cand_list[i] for i in keep]
    surv_vecs = vecs[keep] if keep else np.zeros((0, vecs.shape[1]), dtype=np.float32)

    open_axis, cells = assign_open_cells(
        spec, [c.descriptor for c in survivors], [c.text for c in survivors],
        embedder, seed,
    )
    novelties = _survivor_novelty(surv_vecs, existing_vecs, KNN_K)
    _place_survivors(
        survivors, surv_vecs, cells, novelties, open_axis,
        spec, arc, cand_store, stored_emb,
    )

    slate, slate_ids = _select_slate(arc, stored_emb, cand_store, spec, seed)

    # Monitor the RAW generation (pre-dedup) so a near-duplicate batch still
    # registers as collapsing — dedup would otherwise hide it behind survivors.
    mon = monitor.evaluate(vecs, arc.niche_counts())

    # Namespace preference memory by the session domain so ingest is consistent
    # with remember/recall/parents (all share Session's snapshot resolution).
    domain = sess.domain
    comparisons = state.read_comparisons(domain)
    ask_pairs = memory.select_ask_pairs(slate, stored_emb, comparisons, max_pairs=2)

    _persist_cycle(state, arc, stored_emb, cand_store, vecs, embedder)
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
    elite_vecs = _stack_embeddings(elite_ids, stored_emb, dim=1)
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
