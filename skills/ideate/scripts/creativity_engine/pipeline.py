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

from . import __version__, archive as archive_mod, config, diversity, monitor, novelty
from .config import AxesSpec, Candidate
from .embed import dedupe, get_embedder
from .state import State

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
    """Create state dirs and snapshot the resolved axes for the session."""
    spec = config.load_axes(axes_source)
    state = State(project, home=home).ensure()
    state.write_axes(spec.to_dict())
    meta = state.read_meta()
    meta.update(
        {
            "project": project,
            "domain": spec.domain,
            "unit_of_generation": spec.unit_of_generation,
            "seed": int(seed),
            "version": __version__,
        }
    )
    state.write_meta(meta)
    return {"ok": True, "domain": spec.domain, "paths": state.paths()}


def _resolve_domain(state: State) -> str:
    axes = state.read_axes()
    if axes and isinstance(axes, dict):
        return str(axes.get("domain", "default"))
    return "default"


def _load_spec(state: State, fallback: Optional[AxesSpec] = None) -> AxesSpec:
    axes = state.read_axes()
    if axes:
        return config.axes_spec_from_dict(axes)
    if fallback is not None:
        return fallback
    raise config.ConfigError(
        f"no axes snapshot for project {state.project!r}; run init-project first"
    )


# --------------------------------------------------------------------------- #
# recall
# --------------------------------------------------------------------------- #
def recall(project: str, k: int = 10, home: Optional[Path] = None) -> Dict[str, Any]:
    """Return memory for in-context injection (enriched by memory.py in Phase 5)."""
    state = State(project, home=home)
    domain = _resolve_domain(state)
    try:
        from . import memory  # Phase 5; optional during early phases

        return memory.recall(state, domain, k=k)
    except ImportError:
        comparisons = state.read_comparisons(domain)
        pins = state.read_pins(domain)
        return {"domain": domain, "preferences": comparisons[-k:], "pins": pins}


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
    for i in range(n):
        dist[i, offset + i] = np.inf
    m_eff = ref.shape[0] - 1
    if m_eff <= 0:
        return np.ones((n,), dtype=np.float32)
    kk = min(k, m_eff)
    part = np.partition(dist, kk - 1, axis=1)[:, :kk]
    return np.clip(part.mean(axis=1), 0.0, 2.0).astype(np.float32)


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
        "embedding_ref": record["id"],
    }


def ingest(
    project: str,
    candidates,
    axes_source,
    seed: int = 0,
    home: Optional[Path] = None,
) -> Dict[str, Any]:
    """Embed → dedup → place → novelty → archive → DPP → monitor for one cycle."""
    spec = config.load_axes(axes_source)
    state = State(project, home=home).ensure()
    if state.read_axes() is None:
        state.write_axes(spec.to_dict())

    cand_list = _parse_candidates(candidates)
    arc = archive_mod.Archive.from_dict(spec, state.read_archive())
    stored_emb: Dict[str, List[float]] = state.read_embeddings()
    cand_store: Dict[str, Any] = state.read_candidates()

    if not cand_list:
        return {
            "slate": [],
            "ask_pairs": [],
            "monitor": monitor.evaluate(np.zeros((0, 1)), arc.niche_counts()),
            "parents": [],
        }

    embedder = get_embedder()
    vecs = embedder.embed([c.text for c in cand_list])

    # existing embeddings (archive elites) for dedup + novelty reference
    existing_ids = [eid for eid in arc.elite_ids() if eid in stored_emb]
    if existing_ids:
        existing_vecs = np.asarray(
            [stored_emb[i] for i in existing_ids], dtype=np.float32
        )
    else:
        existing_vecs = np.zeros((0, vecs.shape[1]), dtype=np.float32)

    keep, _drop = dedupe(
        vecs, tau=DEDUP_TAU, existing=existing_vecs if existing_vecs.shape[0] else None
    )
    survivors = [cand_list[i] for i in keep]
    surv_vecs = vecs[keep] if keep else np.zeros((0, vecs.shape[1]), dtype=np.float32)

    # open-axis CVT cells
    open_axis = spec.primary_axis
    cells: List[Optional[int]] = [None] * len(survivors)
    if open_axis is not None and survivors:
        open_texts = [
            str(c.descriptor.get(open_axis.name) or c.text) for c in survivors
        ]
        open_vecs = embedder.embed(open_texts)
        nicher = archive_mod.CVTNicher(
            dim=open_vecs.shape[1], k=OPEN_NICHES, seed=seed
        )
        cells = nicher.cells(open_vecs)  # type: ignore[assignment]

    novelties = _survivor_novelty(surv_vecs, existing_vecs, KNN_K)

    # place survivors
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

    # DPP diverse slate over current elites
    elites: List[Tuple[str, float, str]] = []  # (id, fitness, niche)
    for nid, nf in arc.niches.items():
        if nf.elite_id and nf.elite_id in stored_emb:
            elites.append((nf.elite_id, nf.fitness, nid))
    # cap the pool by novelty for latency
    if len(elites) > MAX_DPP_POOL:
        elites.sort(
            key=lambda e: cand_store.get(e[0], {}).get("novelty", 0.0), reverse=True
        )
        elites = elites[:MAX_DPP_POOL]

    slate_ids: List[str] = []
    slate: List[Dict[str, Any]] = []
    if elites:
        elite_ids = [e[0] for e in elites]
        elite_vecs = np.asarray([stored_emb[i] for i in elite_ids], dtype=np.float32)
        quality = np.asarray([e[1] for e in elites], dtype=np.float64)
        sel = diversity.select_diverse(
            elite_vecs, k=spec.slate_size, quality=quality, seed=seed
        )
        slate_ids = [elite_ids[i] for i in sel]
        slate = [_slate_item(cand_store[i]) for i in slate_ids]

    mon = monitor.evaluate(surv_vecs, arc.niche_counts())

    from . import memory

    comparisons = state.read_comparisons(spec.domain)
    ask_pairs = memory.select_ask_pairs(slate, stored_emb, comparisons, max_pairs=2)

    # persist
    state.write_archive(arc.to_dict())
    state.write_embeddings(stored_emb)
    state.write_candidates(cand_store)
    meta = state.read_meta()
    meta["cycles"] = int(meta.get("cycles", 0)) + 1
    state.write_meta(meta)

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
    state = State(project, home=home)
    spec = _load_spec(state, fallback=config.load_generic_axes())
    arc = archive_mod.Archive.from_dict(spec, state.read_archive())
    stored_emb = state.read_embeddings()
    elite_ids = [i for i in arc.elite_ids() if i in stored_emb]
    if elite_ids:
        elite_vecs = np.asarray([stored_emb[i] for i in elite_ids], dtype=np.float32)
    else:
        elite_vecs = np.zeros((0, 1), dtype=np.float32)
    mon = monitor.evaluate(elite_vecs, arc.niche_counts())
    return {
        "entropy": mon["entropy"],
        "mean_cosine": mon["mean_cosine"],
        "coverage": mon["coverage"],
        "n": len(elite_ids),
    }


# --------------------------------------------------------------------------- #
# Commands implemented in later phases (declared here so dispatch is explicit)
# --------------------------------------------------------------------------- #
def remember(project: str, event: Dict[str, Any],
             home: Optional[Path] = None) -> Dict[str, Any]:
    """Append a comparison/pin to this domain's preference memory."""
    from . import memory

    state = State(project, home=home).ensure()
    domain = _resolve_domain(state)
    return memory.remember(state, domain, event)


def parents(project: str, k: int = 4, seed: int = 0,
            home: Optional[Path] = None) -> Dict[str, Any]:
    """Diverse parents for the next generation; pinned stepping stones kept."""
    from . import memory

    state = State(project, home=home)
    spec = _load_spec(state, fallback=config.load_generic_axes())
    arc = archive_mod.Archive.from_dict(spec, state.read_archive())
    stored_emb = state.read_embeddings()
    cand_store = state.read_candidates()
    pins = state.read_pins(spec.domain)
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


def selftest(project: str = "selftest", live: bool = False, seed: int = 0,
             home: Optional[Path] = None) -> Dict[str, Any]:
    """Full loop with a stubbed LLM + human, plus value gate & collapse reversal."""
    from . import selftest as _selftest

    return _selftest.run(project=project, live=live, seed=seed, home=home)
