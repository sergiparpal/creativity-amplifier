"""End-to-end self-test with a stubbed LLM and a stubbed human.

No interactive input, no live model. It exercises the whole loop on a
domain-neutral brief + generic axes and then proves two things:

* **Value gate** — the engine's diverse slate beats a single-shot baseline on
  mean pairwise distance, Vendi score, and niche entropy; and DPP selection beats
  naive first-N on the *same* pool (isolating the engine's contribution).
* **Induced-collapse reversal** — a samey generation trips the monitor, and once
  diversity pressure is raised the next generation recovers.

The stubbed LLM is a canned candidate generator; the stubbed human auto-picks the
highest-novelty idea in each slate.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import config, diversity, embed, monitor
from .archive import CVTNicher, compute_niche
from .state import State

# Margins the value gate must clear on the seeded fixture.
MARGIN_MPD = 0.10
MARGIN_VENDI = 0.5

_ANGLES = ["growth", "art", "community", "tech", "ritual", "play", "learning", "ecology"]
_SCOPES = ["personal", "local", "regional", "global"]
_FORMS = ["event", "tool", "story", "game", "service", "installation", "campaign", "wager"]
_MECHANISMS = [
    "referral incentives", "immersive spectacle", "interactive branching",
    "shared private audio", "time-limited scarcity", "peer teaching",
    "data sonification", "ambient signage", "gamified streaks",
    "cooperative puzzle", "reverse auction", "mycelial relay",
]
_NOUNS = [
    "lantern", "tide", "compass", "ember", "orchard", "signal",
    "mosaic", "beacon", "ledger", "kite", "prism", "anvil",
]
_OPERATORS = ["mutation", "analogy", "reframe", "scamper", "biomimicry", "inversion"]


# --------------------------------------------------------------------------- #
# Canned generators (the "stubbed LLM")
# --------------------------------------------------------------------------- #
def diverse_candidates(n: int, gen: int = 0, prefix: str = "d") -> List[Dict[str, Any]]:
    """Vary every axis widely -> well-spread embeddings and niches."""
    out = []
    for i in range(n):
        a = _ANGLES[i % len(_ANGLES)]
        s = _SCOPES[(i * 3 + 1) % len(_SCOPES)]
        f = _FORMS[(i * 5 + 2) % len(_FORMS)]
        m = _MECHANISMS[(i * 7 + gen) % len(_MECHANISMS)]
        noun = _NOUNS[(i * 11 + 3) % len(_NOUNS)]
        bold = round((i % 5) / 4.0, 2)
        out.append(
            {
                "id": f"{prefix}{gen}-{i}",
                "text": f"A {f} for {s} {a}: {m} using a {noun}",
                "descriptor": {
                    "angle": a, "scope": s, "form": f,
                    "boldness": bold, "mechanism": m,
                },
                "genealogy": {"operator_id": _OPERATORS[i % len(_OPERATORS)]},
            }
        )
    return out


def single_shot_candidates(n: int, prefix: str = "s") -> List[Dict[str, Any]]:
    """A clichéd single shot: one angle/form/mechanism, minor wording -> clustered."""
    verbs = ["boost", "increase", "drive", "grow", "raise", "lift", "improve", "scale"]
    out = []
    for i in range(n):
        verb = verbs[i % len(verbs)]
        noun = _NOUNS[i % 2]  # only two nouns -> very samey
        out.append(
            {
                "id": f"{prefix}-{i}",
                "text": f"A campaign to {verb} growth with referral incentives and a {noun}",
                "descriptor": {
                    "angle": "growth", "scope": "global", "form": "campaign",
                    "boldness": 0.3, "mechanism": "referral incentives",
                },
                "genealogy": {"operator_id": "single_shot"},
            }
        )
    return out


def collapsing_candidates(n: int, prefix: str = "x") -> List[Dict[str, Any]]:
    """Samey but not exact-duplicate: comfortably survives dedup (cos < 0.92) yet
    trips the monitor (cos > 0.55). Two words vary to sit mid-band."""
    nouns = ["birds", "trees", "rivers", "gardens", "markets", "bridges", "murals", "cafes"]
    adjs = ["vibrant", "quiet", "historic", "hidden", "bustling", "sleepy", "colorful", "windswept"]
    out = []
    for i in range(n):
        noun = nouns[i % len(nouns)]
        adj = adjs[i % len(adjs)]
        out.append(
            {
                "id": f"{prefix}-{i}",
                "text": f"A neighborhood mural festival celebrating {adj} local {noun} every weekend",
                "descriptor": {
                    "angle": "community", "scope": "local", "form": "event",
                    "boldness": 0.4, "mechanism": "immersive spectacle",
                },
                "genealogy": {"operator_id": "mutation"},
            }
        )
    return out


def dpp_isolation_candidates(n: int, slate_size: int) -> List[Dict[str, Any]]:
    """A pool whose first ``slate_size`` items are near-clones (so naive first-N
    is low-diversity) followed by diverse items (so DPP can find spread). Isolates
    the engine's selection value from the generator's."""
    clones = collapsing_candidates(slate_size, prefix="clone")
    rest = diverse_candidates(max(0, n - slate_size), gen=2, prefix="rest")
    return clones + rest


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #
def _slate_diversity(vecs: np.ndarray, niche_ids: List[str]) -> Dict[str, float]:
    counts = Counter(niche_ids)
    return {
        "mean_pairwise_distance": round(diversity.mean_pairwise_distance(vecs), 4),
        "vendi": round(diversity.vendi_score(vecs), 4),
        "niche_entropy": round(monitor.shannon_entropy(list(counts.values())), 4),
        "coverage": len(counts),
        "n": int(vecs.shape[0]),
    }


def _place(candidates, spec, embedder, seed):
    """Embed + niche a raw candidate list (no DPP, no dedup)."""
    texts = [c["text"] for c in candidates]
    vecs = embedder.embed(texts)
    open_axis = spec.primary_axis
    niche_ids = []
    if open_axis is not None:
        open_texts = [str(c["descriptor"].get(open_axis.name) or c["text"]) for c in candidates]
        open_vecs = embedder.embed(open_texts)
        nicher = CVTNicher(dim=open_vecs.shape[1], k=16, seed=seed)
        cells = nicher.cells(open_vecs)
    else:
        cells = [None] * len(candidates)
    for c, cell in zip(candidates, cells):
        ocell = {open_axis.name: cell} if (open_axis and cell is not None) else {}
        nid, _ = compute_niche(c["descriptor"], spec, ocell)
        niche_ids.append(nid)
    return vecs, niche_ids


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(project: str = "selftest", live: bool = False, seed: int = 0,
        home: Optional[Path] = None) -> Dict[str, Any]:
    # deterministic embedder unless a live run is explicitly requested
    os.environ["CREATIVITY_EMBEDDER"] = "local" if live else "hash"
    embed.reset_cache()

    from . import pipeline  # local import to avoid a cycle at module load

    spec = config.load_generic_axes()
    axes = spec.to_dict()
    embedder = embed.get_embedder()

    # ----- diverse loop with stubbed human ------------------------------- #
    dproj = f"{project}-diverse"
    pipeline.init_project(dproj, axes, seed=seed, home=home)
    state = State(dproj, home=home)

    last = None
    cycles = 2
    for gen in range(cycles):
        cands = diverse_candidates(spec.candidates_per_generation, gen=gen)
        last = pipeline.ingest(dproj, cands, axes, seed=seed, home=home)
        _stub_human(pipeline, dproj, last, home)

    engine_slate = last["slate"] if last else []
    emb_store = state.read_embeddings()
    eng_ids = [s["id"] for s in engine_slate]
    eng_vecs = np.asarray([emb_store[i] for i in eng_ids if i in emb_store], dtype=np.float32)
    eng_niches = [s["niche_id"] for s in engine_slate]
    engine_metrics = _slate_diversity(eng_vecs, eng_niches)

    # ----- single-shot baseline ------------------------------------------ #
    base_cands = single_shot_candidates(spec.candidates_per_generation)
    base_vecs, base_niches = _place(base_cands, spec, embedder, seed)
    base_slate = base_vecs[: spec.slate_size]
    base_metrics = _slate_diversity(base_slate, base_niches[: spec.slate_size])

    # ----- DPP-vs-first-N on the SAME pool (isolates the engine's selection) #
    pool_cands = dpp_isolation_candidates(
        spec.candidates_per_generation, spec.slate_size
    )
    pool_vecs, pool_niches = _place(pool_cands, spec, embedder, seed)
    sel = diversity.select_diverse(pool_vecs, k=spec.slate_size, seed=seed)
    dpp_metrics = _slate_diversity(
        pool_vecs[sel], [pool_niches[i] for i in sel]
    )
    firstn_metrics = _slate_diversity(
        pool_vecs[: spec.slate_size], pool_niches[: spec.slate_size]
    )

    value_gate = {
        "engine": engine_metrics,
        "single_shot": base_metrics,
        "dpp_on_pool": dpp_metrics,
        "first_n_on_pool": firstn_metrics,
        "checks": {
            "mpd_beats_single_shot": engine_metrics["mean_pairwise_distance"]
            > base_metrics["mean_pairwise_distance"] + MARGIN_MPD,
            "vendi_beats_single_shot": engine_metrics["vendi"]
            > base_metrics["vendi"] + MARGIN_VENDI,
            "entropy_beats_single_shot": engine_metrics["niche_entropy"]
            > base_metrics["niche_entropy"],
            "dpp_beats_first_n": dpp_metrics["mean_pairwise_distance"]
            > firstn_metrics["mean_pairwise_distance"],
        },
    }
    value_gate["passed"] = all(value_gate["checks"].values())

    # ----- induced-collapse reversal ------------------------------------- #
    cproj = f"{project}-collapse"
    pipeline.init_project(cproj, axes, seed=seed, home=home)
    collapsed = pipeline.ingest(
        cproj, collapsing_candidates(8), axes, seed=seed, home=home
    )
    recovered = pipeline.ingest(
        cproj, diverse_candidates(spec.candidates_per_generation, gen=5, prefix="rec"),
        axes, seed=seed, home=home,
    )
    col_mon, rec_mon = collapsed["monitor"], recovered["monitor"]
    reversal = {
        "collapsed_monitor": col_mon,
        "recovered_monitor": rec_mon,
        "checks": {
            "collapse_detected": bool(col_mon["collapsing"]),
            "recovered_quiet": not rec_mon["collapsing"],
            "diversity_recovered": rec_mon["mean_cosine"] < col_mon["mean_cosine"],
        },
    }
    reversal["passed"] = all(reversal["checks"].values())

    # ----- state files written ------------------------------------------- #
    written = {
        name: Path(p).exists() for name, p in state.paths().items() if name != "root"
    }
    files_ok = all(written.values())

    ok = bool(value_gate["passed"] and reversal["passed"] and files_ok)
    return {
        "ok": ok,
        "value_gate": value_gate,
        "collapse_reversal": reversal,
        "state_files_written": written,
        "cycles": cycles,
        "embedder": "local" if live else "hash",
    }


def _stub_human(pipeline, project, result, home) -> None:
    """Auto-pick the highest-novelty idea; record a comparison and a pin."""
    slate = result.get("slate", [])
    if len(slate) < 2:
        return
    ranked = sorted(slate, key=lambda s: s.get("novelty", 0.0))
    winner, loser = ranked[-1]["id"], ranked[0]["id"]
    pipeline.remember(
        project, {"type": "comparison", "winner": winner, "loser": loser}, home=home
    )
    pipeline.remember(project, {"type": "pin", "id": winner}, home=home)
