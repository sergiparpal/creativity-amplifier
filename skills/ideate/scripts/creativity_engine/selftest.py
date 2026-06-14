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

from . import config, diversity, embed, monitor, pipeline
from .archive import compute_niche
from .state import State

# Margins the value gate must clear on the seeded fixture.
MARGIN_MPD = 0.10
MARGIN_VENDI = 0.5

# Generations the diverse loop runs before its slate is measured.
SELFTEST_CYCLES = 2

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
    """Embed + niche a raw candidate list (no DPP, no dedup).

    Reuses the production placement (`pipeline.assign_open_cells`) so the
    baseline is niched exactly like the engine niches its own candidates.
    """
    texts = [c["text"] for c in candidates]
    descriptors = [c["descriptor"] for c in candidates]
    vecs = embedder.embed(texts)
    open_axis, cells = pipeline.assign_open_cells(spec, descriptors, texts, embedder, seed)
    niche_ids = []
    for c, cell in zip(candidates, cells):
        ocell = {open_axis.name: cell} if (open_axis and cell is not None) else {}
        nid, _ = compute_niche(c["descriptor"], spec, ocell)
        niche_ids.append(nid)
    return vecs, niche_ids


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _diverse_engine_metrics(spec, settings, axes, seed, home, project):
    """Run the full engine loop with a stubbed human; return ``(metrics, state)``.

    ``state`` is the diverse project's handle, reused later for the file checks.
    """
    dproj = f"{project}-diverse"
    pipeline.init_project(dproj, axes, seed=seed, home=home)
    state = State(dproj, home=home)

    last = None
    for gen in range(SELFTEST_CYCLES):
        cands = diverse_candidates(settings.candidates_per_generation, gen=gen)
        last = pipeline.ingest(dproj, cands, axes, seed=seed, home=home)
        _stub_human(pipeline, dproj, last, home)

    slate = last["slate"] if last else []
    emb_store = state.read_embeddings()
    ids = [s["id"] for s in slate]
    vecs = np.asarray([emb_store[i] for i in ids if i in emb_store], dtype=np.float32)
    return _slate_diversity(vecs, [s["niche_id"] for s in slate]), state


def _single_shot_metrics(spec, settings, embedder, seed):
    """Clichéd single-shot baseline: a clustered, low-diversity slate."""
    cands = single_shot_candidates(settings.candidates_per_generation)
    vecs, niches = _place(cands, spec, embedder, seed)
    return _slate_diversity(vecs[: spec.slate_size], niches[: spec.slate_size])


def _dpp_isolation_metrics(spec, settings, embedder, seed):
    """DPP vs first-N on one shared pool — isolates the engine's selection step."""
    cands = dpp_isolation_candidates(settings.candidates_per_generation, spec.slate_size)
    vecs, niches = _place(cands, spec, embedder, seed)
    sel = diversity.select_diverse(vecs, k=spec.slate_size, seed=seed)
    dpp = _slate_diversity(vecs[sel], [niches[i] for i in sel])
    first_n = _slate_diversity(vecs[: spec.slate_size], niches[: spec.slate_size])
    return dpp, first_n


def _collapse_reversal(spec, settings, axes, seed, home, project):
    """A samey generation must trip the monitor; the next diverse one recovers."""
    cproj = f"{project}-collapse"
    pipeline.init_project(cproj, axes, seed=seed, home=home)
    collapsed = pipeline.ingest(
        cproj, collapsing_candidates(8), axes, seed=seed, home=home
    )
    recovered = pipeline.ingest(
        cproj,
        diverse_candidates(settings.candidates_per_generation, gen=5, prefix="rec"),
        axes, seed=seed, home=home,
    )
    col_mon, rec_mon = collapsed["monitor"], recovered["monitor"]
    checks = {
        "collapse_detected": bool(col_mon["collapsing"]),
        "recovered_quiet": not rec_mon["collapsing"],
        "diversity_recovered": rec_mon["mean_cosine"] < col_mon["mean_cosine"],
    }
    return {
        "collapsed_monitor": col_mon,
        "recovered_monitor": rec_mon,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run(project: str = "selftest", live: bool = False, seed: int = 0,
        home: Optional[Path] = None) -> Dict[str, Any]:
    # deterministic embedder unless a live run is explicitly requested
    os.environ["CREATIVITY_EMBEDDER"] = "local" if live else "hash"
    embed.reset_cache()

    spec = config.load_generic_axes()
    settings = config.load_session_settings(config.generic_axes_path())
    axes = spec.to_dict()
    embedder = embed.get_embedder()

    engine_metrics, state = _diverse_engine_metrics(
        spec, settings, axes, seed, home, project
    )
    base_metrics = _single_shot_metrics(spec, settings, embedder, seed)
    dpp_metrics, firstn_metrics = _dpp_isolation_metrics(
        spec, settings, embedder, seed
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

    reversal = _collapse_reversal(spec, settings, axes, seed, home, project)

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
        "cycles": SELFTEST_CYCLES,
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
