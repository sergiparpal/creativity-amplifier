"""Preference memory + heuristic active learning.

Three jobs:

* **remember** — append the user's A-vs-B comparisons and pins to local memory,
  namespaced per domain so switching domains keeps preferences separate.
* **recall** — summarize that memory for in-context injection next session.
* **active learning** — pick the most *informative* A-vs-B pairs to ask, and
  sample **diverse parents** for the next generation while never dropping a
  pinned stepping stone.

None of this judges novelty; it only reflects the human's revealed preferences
and keeps the search both informed and diverse.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .state import State

# Weights for pair informativeness (similar + novel + undecided == informative).
W_SIM = 0.5
W_UNCERTAIN = 0.3
W_NOVELTY = 0.2


# --------------------------------------------------------------------------- #
# remember
# --------------------------------------------------------------------------- #
def remember(state: State, domain: str, event: Dict[str, Any]) -> Dict[str, Any]:
    """Append a comparison or pin to this domain's memory."""
    if not isinstance(event, dict) or "type" not in event:
        raise ValueError("event must be an object with a 'type'")
    etype = event["type"]
    if etype == "comparison":
        winner, loser = event.get("winner"), event.get("loser")
        if not winner or not loser:
            raise ValueError("comparison event needs 'winner' and 'loser'")
        state.append_comparison(
            domain,
            {
                "type": "comparison",
                "winner": winner,
                "loser": loser,
                "context": event.get("context", ""),
            },
        )
        return {"ok": True, "type": "comparison", "winner": winner, "loser": loser}
    if etype == "pin":
        cid = event.get("id")
        if not cid:
            raise ValueError("pin event needs an 'id'")
        pins = state.add_pin(domain, cid)
        return {"ok": True, "type": "pin", "id": cid, "pins": pins}
    raise ValueError(f"unknown event type {etype!r} (expected comparison|pin)")


# --------------------------------------------------------------------------- #
# recall
# --------------------------------------------------------------------------- #
def recall(state: State, domain: str, k: int = 10) -> Dict[str, Any]:
    """Summarize memory for injection: recent comparisons, pins, win tallies."""
    comparisons = state.read_comparisons(domain)
    pins = state.read_pins(domain)

    wins: Counter = Counter()
    losses: Counter = Counter()
    for ev in comparisons:
        if ev.get("type") == "comparison":
            wins[ev.get("winner")] += 1
            losses[ev.get("loser")] += 1

    # preferred descriptor values, learned from winners (if candidate records exist)
    cand_store = state.read_candidates()
    value_wins: Counter = Counter()
    for ev in comparisons:
        w = ev.get("winner")
        rec = cand_store.get(w) if w else None
        if rec:
            for axis, val in (rec.get("descriptor") or {}).items():
                if isinstance(val, (str, int, bool)):
                    value_wins[f"{axis}={val}"] += 1

    return {
        "domain": domain,
        "preferences": comparisons[-k:],
        "pins": pins,
        "summary": {
            "n_comparisons": len(comparisons),
            "win_counts": dict(wins.most_common(k)),
            "preferred_values": dict(value_wins.most_common(k)),
        },
    }


# --------------------------------------------------------------------------- #
# active learning: which pairs to ask
# --------------------------------------------------------------------------- #
def _compared_set(comparisons: Sequence[Dict[str, Any]]) -> Set[frozenset]:
    out: Set[frozenset] = set()
    for ev in comparisons:
        if ev.get("type") == "comparison":
            out.add(frozenset({ev.get("winner"), ev.get("loser")}))
    return out


def select_ask_pairs(
    slate: List[Dict[str, Any]],
    emb_by_id: Dict[str, Sequence[float]],
    comparisons: Optional[Sequence[Dict[str, Any]]] = None,
    max_pairs: int = 2,
) -> List[List[Any]]:
    """Pick the most-informative A-vs-B pairs from the slate.

    Informativeness rewards pairs that are (a) **similar** in embedding (a fine
    distinction the model is unsure about → max judge-disagreement), (b) of
    **uncertain** relative quality (close fitness), and (c) on the **novel**
    frontier — while skipping pairs the user already decided.
    """
    comparisons = comparisons or []
    decided = _compared_set(comparisons)
    n = len(slate)
    if n < 2:
        return []

    scored: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = slate[i], slate[j]
            ida, idb = a["id"], b["id"]
            if frozenset({ida, idb}) in decided:
                continue  # already resolved by the user
            va = emb_by_id.get(ida)
            vb = emb_by_id.get(idb)
            if va is None or vb is None:
                sim = 0.0
            else:
                sim = float(np.dot(np.asarray(va), np.asarray(vb)))
            uncertainty = 1.0 - abs(
                float(a.get("fitness", 1.0)) - float(b.get("fitness", 1.0))
            )
            mean_nov = 0.5 * (float(a.get("novelty", 0.0)) + float(b.get("novelty", 0.0)))
            score = W_SIM * sim + W_UNCERTAIN * uncertainty + W_NOVELTY * mean_nov
            scored.append((score, i, j))

    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    out: List[List[Any]] = []
    for score, i, j in scored[:max_pairs]:
        out.append(
            [
                slate[i]["id"],
                slate[j]["id"],
                f"informative pair (score {score:.2f}): similar, undecided, on the novel frontier",
            ]
        )
    return out


# --------------------------------------------------------------------------- #
# diverse parents (honoring pins)
# --------------------------------------------------------------------------- #
def select_parents(
    elite_ids: List[str],
    emb_by_id: Dict[str, Sequence[float]],
    pins: List[str],
    k: int,
) -> List[str]:
    """Diverse parent ids for the next generation; pins are ALWAYS included.

    Starts from the pinned stepping stones and greedily adds the elites farthest
    (in cosine distance) from everything chosen so far, until ``k`` is reached.
    Pins are never dropped, even if there are more pins than ``k``.
    """
    # pins first, de-duplicated, order preserved — never dropped
    selected: List[str] = []
    for p in pins:
        if p not in selected:
            selected.append(p)

    pool = [e for e in elite_ids if e not in selected and e in emb_by_id]
    if not pool:
        return selected

    sel_vecs = [np.asarray(emb_by_id[s]) for s in selected if s in emb_by_id]
    pool_vecs = np.asarray([emb_by_id[e] for e in pool], dtype=np.float64)

    if not sel_vecs:
        # no usable seed (e.g. pins lack embeddings): farthest-point fill the rest
        from .diversity import farthest_point_sampling

        remaining = max(0, k - len(selected))
        idx = farthest_point_sampling(pool_vecs, remaining)
        return selected + [pool[i] for i in idx]

    # Farthest-point greedy with an incrementally maintained nearest-selected
    # distance (no full recompute / re-vstack per pick). `alive` masks picks.
    sel_mat = np.vstack([np.asarray(v, dtype=np.float64) for v in sel_vecs])
    min_dist = 1.0 - (pool_vecs @ sel_mat.T).max(axis=1)  # nearest selected -> small
    alive = np.ones(len(pool), dtype=bool)
    while len(selected) < k and alive.any():
        pick = int(np.argmax(np.where(alive, min_dist, -np.inf)))  # farthest from set
        selected.append(pool[pick])
        alive[pick] = False
        # fold the newly chosen vector into the running nearest-selected distance
        min_dist = np.minimum(min_dist, 1.0 - pool_vecs @ pool_vecs[pick])

    return selected
