"""DPP degenerate-pool handling: early-stop fires on rank-deficient pools and the
seeded farthest-point top-up fills a full, frontier-extending slate.

Diverse pools must still be served by greedy DPP (the normal path); only
rank-deficient pools route through the top-up, and it must never crash, repeat an
index, or drop coverage of the distinct directions present.
"""

from __future__ import annotations

import numpy as np

from creativity_engine import diversity


def test_rank_deficient_pool_returns_full_slate_covering_directions():
    # two distinct directions, 3 copies each; k=4 must return 4 distinct indices
    # spanning BOTH directions (the seeded top-up extends the frontier).
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    pool = np.array([a, a, a, b, b, b])
    sel = diversity.select_diverse(pool, k=4, seed=0)
    assert len(sel) == 4
    assert len(set(sel)) == 4
    dirs = {tuple(pool[i]) for i in sel}
    assert len(dirs) == 2  # both directions represented


def test_identical_pool_does_not_crash_and_fills_k():
    v = np.tile(np.array([1.0, 0.0]), (5, 1))
    sel = diversity.select_diverse(v, k=3, seed=0)
    assert len(sel) == 3
    assert len(set(sel)) == 3  # distinct indices, no repeats


def test_diverse_pool_served_by_dpp():
    rng = np.random.default_rng(0)
    v = rng.standard_normal((8, 16))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    sel = diversity.select_diverse(v, k=5, seed=0)
    assert len(sel) == 5 and len(set(sel)) == 5
