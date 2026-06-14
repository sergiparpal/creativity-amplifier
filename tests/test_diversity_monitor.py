"""Phase 3: DPP beats random on diversity; the anti-collapse monitor fires on a
near-duplicate stream and stays quiet on a diverse one."""

from __future__ import annotations

import numpy as np

from creativity_engine import diversity, monitor


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / (n if n else 1.0)


def _clustered_pool(groups=5, per=6, dim=12, noise=0.01, seed=0):
    """A pool of ``groups*per`` points: ``groups`` distinct directions, each
    replicated ``per`` times with tiny noise (so duplicates abound)."""
    rng = np.random.default_rng(seed)
    dirs = [_unit(rng.standard_normal(dim)) for _ in range(groups)]
    pool = []
    for d in dirs:
        for _ in range(per):
            pool.append(_unit(d + noise * rng.standard_normal(dim)))
    return np.array(pool)


# --------------------------------------------------------------------------- #
# DPP vs random
# --------------------------------------------------------------------------- #
def test_dpp_beats_random_on_diversity():
    pool = _clustered_pool(groups=5, per=6, seed=1)
    k = 5

    sel = diversity.select_diverse(pool, k=k)
    dpp_vecs = pool[sel]
    dpp_mpd = diversity.mean_pairwise_distance(dpp_vecs)
    dpp_vendi = diversity.vendi_score(dpp_vecs)

    rng = np.random.default_rng(7)
    rand_mpd, rand_vendi = [], []
    for _ in range(200):
        idx = rng.choice(len(pool), size=k, replace=False)
        rand_mpd.append(diversity.mean_pairwise_distance(pool[idx]))
        rand_vendi.append(diversity.vendi_score(pool[idx]))

    assert dpp_mpd > np.mean(rand_mpd) + 0.05
    assert dpp_vendi > np.mean(rand_vendi) + 0.3
    # DPP should recover ~one-per-group => far above the random Vendi baseline
    assert dpp_vendi > 3.5


def test_dpp_picks_across_groups():
    pool = _clustered_pool(groups=5, per=6, seed=2)
    sel = diversity.select_diverse(pool, k=5)
    # the 5 picks should be mutually near-orthogonal (one per group)
    v = pool[sel]
    sims = v @ v.T
    off = sims[np.triu_indices(5, k=1)]
    assert np.max(off) < 0.5


def test_select_diverse_handles_small_pool():
    pool = _clustered_pool(groups=2, per=1, seed=0)
    assert sorted(diversity.select_diverse(pool, k=10)) == [0, 1]
    assert diversity.select_diverse(np.zeros((0, 4)), k=3) == []


def test_vendi_bounds():
    one = _unit(np.eye(6)[0]).reshape(1, -1)
    assert abs(diversity.vendi_score(one) - 1.0) < 1e-6
    dup = np.repeat(one, 5, axis=0)
    assert diversity.vendi_score(dup) < 1.01  # all identical => ~1
    orth = np.eye(6)
    assert diversity.vendi_score(orth) > 5.5   # ~6 distinct


def test_farthest_point_fallback_runs():
    pool = _clustered_pool(groups=4, per=3, seed=3)
    sel = diversity.farthest_point_sampling(pool, k=4)
    assert len(sel) == 4
    assert len(set(sel)) == 4


# --------------------------------------------------------------------------- #
# Monitor
# --------------------------------------------------------------------------- #
def test_monitor_flags_near_duplicate_stream():
    rng = np.random.default_rng(0)
    base = _unit(rng.standard_normal(12))
    stream = np.array([_unit(base + 0.005 * rng.standard_normal(12)) for _ in range(8)])
    res = monitor.evaluate(stream, niche_counts=[8])
    assert res["collapsing"] is True
    assert res["mean_cosine"] > 0.9


def test_monitor_quiet_on_diverse_stream():
    pool = _clustered_pool(groups=6, per=1, seed=4)  # 6 distinct directions
    res = monitor.evaluate(pool, niche_counts=[3, 3, 3, 3, 3, 3])
    assert res["collapsing"] is False
    assert res["mean_cosine"] < 0.5
    assert res["normalized_entropy"] > 0.9


def test_monitor_flags_concentrated_occupancy():
    pool = _clustered_pool(groups=6, per=1, seed=5)  # diverse vectors...
    # ...but occupancy piled into one niche of many => low entropy
    res = monitor.evaluate(pool, niche_counts=[100, 1, 1, 1])
    assert res["collapsing"] is True


def test_entropy_helpers():
    assert monitor.shannon_entropy([1, 1, 1, 1]) > monitor.shannon_entropy([10, 1])
    assert abs(monitor.normalized_entropy([5, 5, 5, 5]) - 1.0) < 1e-9
    assert monitor.normalized_entropy([7]) == 0.0


# --------------------------------------------------------------------------- #
# Calibrated (relative) similarity flag
# --------------------------------------------------------------------------- #
def _pair(cos: float) -> np.ndarray:
    """Two unit vectors whose only pairwise cosine is exactly ``cos``."""
    return np.array([[1.0, 0.0], [cos, np.sqrt(1.0 - cos * cos)]], dtype=np.float64)


def test_relative_flag_fires_below_absolute_threshold():
    # mean cosine 0.45 is BELOW the old absolute 0.55, so without a baseline the
    # monitor stays quiet...
    vecs = _pair(0.45)
    quiet = monitor.evaluate(vecs, niche_counts=[5, 5, 5])
    assert quiet["collapsing"] is False
    assert quiet["mean_cosine"] < 0.55
    # ...but relative to a diverse rolling baseline (~0.20) it is clearly samey.
    flagged = monitor.evaluate(vecs, niche_counts=[5, 5, 5], baseline=[0.2, 0.2, 0.2])
    assert flagged["collapsing"] is True
    assert flagged["mean_cosine"] < 0.55  # the absolute rule alone would miss it
    assert any("baseline" in r for r in flagged["reasons"])


def test_absolute_until_enough_baseline_samples():
    vecs = _pair(0.45)
    # one sample is below DEFAULT_MIN_BASELINE -> fall back to the absolute rule
    res = monitor.evaluate(vecs, niche_counts=[5, 5, 5], baseline=[0.2])
    assert res["collapsing"] is False
    assert res["baseline_n"] == 1


def test_absolute_safety_ceiling_catches_high_baseline():
    # An already-elevated baseline would push baseline+margin to 0.85, but the
    # absolute ceiling (0.80) still catches a genuinely collapsed generation.
    over = monitor.evaluate(_pair(0.82), niche_counts=[5, 5, 5], baseline=[0.7, 0.7])
    assert over["collapsing"] is True
    assert over["cos_limit"] == 0.80
    # just under the ceiling and under baseline+margin -> not flagged
    under = monitor.evaluate(_pair(0.78), niche_counts=[5, 5, 5], baseline=[0.7, 0.7])
    assert under["collapsing"] is False
