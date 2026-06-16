"""Engine tuning lifted to config: defaults reproduce behavior; per-domain overrides.

The knobs (dedup tau, KNN k, open-niche count, DPP pool, monitor thresholds …)
used to be scattered module constants. They now live in ``EngineConfig`` with an
optional per-domain ``engine:`` block; the defaults must reproduce the original
behavior and modules must read from the resolved config.
"""

from __future__ import annotations

import numpy as np
import pytest

from creativity_engine import config, embed, monitor, pipeline
from creativity_engine.config import EngineConfig
from creativity_engine.state import State


def test_defaults_match_module_constants():
    # Drift guard: EngineConfig defaults mirror the module-level fallback constants.
    c = EngineConfig()
    assert c.open_niches == pipeline.OPEN_NICHES
    assert c.open_niche_freeze_factor == pipeline.OPEN_NICHE_FREEZE_FACTOR
    assert c.knn_k == pipeline.KNN_K
    assert c.novelty_ref_cap == pipeline.NOVELTY_REF_CAP
    assert c.max_dpp_pool == pipeline.MAX_DPP_POOL
    assert c.quality_weight == pipeline.QUALITY_WEIGHT
    assert c.monitor_cos_threshold == monitor.DEFAULT_COS_THRESHOLD
    assert c.monitor_entropy_threshold == monitor.DEFAULT_ENTROPY_THRESHOLD
    assert c.monitor_margin == monitor.DEFAULT_MARGIN
    assert c.monitor_cos_ceiling == monitor.DEFAULT_COS_CEILING
    assert c.monitor_window == 5
    assert c.monitor_min_baseline == monitor.DEFAULT_MIN_BASELINE
    # dedup tau defaults to per-embedder (None), which for hash is the global default
    assert c.dedup_tau is None
    assert embed.default_dedup_tau("hash") == embed.DEFAULT_DEDUP_TAU


def test_load_defaults_and_overrides():
    assert config.load_engine_config({}) == EngineConfig()
    cfg = config.load_engine_config(
        {"engine": {"open_niches": 8, "quality_weight": 0.0, "dedup_tau": 0.8}}
    )
    assert cfg.open_niches == 8
    assert cfg.quality_weight == 0.0
    assert cfg.dedup_tau == 0.8
    # unspecified keys keep their defaults
    assert cfg.knn_k == EngineConfig().knn_k
    # prefilter-guard ratio: default and override
    assert EngineConfig().under_generation_ratio == 0.6
    assert config.load_engine_config(
        {"engine": {"under_generation_ratio": 0.25}}
    ).under_generation_ratio == 0.25


def test_example_domain_engine_overrides():
    path = config.generic_axes_path().parent / "examples" / "research_hypotheses.yaml"
    cfg = config.load_engine_config(path)
    assert cfg.open_niches == 32
    assert cfg.quality_weight == 0.4
    # the axes themselves still parse independently of the engine block
    assert config.load_axes(path).primary_axis.name == "causal_mechanism"


@pytest.mark.parametrize(
    "bad, needle",
    [
        ({"engine": {"open_niches": 0}}, "open_niches"),
        ({"engine": {"quality_weight": 2}}, "quality_weight"),
        ({"engine": {"dedup_tau": 1.5}}, "dedup_tau"),
        ({"engine": {"knn_k": "x"}}, "knn_k"),
        ({"engine": {"under_generation_ratio": 1.5}}, "under_generation_ratio"),
        ({"engine": []}, "engine"),
    ],
)
def test_malformed_engine_raises(bad, needle):
    with pytest.raises(config.ConfigError) as exc:
        EngineConfig.from_dict(bad)
    assert needle in str(exc.value)


def test_ingest_respects_open_niches_override(home):
    # An engine override must reach ingest: shrink open_niches + freeze factor so
    # the partition freezes at the overridden threshold with the overridden count.
    axes = {
        "domain": "ov",
        "axes": [{"name": "m", "type": "open", "primary_novelty": True}],
        "engine": {"open_niches": 4, "open_niche_freeze_factor": 2},  # threshold 8
    }
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"]
    cands = [
        {"id": f"c{i}", "text": f"mechanism {w} {i}", "descriptor": {"m": f"mechanism {w} {i}"}}
        for i, w in enumerate(words)
    ]
    pipeline.ingest("ov", cands, axes, seed=0, home=home)
    on = State("ov", home=home).read_open_nicher()
    assert on["frozen"] is True
    assert np.asarray(on["centroids"]).shape[0] == 4  # the overridden open_niches


def test_init_project_persists_engine_config(home):
    axes = {
        "domain": "p",
        "axes": [{"name": "m", "type": "open", "primary_novelty": True}],
        "engine": {"open_niches": 10, "dedup_tau": 0.9},
    }
    pipeline.init_project("p", axes, home=home)
    meta = State("p", home=home).read_meta()
    assert meta["engine"]["open_niches"] == 10
    assert meta["engine"]["dedup_tau"] == 0.9
