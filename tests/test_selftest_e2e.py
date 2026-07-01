"""Phase 6: the stubbed end-to-end selftest, variety gate, and collapse reversal."""

from __future__ import annotations

import json

import pytest

from cambrian_engine import selftest
from cambrian_engine.__main__ import main
from cambrian_engine.selftest import MARGIN_DPP, MARGIN_MPD, MARGIN_VENDI


@pytest.fixture
def report(home):
    return selftest.run(project="e2e", seed=0, home=home)


def test_selftest_ok(report):
    assert report["ok"] is True
    assert report["embedder"] == "hash"
    assert report["cycles"] >= 1


def test_variety_gate_passes_with_margins(report):
    vg = report["variety_gate"]
    assert vg["passed"] is True
    eng, base = vg["engine"], vg["single_shot"]
    # diverse slate beats single-shot by a clear margin on every metric. Import the
    # margins from the source so tightening the gate is tracked here, not silently.
    assert eng["mean_pairwise_distance"] > base["mean_pairwise_distance"] + MARGIN_MPD
    assert eng["vendi"] > base["vendi"] + MARGIN_VENDI
    assert eng["niche_entropy"] > base["niche_entropy"]
    # DPP selection beats naive first-N on the SAME (shuffled) pool, averaged over
    # seeds — the de-rigged, non-tautological signal of the engine's own value.
    assert (
        vg["dpp_on_pool"]["mean_pairwise_distance_avg"]
        > vg["first_n_on_pool"]["mean_pairwise_distance_avg"] + MARGIN_DPP
    )
    # null check: DPP doesn't regress below a random subset on a uniform pool.
    # `passed` already encodes the eps tolerance, so just trust the source gate here.
    assert vg["null_check"]["passed"] is True
    assert all(vg["checks"].values())


def test_live_semantic_skips_cleanly_offline(report):
    # On the hash (non-live) path the semantic check must skip without failing.
    sem = report["live_semantic"]
    assert sem["ran"] is False
    assert sem["skipped"] is True
    # a skipped semantic check never drags the overall result down
    assert report["ok"] is True


def test_collapse_reversal_passes(report):
    cr = report["collapse_reversal"]
    assert cr["passed"] is True
    assert cr["collapsed_monitor"]["collapsing"] is True
    assert cr["recovered_monitor"]["collapsing"] is False
    # diversity measurably recovers after pressure rises
    assert cr["recovered_monitor"]["mean_cosine"] < cr["collapsed_monitor"]["mean_cosine"]


def test_state_files_written(report):
    assert all(report["state_files_written"].values())


def test_selftest_is_deterministic(home):
    r1 = selftest.run(project="det1", seed=0, home=home)
    r2 = selftest.run(project="det2", seed=0, home=home)
    assert r1["variety_gate"]["engine"] == r2["variety_gate"]["engine"]
    assert r1["variety_gate"]["single_shot"] == r2["variety_gate"]["single_shot"]
    # The seeded-shuffle DPP-on-pool path is the most RNG-dependent output; assert it
    # reproduces too so the determinism the design relies on is actually guarded.
    assert r1["variety_gate"]["dpp_on_pool"] == r2["variety_gate"]["dpp_on_pool"]
    assert r1["variety_gate"]["first_n_on_pool"] == r2["variety_gate"]["first_n_on_pool"]
    assert (
        r1["collapse_reversal"]["collapsed_monitor"]["mean_cosine"]
        == r2["collapse_reversal"]["collapsed_monitor"]["mean_cosine"]
    )


def test_cli_selftest_exits_zero(home, capsys):
    code = main(["selftest", "--project", "cli", "--seed", "0"])
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert payload["ok"] is True
