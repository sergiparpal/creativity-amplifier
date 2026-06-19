"""The project lock serializes a whole-project read-modify-write cycle.

It guards ``ingest`` (and an axes-changing ``init``) so two concurrent cycles on the
same project can't clobber each other's generation. The lock is best-effort (same
``mkdir`` mechanism as the pin/discard guard); these tests assert the lifecycle and
that the locked commands still work end to end.
"""

from __future__ import annotations

from creativity_engine import config, pipeline
from creativity_engine.state import State


def _generic():
    return config.load_generic_axes().to_dict()


def test_project_lock_creates_and_releases(home):
    st = State("p", home=home).ensure()
    lock_dir = st.root / ".project.lock"
    assert not lock_dir.exists()
    with st.project_lock():
        assert lock_dir.exists() and lock_dir.is_dir()
    assert not lock_dir.exists()  # released on context exit


def test_project_lock_sequential_acquisitions_do_not_leak(home):
    st = State("p", home=home).ensure()
    for _ in range(3):
        with st.project_lock():
            pass
    assert not (st.root / ".project.lock").exists()


def test_ingest_releases_lock_so_next_cycle_proceeds(home):
    # Two sequential ingests must both run (the first releases the lock on exit), and
    # the lock dir must not linger afterwards.
    from creativity_engine import selftest

    axes = _generic()
    pipeline.init_project("p", axes, seed=0, home=home)
    target = int(State("p", home=home).read_meta()["candidates_per_generation"])
    pipeline.ingest("p", selftest.diverse_candidates(target, gen=0), axes, seed=0, home=home)
    r2 = pipeline.ingest("p", selftest.diverse_candidates(target, gen=1), axes, seed=0, home=home)
    assert r2["slate"]  # second cycle ran
    assert not (State("p", home=home).root / ".project.lock").exists()
