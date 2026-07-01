"""Hermetic tests for the self-provisioning bootstrap.

These exercise only the pure logic — path resolution, the idempotency stamp, the
readiness check, and the concurrency lock. No venv is created and nothing is
installed, so the suite stays offline.
"""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

import pytest

_BOOT_PATH = Path(__file__).resolve().parents[1] / "skills/ideate/scripts/bootstrap.py"


def _load_bootstrap():
    spec = importlib.util.spec_from_file_location("cambrian_bootstrap", _BOOT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bootstrap = _load_bootstrap()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Drop any inherited provisioning env so resolution is deterministic."""
    for var in ("CAMBRIAN_VENV", "CLAUDE_PLUGIN_DATA"):
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------- #
# _clean / resolve_venv_dir
# --------------------------------------------------------------------------- #
def test_clean_drops_empty_and_unsubstituted():
    assert bootstrap._clean("") == ""
    assert bootstrap._clean(None) == ""
    assert bootstrap._clean("   ") == ""
    # an unsubstituted ${VAR} (e.g. CLAUDE_PLUGIN_DATA unset in dev) must not be used
    assert bootstrap._clean("${CLAUDE_PLUGIN_DATA}/venv".split("/")[0]) == ""
    assert bootstrap._clean("/venv") == ""
    assert bootstrap._clean("  /real/path ") == "/real/path"


def test_resolve_priority_explicit_arg_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("CAMBRIAN_VENV", str(tmp_path / "env"))
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path / "data"))
    got = bootstrap.resolve_venv_dir(str(tmp_path / "explicit"))
    assert got == (tmp_path / "explicit").resolve()


def test_resolve_priority_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CAMBRIAN_VENV", str(tmp_path / "env"))
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path / "data"))
    assert bootstrap.resolve_venv_dir(None) == (tmp_path / "env").resolve()


def test_resolve_priority_plugin_data(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path / "data"))
    assert bootstrap.resolve_venv_dir(None) == (tmp_path / "data" / "venv").resolve()


def test_resolve_dev_fallback():
    # No env, no arg, and an empty/unsubstituted --venv all fall back to the dev tree.
    expected = (bootstrap.SKILL_DIR / ".venv").resolve()
    assert bootstrap.resolve_venv_dir(None) == expected
    assert bootstrap.resolve_venv_dir("") == expected
    assert bootstrap.resolve_venv_dir("/venv") == expected


def test_installed_mode_distinguishes_dev_from_data(tmp_path):
    assert bootstrap.installed_mode(tmp_path / "data" / "venv") is True
    assert bootstrap.installed_mode((bootstrap.SKILL_DIR / ".venv").resolve()) is False


def test_venv_python_matches_os(tmp_path):
    py = bootstrap.venv_python(tmp_path)
    if os.name == "nt":
        assert py == tmp_path / "Scripts" / "python.exe"
    else:
        assert py == tmp_path / "bin" / "python"


# --------------------------------------------------------------------------- #
# compute_stamp
# --------------------------------------------------------------------------- #
def test_compute_stamp_is_deterministic():
    assert bootstrap.compute_stamp(include_source=False) == bootstrap.compute_stamp(
        include_source=False
    )


def test_compute_stamp_source_sensitivity():
    # Including the engine sources must change the hash (there are .py files).
    with_src = bootstrap.compute_stamp(include_source=True)
    without_src = bootstrap.compute_stamp(include_source=False)
    assert with_src != without_src


def test_compute_stamp_dev_tracks_requirements_dev(tmp_path, monkeypatch):
    # Editable/dev mode (include_source=False) installs requirements-dev.txt, so a
    # dev-only dep bump must change the stamp and trigger a rebuild.
    fake = tmp_path / "requirements-dev.txt"
    fake.write_text("-r requirements.txt\npytest>=8,<9\n", encoding="utf-8")
    monkeypatch.setattr(bootstrap, "REQS_DEV", fake)
    s1 = bootstrap.compute_stamp(include_source=False)
    fake.write_text("-r requirements.txt\npytest>=9,<10\n", encoding="utf-8")
    s2 = bootstrap.compute_stamp(include_source=False)
    assert s1 != s2
    # Non-editable installs never use requirements-dev.txt, so it must not feed the
    # source-inclusive stamp.
    src_a = bootstrap.compute_stamp(include_source=True)
    fake.write_text("-r requirements.txt\npytest>=8,<9\n", encoding="utf-8")
    assert bootstrap.compute_stamp(include_source=True) == src_a


def test_do_install_removes_venv_on_failure(tmp_path, monkeypatch):
    # A failed dep install must not leave a partial venv that create_venv would
    # later "reuse"; do_install removes it so the next provision rebuilds clean.
    import subprocess

    venv_dir = tmp_path / "venv"

    def fake_create(vd, uv):
        py = bootstrap.venv_python(vd)
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_text("#!stub\n", encoding="utf-8")
        return py

    def fail_install(*a, **k):
        raise subprocess.CalledProcessError(1, ["pip", "install"])

    monkeypatch.setattr(bootstrap, "create_venv", fake_create)
    monkeypatch.setattr(bootstrap, "install_deps", fail_install)
    with pytest.raises(subprocess.CalledProcessError):
        bootstrap.do_install(venv_dir, "stamp")
    assert not venv_dir.exists()


# --------------------------------------------------------------------------- #
# is_ready
# --------------------------------------------------------------------------- #
def _fake_venv(venv_dir: Path, stamp: str) -> None:
    py = bootstrap.venv_python(venv_dir)
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("#!stub\n", encoding="utf-8")
    (venv_dir / bootstrap.PTR_NAME).write_text(py.as_posix(), encoding="utf-8")
    (venv_dir / bootstrap.STAMP_NAME).write_text(stamp, encoding="utf-8")


def test_is_ready_false_when_missing(tmp_path):
    assert bootstrap.is_ready(tmp_path / "venv", "abc") is False


def test_is_ready_true_when_complete_and_matching(tmp_path):
    venv_dir = tmp_path / "venv"
    _fake_venv(venv_dir, "abc")
    assert bootstrap.is_ready(venv_dir, "abc") is True
    # A changed stamp (e.g. plugin update changed deps) invalidates readiness.
    assert bootstrap.is_ready(venv_dir, "different") is False


# --------------------------------------------------------------------------- #
# lock
# --------------------------------------------------------------------------- #
def test_lock_is_mutually_exclusive(tmp_path):
    venv_dir = tmp_path / "venv"
    assert bootstrap.try_acquire(venv_dir) is True
    assert bootstrap.try_acquire(venv_dir) is False  # second caller is locked out
    bootstrap.release(venv_dir)
    assert bootstrap.try_acquire(venv_dir) is True    # released -> acquirable again
    bootstrap.release(venv_dir)


def test_stale_lock_is_stolen(tmp_path):
    venv_dir = tmp_path / "venv"
    assert bootstrap.try_acquire(venv_dir) is True
    lock = bootstrap._lock_dir(venv_dir)
    old = time.time() - bootstrap.STALE_LOCK_SECS - 60
    os.utime(lock, (old, old))
    # A fresh provisioner reclaims an abandoned lock instead of waiting forever.
    assert bootstrap.try_acquire(venv_dir) is True
    bootstrap.release(venv_dir)
