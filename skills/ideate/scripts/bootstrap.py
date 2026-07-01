#!/usr/bin/env python3
"""Cross-platform, self-provisioning bootstrap for the Cambrian engine.

This is the single source of truth for building the engine's virtualenv. It is
invoked three ways, all idempotent and safe to re-run:

* by the plugin's ``SessionStart`` hook (``hooks/provision.sh`` / ``provision.ps1``)
  right after the plugin is installed/loaded — see ``--background`` below;
* by the ``ideate`` skill as a foreground last-resort if a user runs ``/ideate``
  before the background provision has finished (graceful catch-up);
* by a developer from a shell (``python bootstrap.py`` or ``bash setup.sh``).

What it does (torch-free multilingual stack by default):

* creates a venv (``uv venv`` if ``uv`` is on PATH, else stdlib ``venv``);
* installs ``requirements.txt`` (model2vec / numpy / scikit-learn / pyyaml — the
  **static** ``minishlab/potion-multilingual-128M`` embedder is the default, ~120 MB
  and numpy-only at inference; the heavier ``local`` bge / sentence-transformers
  embedder is opt-in via ``requirements-local.txt``) plus the ``cambrian_engine``
  package (``uv pip`` if available, else ``pip``);
* records the resolved venv interpreter path in ``<venv>/engine-python.txt`` so the
  skill can find it on any OS without hard-coding ``bin/python`` vs
  ``Scripts\\python.exe``.

Where the venv lives (first match wins):

1. ``--venv PATH``                       (explicit; passed by the skill)
2. ``$CAMBRIAN_VENV``        (explicit override)
3. ``$CLAUDE_PLUGIN_DATA/venv``          (installed via marketplace — persists across
                                          plugin updates; the recommended location)
4. ``<skill_dir>/.venv``                 (developer fallback: ``--plugin-dir .`` / shell)

Idempotency & robustness:

* A content stamp (hash of ``requirements.txt`` + ``pyproject.toml`` + — when
  installed non-editable — the engine sources) lets a fast path skip work when the
  venv is already current, and forces a rebuild when a plugin update changes deps
  or engine code.
* An atomic lock dir serializes concurrent provisions (the two hook entries, extra
  terminals, a skill catch-up racing the hook) so two builds never clobber a venv.
* ``--background`` re-spawns a fully detached worker and returns in milliseconds, so
  even a Claude Code without ``async`` hook support never blocks on the install.

Launch with any system Python >= 3.11:

    python  bootstrap.py        # Windows (or:  py bootstrap.py)
    python3 bootstrap.py        # macOS / Linux
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
import venv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent      # skills/ideate/scripts
SKILL_DIR = SCRIPT_DIR.parent                     # skills/ideate
REQS = SCRIPT_DIR / "requirements.txt"
REQS_DEV = SCRIPT_DIR / "requirements-dev.txt"  # editable/dev installs only (adds pytest)
PYPROJECT = SCRIPT_DIR / "pyproject.toml"
ENGINE_PKG = SCRIPT_DIR / "cambrian_engine"

MIN_PY = (3, 11)
PTR_NAME = "engine-python.txt"      # interpreter pointer, written inside the venv dir
STAMP_NAME = "install.stamp"        # content hash of the install inputs
LOCK_NAME = ".cambrian-provision.lock"   # atomic lock dir, kept beside the venv
STALE_LOCK_SECS = 30 * 60           # treat a lock older than this as abandoned
LOG_NAME = "provision.log"          # where the detached worker logs
SCHEMA = "1"                        # bump to force every venv to rebuild


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
def _clean(value: str | None) -> str:
    """Drop empty / unsubstituted ``${...}`` values from env or args."""
    if not value:
        return ""
    value = value.strip()
    if not value or value.startswith("${") or value in ("/venv",):
        return ""
    return value


def resolve_venv_dir(explicit: str | None = None) -> Path:
    """Where the engine venv should live, by priority (see module docstring)."""
    chosen = _clean(explicit)
    if chosen:
        return Path(chosen).expanduser().resolve()

    override = _clean(os.environ.get("CAMBRIAN_VENV"))
    if override:
        return Path(override).expanduser().resolve()

    plugin_data = _clean(os.environ.get("CLAUDE_PLUGIN_DATA"))
    if plugin_data:
        return (Path(plugin_data).expanduser() / "venv").resolve()

    return (SKILL_DIR / ".venv").resolve()


def installed_mode(venv_dir: Path) -> bool:
    """True when provisioning into a persistent plugin-data venv (not the dev tree).

    In dev mode we install the engine editable so source edits take effect with no
    rebuild; installed mode copies the package into the persistent venv so it stays
    in sync as ``CLAUDE_PLUGIN_ROOT`` rotates on plugin updates.
    """
    return (SKILL_DIR / ".venv").resolve() != venv_dir


def venv_python(venv_dir: Path) -> Path:
    """Path to the venv's interpreter for the current OS."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


# --------------------------------------------------------------------------- #
# Idempotency: content stamp
# --------------------------------------------------------------------------- #
def compute_stamp(*, include_source: bool) -> str:
    """Hash the inputs whose change should trigger a rebuild."""
    h = hashlib.sha256()
    h.update(SCHEMA.encode())
    stamp_files = [REQS, PYPROJECT]
    # Editable/dev installs (not include_source) install requirements-dev.txt, so a
    # dev-only dep bump (e.g. the pytest pin) must trigger a rebuild too; non-editable
    # installs never use it, so it stays out of their stamp.
    if not include_source and REQS_DEV.exists():
        stamp_files.append(REQS_DEV)
    for path in stamp_files:
        h.update(path.name.encode())
        h.update(b"\0")
        h.update(path.read_bytes() if path.exists() else b"")
        h.update(b"\0")
    if include_source and ENGINE_PKG.is_dir():
        for src in sorted(ENGINE_PKG.rglob("*.py")):
            h.update(src.relative_to(ENGINE_PKG).as_posix().encode())
            h.update(b"\0")
            h.update(src.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def is_ready(venv_dir: Path, stamp: str) -> bool:
    """True when the venv already satisfies the current stamp."""
    py = venv_python(venv_dir)
    ptr = venv_dir / PTR_NAME
    stamp_file = venv_dir / STAMP_NAME
    if not (py.exists() and ptr.exists() and stamp_file.exists()):
        return False
    try:
        return stamp_file.read_text(encoding="utf-8").strip() == stamp
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# Lock (atomic mkdir; steals abandoned locks)
# --------------------------------------------------------------------------- #
def _lock_dir(venv_dir: Path) -> Path:
    # Beside the venv, not inside it, so a half-built venv can't shadow the lock.
    return venv_dir.parent / LOCK_NAME


def try_acquire(venv_dir: Path) -> bool:
    lock = _lock_dir(venv_dir)
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError:
        try:
            age = time.time() - lock.stat().st_mtime
        except OSError:
            age = 0.0
        if age > STALE_LOCK_SECS:
            # Steal atomically: renaming a directory is atomic, so exactly one
            # racer can move the stale lock aside (the loser gets the lock already
            # gone and backs off). rmtree+mkdir alone is not atomic together — two
            # simultaneous stealers could both proceed.
            sidelined = lock.parent / f"{LOCK_NAME}.stale-{os.getpid()}"
            try:
                os.replace(lock, sidelined)
            except OSError:
                return False  # lost the steal race; caller re-loops and waits
            shutil.rmtree(sidelined, ignore_errors=True)
            try:
                lock.mkdir()
            except OSError:
                return False
        else:
            return False
    try:
        (lock / "info").write_text(f"pid={os.getpid()} t={time.time():.0f}\n", "utf-8")
    except OSError:
        pass
    return True


def release(venv_dir: Path) -> None:
    shutil.rmtree(_lock_dir(venv_dir), ignore_errors=True)


# --------------------------------------------------------------------------- #
# Install
# --------------------------------------------------------------------------- #
def run(cmd: list[str]) -> None:
    print(f"[bootstrap] $ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def create_venv(venv_dir: Path, uv: str | None) -> Path:
    py = venv_python(venv_dir)
    if py.exists():
        print(f"[bootstrap] Reusing existing venv at {venv_dir}", flush=True)
        return py
    print(f"[bootstrap] Creating venv at {venv_dir}", flush=True)
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    if uv:
        # Seed uv's venv from the interpreter we already validated (>= 3.11) so uv
        # never silently downloads a different Python.
        run([uv, "venv", "--python", sys.executable, str(venv_dir)])
    else:
        try:
            venv.EnvBuilder(with_pip=True).create(venv_dir)
        except Exception as exc:  # ensurepip/venv unavailable on some distros
            raise SystemExit(
                f"[bootstrap] Could not create venv: {exc}\n"
                "[bootstrap] On Debian/Ubuntu you may need: sudo apt install python3-venv"
            )
    py = venv_python(venv_dir)
    if not py.exists():
        raise SystemExit(f"[bootstrap] venv interpreter not found at {py}")
    return py


def install_deps(py: Path, uv: str | None, editable: bool) -> None:
    engine_target = ["-e", str(SCRIPT_DIR)] if editable else [str(SCRIPT_DIR)]
    # In editable/dev mode also install the test tooling so a developer who runs
    # setup.sh can run the suite immediately; end-user installs get runtime only.
    reqs = str(REQS_DEV if (editable and REQS_DEV.exists()) else REQS)
    if uv:
        print("[bootstrap] Installing requirements with uv", flush=True)
        run([uv, "pip", "install", "--python", str(py), "-r", reqs])
        run([uv, "pip", "install", "--python", str(py), *engine_target, "--no-deps"])
    else:
        print("[bootstrap] Upgrading pip tooling", flush=True)
        run([str(py), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])
        print("[bootstrap] Installing requirements with pip", flush=True)
        run([str(py), "-m", "pip", "install", "-r", reqs])
        run([str(py), "-m", "pip", "install", *engine_target, "--no-deps"])

    print("[bootstrap] Verifying core imports", flush=True)
    run([
        str(py), "-c",
        "import numpy, sklearn, yaml, cambrian_engine; "
        "print('[bootstrap] core imports OK')",
    ])


def _atomic_write_text(path: Path, text: str) -> None:
    """Crash-safe text write (temp + fsync + os.replace).

    The pointer/stamp are the engine's readiness gate, so a half-written stamp
    (crash/disk-full mid-write) must never be left behind — it would either fake
    "ready" forever or never rebuild. Writing atomically guarantees readers see
    either the old file or the complete new one.
    """
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def do_install(venv_dir: Path, stamp: str) -> Path:
    if not REQS.exists():
        raise SystemExit(f"[bootstrap] requirements.txt not found at {REQS}")
    uv = shutil.which("uv")
    if uv:
        print(f"[bootstrap] Found uv at {uv} — using it for a faster install", flush=True)
    else:
        print("[bootstrap] uv not on PATH — using python -m venv + pip", flush=True)

    py = create_venv(venv_dir, uv)
    try:
        install_deps(py, uv, editable=not installed_mode(venv_dir))
    except BaseException:
        # A failed/interrupted dep install leaves a venv with an interpreter but a
        # partial dependency graph that create_venv would silently "reuse" next run
        # (pip can't always self-heal a half-built tree). Remove it so the next
        # provision rebuilds clean. The lock lives BESIDE the venv (_lock_dir), so
        # this never deletes the lock this process still holds.
        shutil.rmtree(venv_dir, ignore_errors=True)
        raise

    # Single source of truth for the skill, on every OS. Forward slashes work in
    # Git Bash, PowerShell, and cmd alike, so the recorded path is shell-agnostic.
    # Pointer first, then stamp (both atomic): a crash between them never leaves a
    # matching stamp without a usable interpreter pointer.
    _atomic_write_text(venv_dir / PTR_NAME, py.as_posix())
    _atomic_write_text(venv_dir / STAMP_NAME, stamp)
    print(f"[bootstrap] Engine interpreter: {py.as_posix()}", flush=True)
    print(f"[bootstrap] Wrote {venv_dir / PTR_NAME}", flush=True)
    return py


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def provision(venv_dir: Path, *, wait_secs: float) -> int:
    """Ensure the venv is current. Foreground; returns a process exit code."""
    stamp = compute_stamp(include_source=installed_mode(venv_dir))
    if is_ready(venv_dir, stamp):
        print(f"[bootstrap] Engine already provisioned at {venv_dir}", flush=True)
        return 0

    if sys.version_info < MIN_PY:
        sys.stderr.write(
            f"[bootstrap] Need Python >= {MIN_PY[0]}.{MIN_PY[1]} to build the engine, "
            f"but this interpreter is {sys.version.split()[0]} ({sys.executable}).\n"
            "[bootstrap] Install Python 3.11+ (python.org / your package manager / "
            "`winget install Python.Python.3.12`) and run /ideate again.\n"
        )
        return 3

    # Serialize against any other provisioner (the second hook entry, more terminals,
    # the skill racing the background hook).
    deadline = time.time() + max(0.0, wait_secs)
    while not try_acquire(venv_dir):
        if is_ready(venv_dir, stamp):
            print("[bootstrap] Another setup just finished — engine ready.", flush=True)
            return 0
        if time.time() >= deadline:
            print(
                "[bootstrap] Another setup is already in progress; "
                "it will finish in the background. Try /ideate again shortly.",
                flush=True,
            )
            return 0
        time.sleep(2.0)

    try:
        if is_ready(venv_dir, stamp):  # re-check now that we hold the lock
            return 0
        do_install(venv_dir, stamp)
        print("[bootstrap] Done.", flush=True)
        return 0
    except subprocess.CalledProcessError as exc:
        # A failed pip/uv/venv command: in the foreground catch-up path (the skill
        # running /ideate before the background build finished) show a clean,
        # actionable line instead of a raw traceback. The detached worker logs the
        # same to provision.log.
        log_path = venv_dir.parent / LOG_NAME
        sys.stderr.write(
            f"[bootstrap] Install step failed (exit {exc.returncode}): "
            f"{' '.join(str(c) for c in exc.cmd)}\n"
            f"[bootstrap] See {log_path} for details, then run /ideate again.\n"
        )
        return 1
    finally:
        release(venv_dir)


def spawn_background(venv_dir: Path) -> int:
    """Re-spawn a fully detached worker and return immediately (non-blocking)."""
    stamp = compute_stamp(include_source=installed_mode(venv_dir))
    if is_ready(venv_dir, stamp):
        return 0
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = venv_dir.parent / LOG_NAME
    try:
        log = open(log_path, "ab", buffering=0)
    except OSError:
        log = subprocess.DEVNULL  # type: ignore[assignment]

    kwargs: dict = {}
    if os.name == "nt":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        CREATE_NO_WINDOW = 0x08000000
        kwargs["creationflags"] = (
            DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        )
    else:
        kwargs["start_new_session"] = True

    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--run", "--venv", str(venv_dir)],
        stdout=log,
        stderr=log,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        **kwargs,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Provision the Cambrian engine venv.")
    parser.add_argument("--venv", default=None, help="explicit venv directory")
    parser.add_argument(
        "--background",
        action="store_true",
        help="spawn a detached worker and return immediately (used by the hook)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="internal: the detached worker entrypoint",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=20.0 * 60,
        help="seconds a foreground run waits for an in-flight provision",
    )
    args = parser.parse_args(argv)

    venv_dir = resolve_venv_dir(args.venv)
    print(
        f"[bootstrap] System Python: {sys.version.split()[0]} ({sys.executable})",
        flush=True,
    )
    print(f"[bootstrap] Target venv: {venv_dir}", flush=True)

    if args.background:
        return spawn_background(venv_dir)
    # --run (detached worker) and the default/manual path are both foreground here.
    return provision(venv_dir, wait_secs=args.wait)


if __name__ == "__main__":
    raise SystemExit(main())
