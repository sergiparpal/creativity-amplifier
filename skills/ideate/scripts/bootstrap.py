#!/usr/bin/env python3
"""Cross-platform bootstrap for the creativity engine.

Creates a local virtualenv at ``<skill_dir>/.venv``, installs the engine and its
dependencies, and records the venv interpreter path so the skill can locate it on
any OS without hard-coding ``bin/python`` vs ``Scripts\\python.exe``.

Idempotent: safe to re-run. Launch with any system Python >= 3.11:

    python  bootstrap.py        # Windows (or:  py bootstrap.py)
    python3 bootstrap.py        # macOS / Linux

On success the resolved interpreter path is written to
``<skill_dir>/.venv/engine-python.txt``; read that file to build the
``ENGINE = "<path>" -m creativity_engine`` command.
"""
from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent      # skills/ideate/scripts
SKILL_DIR = SCRIPT_DIR.parent                     # skills/ideate
VENV_DIR = SKILL_DIR / ".venv"
REQS = SCRIPT_DIR / "requirements.txt"
PYTHON_PTR = VENV_DIR / "engine-python.txt"       # records the venv interpreter path

MIN_PY = (3, 11)


def venv_python(venv_dir: Path) -> Path:
    """Path to the venv's interpreter for the current OS."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run(cmd: list[str]) -> None:
    print(f"[bootstrap] $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def main() -> int:
    if sys.version_info < MIN_PY:
        sys.exit(
            f"[bootstrap] Need Python >= {MIN_PY[0]}.{MIN_PY[1]}, "
            f"got {sys.version.split()[0]} ({sys.executable})"
        )
    if not REQS.exists():
        sys.exit(f"[bootstrap] requirements.txt not found at {REQS}")

    print(f"[bootstrap] System Python: {sys.version.split()[0]} ({sys.executable})")

    py = venv_python(VENV_DIR)
    if not py.exists():
        print(f"[bootstrap] Creating venv at {VENV_DIR}")
        try:
            venv.EnvBuilder(with_pip=True).create(VENV_DIR)
        except Exception as exc:  # ensurepip/venv unavailable on some distros
            sys.exit(
                f"[bootstrap] Could not create venv: {exc}\n"
                "[bootstrap] On Debian/Ubuntu you may need: "
                "sudo apt install python3-venv"
            )
        py = venv_python(VENV_DIR)
    else:
        print(f"[bootstrap] Reusing existing venv at {VENV_DIR}")

    if not py.exists():
        sys.exit(f"[bootstrap] venv interpreter not found at {py}")

    print("[bootstrap] Upgrading pip tooling")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])

    print("[bootstrap] Installing requirements")
    run([str(py), "-m", "pip", "install", "-r", str(REQS)])

    print("[bootstrap] Installing creativity_engine (editable, no extra deps)")
    run([str(py), "-m", "pip", "install", "-e", str(SCRIPT_DIR), "--no-deps"])

    print("[bootstrap] Verifying imports")
    run([
        str(py), "-c",
        "import numpy, sklearn, yaml, creativity_engine; "
        "print('[bootstrap] core imports OK')",
    ])

    # Single source of truth for the skill, on every OS. Forward slashes work in
    # Git Bash, PowerShell, and cmd alike, so the recorded path is shell-agnostic.
    PYTHON_PTR.write_text(py.as_posix(), encoding="utf-8")
    print(f"[bootstrap] Engine interpreter: {py.as_posix()}")
    print(f"[bootstrap] Wrote {PYTHON_PTR}")
    print("[bootstrap] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
