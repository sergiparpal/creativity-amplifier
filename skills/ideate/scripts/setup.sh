#!/usr/bin/env bash
# Create the local virtualenv for the creativity engine and install dependencies.
# Idempotent: safe to re-run. The venv lives at ${CLAUDE_SKILL_DIR}/.venv so the
# skill can call ${CLAUDE_SKILL_DIR}/.venv/bin/python -m creativity_engine.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # skills/ideate/scripts
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"                    # skills/ideate
VENV_DIR="$SKILL_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[setup] Python: $("$PYTHON_BIN" --version 2>&1)"
echo "[setup] Creating venv at $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

echo "[setup] Upgrading pip tooling"
"$VENV_DIR/bin/python" -m pip install --upgrade pip wheel setuptools >/dev/null

echo "[setup] Installing requirements"
"$VENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"

echo "[setup] Installing creativity_engine (editable, no extra deps)"
"$VENV_DIR/bin/python" -m pip install -e "$SCRIPT_DIR" --no-deps

echo "[setup] Verifying imports"
"$VENV_DIR/bin/python" -c "import numpy, scipy, sklearn, yaml, creativity_engine; print('[setup] core imports OK')"

echo "[setup] Done. Engine: $VENV_DIR/bin/python -m creativity_engine"
