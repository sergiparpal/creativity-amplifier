#!/bin/sh
# SessionStart hook (POSIX: sh / Git Bash / WSL / macOS / Linux).
#
# Thin launcher: find a Python >= 3.11 and hand off to bootstrap.py, which does the
# real, idempotent, concurrency-safe provisioning in a detached background process.
# Registered with "async": true in hooks/hooks.json, so it never blocks session
# startup or the desktop UI. All the heavy logic lives in bootstrap.py.
#
# On Windows-without-Git-Bash this command runs under PowerShell instead and `sh`
# is absent, so it no-ops harmlessly; provision.ps1 covers that case.
set -u

ROOT="${CLAUDE_PLUGIN_ROOT:-}"
if [ -z "$ROOT" ]; then
  # Dev fallback (`--plugin-dir .` without the var, or run by hand): hooks/ -> repo root.
  ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." 2>/dev/null && pwd)" || exit 0
fi

BOOT="$ROOT/skills/ideate/scripts/bootstrap.py"
[ -f "$BOOT" ] || exit 0

# Fast path: already provisioned in the persistent data venv? Skip without a Python spawn.
DATA="${CLAUDE_PLUGIN_DATA:-}"
if [ -n "$DATA" ] && [ -f "$DATA/venv/engine-python.txt" ]; then
  exit 0
fi

# Find a Python >= 3.11. Prefer names likely to be a real CPython on each OS.
PY=""
for cand in python3 python py python3.13 python3.12 python3.11; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)' >/dev/null 2>&1; then
      PY="$cand"
      break
    fi
  fi
done

# No suitable Python: stay silent here (this is a background hook). The skill surfaces
# a clear, actionable message the first time the user runs /ideate.
[ -n "$PY" ] || exit 0

exec "$PY" "$BOOT" --background
