#!/usr/bin/env bash
# Back-compat shim: the real, cross-platform bootstrap is bootstrap.py.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-python3}" "$SCRIPT_DIR/bootstrap.py" "$@"
