# SessionStart hook (Windows PowerShell — for native Windows without Git Bash).
#
# Thin launcher: find a Python >= 3.11 and hand off to bootstrap.py, which does the
# real, idempotent, concurrency-safe provisioning in a detached background process.
# Registered with "async": true in hooks/hooks.json, so it never blocks session
# startup or the desktop UI. All the heavy logic lives in bootstrap.py.
#
# On macOS/Linux this command is handed to `sh`, where `powershell` is absent, so it
# no-ops harmlessly; provision.sh covers those platforms.
$ErrorActionPreference = 'SilentlyContinue'

$root = $env:CLAUDE_PLUGIN_ROOT
if (-not $root) {
  # Dev fallback: this script lives in hooks/, so the repo root is its grandparent.
  $root = Split-Path -Parent $PSScriptRoot
}

$boot = Join-Path $root 'skills/ideate/scripts/bootstrap.py'
if (-not (Test-Path $boot)) { exit 0 }

# Fast path: already provisioned in the persistent data venv? Skip without a Python spawn.
$data = $env:CLAUDE_PLUGIN_DATA
if ($data -and (Test-Path (Join-Path $data 'venv/engine-python.txt'))) { exit 0 }

# Find a Python >= 3.11. `py` (the Windows launcher) and `python` come first.
$py = $null
foreach ($cand in @('py', 'python', 'python3')) {
  if (Get-Command $cand -ErrorAction SilentlyContinue) {
    & $cand -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)' 2>$null
    if ($LASTEXITCODE -eq 0) { $py = $cand; break }
  }
}

# No suitable Python: stay silent (background hook). The skill surfaces a clear,
# actionable message the first time the user runs /ideate.
if (-not $py) { exit 0 }

& $py $boot --background
exit 0
