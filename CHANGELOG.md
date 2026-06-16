# Changelog

All notable changes to the `creativity-amplifier` plugin are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(pre-1.0: minor versions may include breaking changes).

## [0.2.0] — 2026-06-16

### Breaking

- **Default embedder changed from `BAAI/bge-small-en-v1.5` (384-dim) to
  `minishlab/potion-multilingual-128M` (256-dim).** The engine refuses to mix
  embedding widths within one project, so the new default applies cleanly only to
  **new** projects. An existing project must be re-embedded (start a fresh project)
  or pinned to the old embedder with `export CREATIVITY_EMBEDDER=local` (which now
  also requires installing `requirements-local.txt`).

### Added

- **Continuous integration** (`.github/workflows/ci.yml`): a 3 × 3 matrix
  (ubuntu/macos/windows × Python 3.11/3.12/3.13) running the hermetic test suite and
  the engine `selftest`, plus best-effort plugin-manifest validation. No model is
  ever downloaded in CI.
- **Multilingual, torch-free default embedder** (`static`, model2vec
  `potion-multilingual-128M`): 256-dim, 101 languages, numpy-only inference,
  ~120 MB. The English-only `bge` embedder remains available as an opt-in
  (`CREATIVITY_EMBEDDER=local` + `requirements-local.txt`).
- **Prefilter guard**: `ingest` emits a soft `monitor.under_generation` flag (with
  `submitted` / `target_candidates`) when far fewer candidates than the per-generation
  target reach the engine — a mechanical check that the agent isn't cutting variety
  under cover of "off-brief".
- **Open-axis freeze observability**: `ingest` and `metrics` return an `open_axis`
  block (`accumulated` / `freeze_threshold` / `progress` / `frozen`).
- **State pruning** for long sessions: `ingest` drops candidate records/embeddings
  nothing reads again once the store exceeds `engine.state_prune_threshold`
  (default 2000, 0 disables), keeping exactly archive elites, pins, and comparison
  ids — output-neutral.
- **New `paths` CLI command** + a per-project `tmp/` scratch dir for the skill's
  hand-off files (`axes.json` / `candidates.json` / `event.json`), inside the state
  home instead of the user's cwd.
- **Tunable active-learning pair policy**: `engine.ask_sim_weight` /
  `ask_uncertainty_weight` / `ask_novelty_weight` (a non-positive `ask_sim_weight`
  flips from "compare similar" to "compare region-separating" / explore).
- This `CHANGELOG.md`.

### Changed

- **License changed from GPL-3.0-or-later to MIT** to lower the barrier to adoption
  and forks (`LICENSE`, `plugin.json`).
- **Open-axis partition freeze threshold** lowered (`open_niche_freeze_factor`
  4 → 2, i.e. 96 → 48 mechanisms, ~4–5 generations) so the data-adaptive
  fit-once-then-freeze partition actually activates in a realistic session.
- **Cross-platform SessionStart hook** consolidated to a single Node dispatcher
  (`hooks/provision.mjs`) that detects the OS and invokes only the matching launcher
  (`provision.sh` / `provision.ps1`).
- `requirements.txt` now ships `model2vec` (no torch) by default; `sentence-transformers`
  moved to an opt-in `requirements-local.txt`.
- Documentation: scoped the novelty/placement division of labor precisely, documented
  the (now-tunable) `ask_pairs` policy tension, the migration caveat, and the real
  ~120 MB embedder footprint across `README`, `CLAUDE.md`, `docs/PAPER.md`, and the
  skill references.

### Fixed

- **Dual-hook log noise**: the previous setup launched both `sh` and `powershell` on
  every `SessionStart`, so the wrong-OS launcher always failed noisily. The single
  Node dispatcher only ever spawns the launcher that exists.
- **Invisible provisioning failures**: the skill's "engine not ready" path now tails
  `provision.log` before/after the foreground bootstrap, so a failed background build
  is diagnosed instead of re-run blind.
- `api` embedder is now honestly documented as a stub / extension point (it raises
  until a backend is wired up) rather than a supported option.

## [0.1.0] — 2026-06-13

- Initial release: the `ideate` skill + the deterministic diversity engine
  (embeddings, MAP-Elites archive, k-NN novelty, DPP selection, anti-collapse
  monitor, preference memory), self-provisioning via a `SessionStart` hook, and a
  one-command marketplace install.

[0.2.0]: https://github.com/sergiparpal/creativity-amplifier/releases/tag/v0.2.0
[0.1.0]: https://github.com/sergiparpal/creativity-amplifier/releases/tag/v0.1.0
