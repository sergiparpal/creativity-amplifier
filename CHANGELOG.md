# Changelog

All notable changes to **creativity-amplifier** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The one invariant behind every release: **diversity is owned by geometry, never by
the judge.** New signals are added as *measurement* (advisory, reported) — they are
never wired into selection, the monitor's `collapsing` flag, or the self-test's `ok`
verdict unless explicitly noted.

## [0.5.0] — 2026-06-20

Mechanism-first generation and three new observability signals, all advisory —
*measured, never selected on* — plus a robustness pass on state/geometry integrity.

### Added
- **Discards (human veto).** The user can discard any slate idea at round end — the
  symmetric negative of a pin. A discard removes the idea from the *presented* slate
  pool and the *parent* pool (so it stops re-appearing and is never bred from), but is
  **never** wired into novelty, the DPP kernel, niching, fitness, or the monitor — it is
  the user pruning, not a quality heuristic, so it preserves the invariant. Pins and
  discards are **mutually exclusive, latest action wins** (re-pinning un-discards).
  Persisted per-domain in `discards.json`; `remember` accepts `{"type":"discard"}` and
  `parents` always excludes discards.
- **Mechanism-first generation (G4).** Generation now commits to a distinct `mechanism`
  (the open / `primary_novelty` axis value, the "core how") *before* writing each surface
  idea, via the new `principle_first` operator — *same mechanism = same idea*. This aligns
  variation with the axis the engine actually niches on.
- **Mechanism-space novelty (S4).** Advisory `mechanism_novelty` per slate item,
  `slate_mechanism_novelty` on each cycle, and archive-scoped `mechanism_spread` /
  `mechanism_n` in `metrics`, backed by a parallel `mech_embeddings.json` store. Same
  k-NN kernel as the surface `novelty`, run on the open-axis embedding. Measurement only —
  never in dedup, the DPP `q`/kernel, parents, the fitness clip, the monitor, or `selftest`.
- **Surface/mechanism gap probe.** Opt-in `engine.gap_probe` (default **off** ⇒ zero cost,
  output byte-for-byte unchanged). When enabled, `ingest` emits a `surface_mechanism_gap`
  record (`surface_spread`, `mechanism_spread`, `gap = surface − mechanism`, pairwise-distance
  `corr`) on the slate and appends a bounded record to a `gap_log` that `metrics` surfaces;
  the skill folds it into a plain-language, session-end summary. Quantifies whether a slate's
  *wording* diversity overstates its *approach* diversity. Advisory measurement only — never a
  gate, never wired into selection. The self-test gained a report-only gap probe with a
  mechanism-monotone sanity fixture (excluded from `ok`).

### Fixed
- **State / geometry integrity.** `init-project` now also resets the parallel mechanism-embedding
  store along with the other geometry-coupled state. `ingest` refuses an `--axes` whose axes
  differ from the project's persisted snapshot (niche keys are built from the axes), with a clean
  `ConfigError` instead of silently mixing incompatible niche keys. `Candidate.from_dict` rejects
  empty/whitespace `text` (it would embed to a zero vector that never dedups and always scores
  maximally novel). An unknown key in the `engine:` config block is now a loud `ConfigError`, not
  a silent fall-back to defaults.
- **Engine robustness.** Reset stale geometry on re-init, lock cycles against concurrent runs, and
  cap the `metrics` mean-cosine cost on large archives.

### Changed
- **Docs.** README, `CLAUDE.md`, `docs/PAPER.md`, and engine docstrings document mechanism-first
  generation, mechanism-space novelty, the gap probe, discards, the ingest axes guard, unknown-key
  rejection, and the mechanism store. Git workflow note: commit on `main`, don't branch unless asked.

## [0.4.1] — 2026-06-19

Documentation only; no engine code or behavior changes.

### Changed
- **skill:** pinning is now an explicit, active step in the selection dialogue — the skill invites
  pinning of *any* slate idea (not only the two suggested A-vs-B pairs) and documents the signal
  hierarchy (a pin is the strong durable signal; `ask_pairs` only refine a bounded, low-weight fitness).
- **readme/docs:** emphasize pinning any idea; document the plugin update command, the
  variety-erosion + prefilter-guard sensors, the explore schedule, the originality module, and the
  `provision.mjs` dispatcher; paper refresh.

## [0.4.0] — 2026-06-18

Two opt-in/advisory loop features plus a review-driven robustness pass.

### Added
- **Generation-aware ask-policy schedule (S3):** opt-in `engine.explore_until_generation` schedules
  explore-first (region-separating pairs) then refine (similar pairs) by generation index; `0` = off.
- **Variety-erosion sensor (S2):** advisory `variety_eroding` monitor flag that fires on the
  *acceleration* of survivor-novelty decay with healthy submit counts. Strictly advisory — never
  affects `collapsing`, keeps its own `novelty_window` / `erosion_streak`. Knobs: `erosion_window`
  (W), `erosion_accel_ratio` (ρ), `erosion_persist` (K).

### Fixed
- **state:** lossy project/domain id slugs no longer collide on one directory; atomic writes fsync the
  parent dir for crash-durability; stale-lock steal is atomic (no TOCTOU double-steal).
- **pipeline:** open-axis niching knobs are pinned to the init snapshot (no mid-session CVT refit);
  `metrics` tolerates a null engine block; empty-candidate cycles return the full response schema.
- **diversity:** DPP early-stop tied to the kernel jitter; farthest-point top-up seeded by the current
  selection (extends the frontier instead of restarting).

### Changed
- **deps:** model2vec cap tightened to `<0.9`. **provisioning:** the dev/editable install stamp tracks
  `requirements-dev.txt`, and a failed dep install cleans the partial venv for a clean rebuild.

## [0.3.0] — 2026-06-18

### Added
- **Advisory originality probe** and **anti-cliché generation** with a held-out obvious-set split
  (`O_train` repels at generation; only the disjoint `O_test` half is ever scored).

### Fixed
- Substantial robustness pass: crash-safe preference memory, non-finite input guards, atomic/locked
  state writes, hardened provisioning, version-bounded deps with a runtime/dev split, and CI coverage
  for the provisioner.

## [0.2.0] — 2026-06-16

### Added
- Cross-platform CI protecting the test matrix and the correctness contract; a single cross-platform
  hook dispatcher; MIT license.

### Changed
- Default embedder is now **torch-free and multilingual** (model2vec `potion-multilingual-128M`);
  the heavy `bge`/sentence-transformers stack is opt-in.
- Documented the novelty/placement division of labor; made the ask-pair policy tunable.

### Fixed
- Engine correctness/cleanup: reachable open-axis freeze, honest API stub, bounded state; prefilter
  guard, visible provisioning failures, temp-file hygiene.

## [0.1.0] — 2026-06-15

Initial release: the `ideate` skill plus the deterministic diversity engine (embeddings, MAP-Elites
archive, k-NN novelty, DPP selection, anti-collapse monitor, preference memory), self-provisioning via
a `SessionStart` hook, and a one-command marketplace install.

[0.5.0]: https://github.com/sergiparpal/creativity-amplifier/releases/tag/v0.5.0
[0.4.1]: https://github.com/sergiparpal/creativity-amplifier/releases/tag/v0.4.1
[0.4.0]: https://github.com/sergiparpal/creativity-amplifier/releases/tag/v0.4.0
[0.3.0]: https://github.com/sergiparpal/creativity-amplifier/releases/tag/v0.3.0
[0.2.0]: https://github.com/sergiparpal/creativity-amplifier/releases/tag/v0.2.0
[0.1.0]: https://github.com/sergiparpal/creativity-amplifier/releases/tag/v0.1.0
