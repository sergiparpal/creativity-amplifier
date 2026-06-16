# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Claude Code **plugin** named `creativity-amplifier`. It ships **one model-invoked skill**
(`skills/ideate/`, invoked as `/creativity-amplifier:ideate <brief>`) plus a bundled, server-less
**Python "creativity engine"** (`skills/ideate/scripts/creativity_engine/`, a CLI) that owns the
anti-convergence math.

The work is split across two halves that meet at a JSON contract:

- **The LLM half** — `skills/ideate/SKILL.md` + `skills/ideate/references/*.md`. These are
  *instructions to a future Claude instance*, which acts as the runtime "agent": it generates idea
  variations, prefilters for validity/on-brief, and runs the human selection dialogue. There is no
  extra chat-LLM API key — Claude itself is the generator and judge.
- **The engine half** — the deterministic Python CLI. Pure, JSON-in/JSON-out, no LLM calls.

When editing here you may be doing either job: maintaining the Python engine, or refining the
prose contract in `SKILL.md`/`references/` that future Claude instances execute.

## The one invariant: diversity is decoupled from the judge

This principle is load-bearing across `SKILL.md`, `pipeline.py`, `archive.py`, and `monitor.py`,
and any change must preserve it:

- **Geometry (the engine) owns *novelty / diversity*** — embeddings, MAP-Elites niching, k-NN
  novelty, DPP selection.
- **The LLM (agent) owns only *validity* and *within-niche ranking***. The judge filters off-brief
  candidates and may attach a `fitness` (0–1) that ranks *within* a niche. That fitness is also
  allowed a **bounded, low-weight** say in the DPP slate (quality-weighted diversity): it is
  affine-rescaled and **clipped to a [0.7, 1.3] multiplier** and blended at `QUALITY_WEIGHT=0.3`, so
  it can nudge ordering but can **never** prune variety, collapse the slate's diversity, or pick the
  final slate. (Set the weight to 0 for pure diversity.)
- **The user is the real selector.** The engine proposes a diverse slate + the most-informative
  A-vs-B pairs; the human chooses and pins "stepping stones".
- The embedder is deliberately a **different model family** from Claude (default
  `minishlab/potion-multilingual-128M`; opt-in high-fidelity `BAAI/bge-small-en-v1.5`) so
  "what's novel" isn't judged by the lineage that generated the ideas.
- The **anti-collapse monitor is never bypassed** — when it flags convergence the skill raises
  diversity pressure next round; it is never removed or worked around.

## Install & provisioning

Two install paths, one provisioner (`skills/ideate/scripts/bootstrap.py`):

- **End users (marketplace):** `/plugin marketplace add sergiparpal/creativity-amplifier`
  then `/plugin install creativity-amplifier@sergiparpal`. A `SessionStart` hook
  (`hooks/hooks.json`, `async: true`) runs `bootstrap.py` in a detached background
  process right after load — non-blocking, idempotent, concurrency-safe. The venv is
  built in **`${CLAUDE_PLUGIN_DATA}/venv`** (the plugin's persistent data dir, so it
  survives plugin updates) and the engine is installed **non-editable** there. Default
  install is the **torch-free multilingual stack** (the `static` model2vec embedder,
  `potion-multilingual-128M`, ~120 MB); the heavier `local` (bge / sentence-transformers)
  embedder is opt-in via `requirements-local.txt`.
- **Developers:** `bash skills/ideate/scripts/setup.sh` (or `python3
  skills/ideate/scripts/bootstrap.py`) builds `skills/ideate/.venv` with the engine
  installed **editable**, then `claude --plugin-dir .`.

`bootstrap.py` uses `uv` when it is on PATH (faster) and falls back to `python -m venv`
+ `pip` otherwise; it never auto-installs `uv` or any other tool. It writes the resolved
interpreter path to `<venv>/engine-python.txt` (the skill reads this to locate the
interpreter) and a content hash to `<venv>/install.stamp` (rebuild trigger on a plugin
update that changes deps or — in non-editable mode — engine sources). The two
`hooks/provision.*` launchers are thin: they only find a Python ≥ 3.11 and hand off to
`bootstrap.py --background`. If the engine isn't ready when `/ideate` runs, the skill
shows a one-time "setting up…" message and finishes the build in the foreground.

## Commands

Run Python through the engine's own venv (`skills/ideate/.venv/bin/python` in dev), not
system Python.

```bash
# One-time setup: create skills/ideate/.venv and install deps (idempotent)
bash skills/ideate/scripts/setup.sh

# Full test suite (hermetic: hash embedder + temp state home, no model download)
skills/ideate/.venv/bin/python -m pytest -q

# A single test file / test
skills/ideate/.venv/bin/python -m pytest tests/test_diversity_monitor.py -q
skills/ideate/.venv/bin/python -m pytest -k "dpp" -q

# Offline end-to-end self-test (the correctness contract; exits 0 on pass)
skills/ideate/.venv/bin/python -m creativity_engine selftest
# ...with the live embedder instead of the hash one:
skills/ideate/.venv/bin/python -m creativity_engine selftest --live

# Load the plugin in Claude Code without installing it, then validate it
claude --plugin-dir .
claude plugin validate .            # or: claude plugin validate --strict .
```

Set `CREATIVITY_DEBUG=1` to get a full traceback from the CLI instead of the clean one-line error.

## Engine CLI

`python -m creativity_engine <command> --project <id> [--axes axes.json] [--seed N]` — every command
reads/writes JSON, prints JSON to stdout, errors to stderr with a non-zero exit.

| Command | Does |
| :-- | :-- |
| `init-project` | create state dirs, snapshot the resolved axes + session settings |
| `paths` | ensure the project state dir (incl. its `tmp/` scratch dir) + return resolved paths |
| `recall` | return preference memory for in-context injection |
| `ingest` | embed → dedup → place → novelty → archive → DPP → monitor (one cycle) |
| `remember` | append a comparison/pin to preference memory |
| `parents` | diverse parents for the next generation (pins always kept) |
| `metrics` | archive health (entropy, mean cosine, coverage, n) + open-axis freeze progress |
| `selftest` | full loop with stubbed LLM + human; value gate + collapse reversal |

## Architecture

**Module layering** (`skills/ideate/scripts/creativity_engine/`), lowest to highest:

- `config.py` — foundation. Dataclasses (`Axis`, `AxesSpec`, `Candidate`, `Niche`,
  `SessionSettings`, `EngineConfig`) and axes loading/validation. The engine **never assumes a
  domain**; every command receives resolved axes. **Tuning knobs live in `EngineConfig`** (dedup τ,
  KNN k, open-niche count + freeze factor, DPP pool, novelty-ref cap, quality weight, monitor
  thresholds/margins/window), overridable per domain via an optional `engine:` block; defaults
  reproduce the original behavior and `ingest` resolves them with `load_engine_config`. The
  pipeline/monitor module constants remain only as fallback defaults for direct callers and the
  self-test (kept in sync with `EngineConfig` by `test_engine_config`).
- `embed.py`, `novelty.py`, `diversity.py`, `monitor.py`, `archive.py` — the math (see below).
- `state.py`, `memory.py` — file-based persistence and preference memory.
- `session.py` — per-invocation context (`Session`): bundles the `State` handle, the resolved
  preference **domain** (memory namespace), the axes **spec**, and the **embedder**, all resolved
  lazily. It centralizes the rule that the memory namespace *is* the domain of the persisted axes
  snapshot, so `ingest`/`recall`/`remember`/`parents` can never drift on which namespace a
  project's memory lives in.
- `pipeline.py` — orchestration; one public function per CLI command, built on `Session` and each
  returning a JSON-serializable dict. **This is where to start reading.**
- `__main__.py` — thin argparse wrapper over `pipeline`.

**The `ingest` cycle** (`pipeline.ingest`, the heart of the loop):
`embed → dedup (cosine > per-embedder τ) → place into MAP-Elites niches → geometric k-NN novelty →
keep one elite per niche → DPP diverse slate → anti-collapse monitor`.
Subtlety worth preserving: the **monitor runs on the RAW pre-dedup generation vectors**, so a
near-duplicate batch still registers as collapsing instead of being hidden behind survivors.
Two thresholds that used to be fixed constants are now **calibrated**: the dedup τ is per-embedder
(`embed.default_dedup_tau`), and the monitor's similarity flag is relative to a rolling window of
recent generations' mean cosine (`baseline + margin`, with an absolute safety ceiling), falling back
to the absolute threshold until the window has enough samples. Crucially, once calibrated the window
trains only on **healthy** generations — `_persist_cycle` excludes any generation the relative rule
flags as too similar — so a sustained collapse can't drag the baseline up past itself and blind the
monitor (a "boiling-frog" failure). While still bootstrapping (no calibrated baseline yet) every
generation is admitted, so the window can form even under an embedder whose natural cosine scale
trips the absolute fallback.

**Prefilter guard** (the load-bearing invariant, mechanically sensed): the monitor covers the
*generation* stage (it runs on raw submitted vectors), but the agent's **prefilter** — dropping
candidates as "off-brief" — is the one stage the engine never sees, so an agent could cut variety
under cover of validity. `ingest` therefore compares `len(cand_list)` **submitted** (pre-dedup, *not*
post-dedup survivors — dedup is the engine's own job) against `candidates_per_generation`, and emits a
**soft** `monitor.under_generation` flag (with `submitted`/`target_candidates`) when it falls below
`engine.under_generation_ratio` (default 0.6). The flag is advisory: it never touches `collapsing` or
the calibration window; `SKILL.md`/`loop.md` tell the agent to generate more / prefilter less.

**Niching** (`archive.py`): a niche key combines one bucket per axis — `categorical` → the value,
`continuous` → bin index over its range, `open` → a **frozen Voronoi cell** over the *embedding* of
the axis value (`CVTNicher`). The open-axis partition is **data-adaptive (fit-once-then-freeze)**:
early cycles use deterministic cold-start centroids seeded by `--seed`; once `OPEN_NICHE_FREEZE_FACTOR
* OPEN_NICHES` (now **2 × 24 = 48**, ~4–5 generations — lowered from 4× so the data-adaptive partition
actually activates in a realistic session) mechanism embeddings have accumulated, k-means is fit
**once** (`KMeans(random_state=seed)`, with the benign "fewer distinct clusters than k" warning
silenced — clustered idea embeddings legitimately under-fill the partition), the centroids are
persisted (`open_nicher.json`), and the archive is **re-keyed** onto them (`Archive.rekey_open_axis`,
merging collapsed niches by the elite rule). It never refits after freezing, so niche ids stay stable.
Exactly one axis may be the `primary_novelty` "open" axis (the novelty carrier). Within a niche, higher
`fitness` wins; ties break toward higher novelty. Most short sessions never reach the threshold and run
entirely on the cold-start partition (validated good under the real embedder), so `ingest`/`metrics`
surface an **`open_axis`** progress block (`accumulated` / `freeze_threshold` / `progress` / `frozen`)
to make the otherwise-silent freeze observable.

**Embedders** (`embed.py`), selected by `CREATIVITY_EMBEDDER` (`static` default):
- `static` — model2vec `minishlab/potion-multilingual-128M`, **256-dim, 101 languages**, CPU,
  **numpy-only inference (no torch)**, ~120 MB, lazily downloaded. The default for real runs.
- `local` — sentence-transformers `BAAI/bge-small-en-v1.5`, **384-dim, English-only**, CPU, pulls
  the ~2 GB torch stack (opt-in via `requirements-local.txt`); the high-fidelity escape hatch.
- `hash` — deterministic char-n-gram `HashingVectorizer`, no downloads (tests + offline selftest).
- `api` — a stub; constructing it is cheap, embedding raises until a backend is wired in `embed.py`.

Switching the default from `local` (384-dim) to `static` (256-dim) is **breaking for projects
persisted under the old default**: `_guard_embedding_dim` refuses to mix widths, so an old project
must be re-embedded or pinned to `CREATIVITY_EMBEDDER=local`. Each family has its own dedup τ in
`DEDUP_TAU_BY_EMBEDDER` (`static: 0.93`, calibrated on an EN+ES near-dup/distinct sample).

All embedders return **L2-normalized rows**, so cosine similarity is a plain dot product — this
assumption is relied on throughout the math modules. `ingest` guards against mixing embedding
dimensions within a project and fails loudly.

**Axes resolution** (per session, done by the agent in `SKILL.md`, validated by `config.py`): named
domain config → inferred-and-confirmed axes → `config/domains/generic.yaml` fallback. Domain
templates live in `config/domains/examples/*.yaml`; `_schema.md` documents the format. Nothing
about a domain is baked into the plugin or engine.

**State** (`state.py`) is written **outside the plugin** so reinstalls don't wipe it:
`~/.creativity-amplifier/<project>/` (override the base with `CREATIVITY_AMPLIFIER_HOME`). Writes
are atomic (temp file + `os.replace`). Per-project files are `meta.json` (project/session
settings), `axes.json` (the resolved axes geometry — kept separate from settings so the engine's
`AxesSpec` stays pure), `archive.json`, `candidates.json`, `embeddings.json`, and `open_nicher.json`
(the frozen CVT centroids, written once the open-axis partition freezes — see Niching). Preference memory
(`comparisons.jsonl`, `pins.json`) lives in a per-domain sub-directory, **namespaced per domain**
so switching domains keeps preferences separate. A per-project `tmp/` scratch dir (created by
`State.ensure`, surfaced by the `paths` command) holds the skill's hand-off files (`axes.json`,
`candidates.json`, `event.json`) inside the state home so they never clutter the user's cwd or
collide across concurrent sessions. `candidates.json`/`embeddings.json` are rewritten whole each
cycle; for long sessions `ingest` **prunes** records nothing reads again once the store exceeds
`engine.state_prune_threshold` (default 2000, 0 disables) — keeping exactly archive elites, pins, and
comparison ids, so the pruning is output-neutral.

**The self-test is the correctness contract** (`selftest.py`). It enforces a **value gate** — the
engine's diverse slate must beat a single-shot baseline on mean pairwise distance, Vendi score, and
niche entropy, and DPP must beat naive first-N on the *same* pool — plus an **induced-collapse
reversal** (a samey generation trips the monitor; the next generation recovers once diversity
pressure rises). Treat a `selftest` failure as a real regression in the diversity guarantees.

## Conventions & gotchas

- **`requirements.txt` is the single source of truth for dependencies.** `pyproject.toml` keeps
  `dependencies = []` on purpose; `setup.sh` runs `pip install -r requirements.txt` then
  `pip install -e . --no-deps`.
- Tests are hermetic via `tests/conftest.py` (forces `CREATIVITY_EMBEDDER=hash` and an isolated
  `CREATIVITY_AMPLIFIER_HOME`). Keep new tests offline — never trigger a model download.
- Determinism matters: niching/DPP/CVT take a `--seed`; reuse it across a session's cycles.
- `config.ConfigError` messages are user-facing (printed by the CLI) — write them for the operator.
- `docs/PAPER.md` is the reference-architecture paper (rationale and positioning), not the
  implementation spec — `SKILL.md` + this file are the spec.
