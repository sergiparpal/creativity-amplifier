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
- The local embedder is deliberately a **different model family** from Claude
  (`BAAI/bge-small-en-v1.5`) so "what's novel" isn't judged by the lineage that generated the ideas.
- The **anti-collapse monitor is never bypassed** — when it flags convergence the skill raises
  diversity pressure next round; it is never removed or worked around.

## Commands

Run Python through the engine's own venv (`skills/ideate/.venv/bin/python`), not system Python.

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
| `recall` | return preference memory for in-context injection |
| `ingest` | embed → dedup → place → novelty → archive → DPP → monitor (one cycle) |
| `remember` | append a comparison/pin to preference memory |
| `parents` | diverse parents for the next generation (pins always kept) |
| `metrics` | archive health (entropy, mean cosine, coverage, n) |
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
to the absolute threshold until the window has enough samples.

**Niching** (`archive.py`): a niche key combines one bucket per axis — `categorical` → the value,
`continuous` → bin index over its range, `open` → a **frozen Voronoi cell** over the *embedding* of
the axis value (`CVTNicher`). The open-axis partition is **data-adaptive (fit-once-then-freeze)**:
early cycles use deterministic cold-start centroids seeded by `--seed`; once `OPEN_NICHE_FREEZE_FACTOR
* OPEN_NICHES` mechanism embeddings have accumulated, k-means is fit **once** (`KMeans(random_state=
seed)`), the centroids are persisted (`open_nicher.json`), and the archive is **re-keyed** onto them
(`Archive.rekey_open_axis`, merging collapsed niches by the elite rule). It never refits after
freezing, so niche ids stay stable. Exactly one axis may be the `primary_novelty` "open" axis (the
novelty carrier). Within a niche, higher `fitness` wins; ties break toward higher novelty.

**Embedders** (`embed.py`), selected by `CREATIVITY_EMBEDDER` (`local` default):
- `local` — sentence-transformers `BAAI/bge-small-en-v1.5`, CPU, lazily downloaded (real runs).
- `hash` — deterministic char-n-gram `HashingVectorizer`, no downloads (tests + offline selftest).
- `api` — a stub; constructing it is cheap, embedding raises until a backend is wired in `embed.py`.

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
so switching domains keeps preferences separate.

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
