# Creativity Amplifier

A **domain-agnostic** Claude Code plugin that turns a creative brief in *any*
subject into a **diverse, non-cliché slate** of ideas, using a blind-variation →
diverse-archive → human-selection loop with you (the user) as the selector.

It ships as **one model-invoked skill** (`/creativity-amplifier:ideate`) plus a
bundled, server-less **Python engine** that owns the anti-convergence math. The
LLM parts — generating variations and the skeptical judge prefilter — are done by
the agent (Claude) itself, so **no extra chat-LLM API key is needed**.

## Why it's different

- **Diversity is decoupled from the judge.** Geometry owns *novelty* (embeddings,
  MAP-Elites niching, k-NN novelty, DPP selection); the judge only filters
  *validity / on-brief* and ranks *within* a niche. Its fitness has just a
  **bounded, low-weight** say in the DPP slate — quality-weighted diversity
  (weight 0.3, fitness clipped to a [0.7, 1.3] multiplier) — so it can never
  prune variety or collapse the slate's diversity, and **you** remain the
  final selector.
- **A different embedding family from the agent.** The local embedder is
  `sentence-transformers` (`BAAI/bge-small-en-v1.5`, CPU) — a different model
  family from Claude — so "what's novel" isn't judged by the same lineage that
  generated the ideas.
- **An anti-collapse monitor that's never bypassed.** Shannon entropy over niche
  occupancy + mean pairwise cosine flag convergence; the similarity signal is
  **calibrated to a rolling baseline** (and the dedup threshold is per-embedder),
  so it doesn't misfire when the embedder or domain changes. When it fires, the
  skill raises diversity pressure next round.
- **Axes resolved per session.** "Domain-agnostic" doesn't remove the need for
  descriptor axes — it resolves them per session (named domain → inferred &
  confirmed → generic fallback). Nothing about a domain is baked into the plugin.

## Install & run (local development)

Requirements: Claude Code (latest), Python 3.11+.

```bash
# 1. Build the engine venv (Windows / macOS / Linux)
python3 skills/ideate/scripts/bootstrap.py     # Windows: python ... or py ...

# 2. Load the plugin in Claude Code without installing it
claude --plugin-dir .

# 3. In Claude Code, invoke the skill with a brief in ANY subject
/creativity-amplifier:ideate names for a privacy-first calendar app
/creativity-amplifier:ideate research hypotheses for why week-2 retention dropped
```

Validate the plugin at any time:

```bash
claude plugin validate .          # or: claude plugin validate --strict .
```

### The one configuration choice (embedding provider)

By default the engine uses the **local** `sentence-transformers` embedder — no API
key, CPU-only, downloaded once on first use. To use a hosted provider instead, set
environment variables before launching Claude Code:

```bash
export CREATIVITY_EMBEDDER=api          # hash | local | api
export CREATIVITY_EMBED_API=voyage      # provider name (stub — wire up in embed.py)
export CREATIVITY_EMBED_API_KEY=...      # your key
```

`CREATIVITY_EMBEDDER=hash` selects a deterministic, dependency-light embedder used
by the tests and the offline self-test (no model download).

## How a session works

The skill follows `skills/ideate/references/loop.md`. One cycle:

1. **Resolve axes.** If you name a domain with a config in
   `skills/ideate/config/domains/examples/`, it's loaded. Otherwise the agent
   **infers 4–6 axes** from your brief (marking one `open` axis as the primary
   novelty carrier) and confirms them with **one short question**. If it can't,
   it falls back to `config/domains/generic.yaml`.
2. **Generate (agent).** Claude applies several variation operators
   (`references/operators.md`) to draft candidates, each with a descriptor on the
   resolved axes and genealogy.
3. **Prefilter (agent).** Claude applies `references/judge_rubric.md` to drop only
   invalid / off-brief candidates — never to cut variety.
4. **Ingest (engine).** Survivors are embedded, deduped, placed into MAP-Elites
   niches over the resolved axes (the open "mechanism" axis uses a **data-adaptive
   partition** — deterministic cold-start cells that fit once via k-means and then
   freeze, so niche ids stay stable), scored for novelty, kept one-elite-per-niche,
   and a **DPP** picks a quality-weighted diverse slate (geometry dominates; the
   judge's bounded fitness only nudges ordering). The **anti-collapse monitor** runs.
5. **Select (you).** Claude shows the slate with each idea's niche coordinates and
   asks only the most-informative A-vs-B pairs. You can **pin** stepping stones.
6. **Remember & loop.** Choices/pins go to local preference memory (namespaced per
   domain); diverse parents seed the next generation.

## Adding a domain template

Domain templates are optional conveniences — the skill works without them. To add
one, drop a YAML file in `skills/ideate/config/domains/examples/` following
`skills/ideate/config/domains/_schema.md`:

```yaml
domain: naming
unit_of_generation: name
axes:
  - {name: tone, type: categorical}
  - {name: imagery, type: categorical}
  - {name: length, type: continuous, range: [1, 4]}
  - {name: construction, type: open, primary_novelty: true}
judge_rubric: references/judge_rubric.md
slate_size: 6
candidates_per_generation: 12
```

Then a user who says "name ideas (naming)" gets these axes; everyone else gets
inferred-or-generic axes. See `references/axis_inference.md` for how inference
works. A template may also carry an optional `engine:` block to override tuning
knobs (open-niche count, dedup τ, quality weight, monitor thresholds, …) per
domain — defaults reproduce the standard behavior; see `_schema.md` for the keys.

## The engine CLI (for the curious / for tests)

```
python -m creativity_engine <command> --project <id> [--axes axes.json] [--seed N]
```

| Command | Does |
| :-- | :-- |
| `init-project` | create state dirs, snapshot the resolved axes + session settings |
| `recall` | return preference memory for in-context injection |
| `ingest` | embed → dedup → place → novelty → archive → DPP → monitor |
| `remember` | append a comparison/pin to preference memory |
| `parents` | diverse parents for the next generation (pins always kept) |
| `metrics` | current archive health (entropy, mean cosine, coverage, n) |
| `selftest` | full loop with a stubbed LLM + human; value gate + collapse reversal |

Runtime state is written **outside** the plugin (so reinstalls don't wipe it):
`~/.creativity-amplifier/<project>/...`, preferences namespaced per domain.
Override the base directory with `CREATIVITY_AMPLIFIER_HOME`.

## Running the self-test & the suite

```bash
# offline end-to-end check (stubbed LLM + human, no model download); exits 0
skills/ideate/.venv/bin/python -m creativity_engine selftest

# unit / property / e2e tests
skills/ideate/.venv/bin/python -m pytest -q
```

The self-test enforces a **value gate** — the engine's diverse slate must beat a
single-shot baseline on mean pairwise distance, Vendi score, and niche entropy,
and DPP must beat naive first-N selection on the same pool (**shuffled** so
first-N isn't trivially the near-clones, and **averaged over several seeds**),
with a **null check** that DPP doesn't regress below a random subset on an
already-uniform pool — plus an **induced-collapse reversal** (a samey generation
trips the monitor; the next generation recovers once diversity pressure rises).
A `--live` run adds a semantic sanity check (a paraphrase beats an unrelated
sentence under the real embedder), skipped cleanly when sentence-transformers
isn't installed.

## Layout

```
creativity-amplifier/                  # plugin root (pass to --plugin-dir)
├── .claude-plugin/plugin.json         # manifest
├── skills/ideate/
│   ├── SKILL.md                       # model-invoked orchestration (concise)
│   ├── references/                    # loop, operators, judge rubric, axis inference
│   ├── config/domains/                # _schema.md, generic.yaml, examples/*.yaml
│   └── scripts/
│       ├── setup.sh, requirements.txt, pyproject.toml
│       └── creativity_engine/         # the Python engine (CLI)
├── tests/                             # pytest (dev-only)
├── docs/PAPER.md                      # reference-architecture paper (rationale)
└── README.md
```

## Background

The design rationale — why diversity is owned by geometry while the judge's say in the slate stays
bounded, and how this maps onto blind-variation/selective-retention, Quality-Diversity (MAP-Elites,
novelty search), and DPP selection — is written up in [`docs/PAPER.md`](docs/PAPER.md).

## License

GNU GPL v3 — see `LICENSE`.
