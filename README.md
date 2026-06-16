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
- **A different embedding family from the agent.** The default embedder is
  `model2vec` (`minishlab/potion-multilingual-128M`, CPU) — a different model
  family from Claude, **multilingual (101 languages)**, and torch-free — so
  "what's novel" isn't judged by the same lineage that generated the ideas, in
  any language. A higher-fidelity English-only `BAAI/bge-small-en-v1.5` embedder
  is available as an opt-in.
- **An anti-collapse monitor that's never bypassed.** Shannon entropy over niche
  occupancy + mean pairwise cosine flag convergence; the similarity signal is
  **calibrated to a rolling baseline** (and the dedup threshold is per-embedder),
  so it doesn't misfire when the embedder or domain changes. When it fires, the
  skill raises diversity pressure next round.
- **Axes resolved per session.** "Domain-agnostic" doesn't remove the need for
  descriptor axes — it resolves them per session (named domain → inferred &
  confirmed → generic fallback). Nothing about a domain is baked into the plugin.

## Install (one command)

Requirements: Claude Code (latest), Python 3.11+ on your PATH. Works the same in the
**CLI** and the **desktop app**, on **Windows (incl. WSL), macOS, and Linux**.

In Claude Code, add this repo as a plugin marketplace and install the plugin:

```
/plugin marketplace add sergiparpal/creativity-amplifier
/plugin install creativity-amplifier@sergiparpal
```

That's it — no clone, no `setup.sh`, no `--plugin-dir`. On the next session start, the
plugin **provisions its own Python engine in the background**: it builds a virtualenv
and installs the default stack (numpy, scikit-learn, and the multilingual
`model2vec` embedder, `minishlab/potion-multilingual-128M`). The embedder weights are
**~120 MB** and need **only numpy at inference — no PyTorch**, so the first-run download
is small and fast; it runs **non-blocking**, so Claude Code stays usable the whole time.
The venv is stored in the plugin's persistent data directory, so it **survives plugin
updates**.

If `uv` is on your PATH it's used for a faster install; otherwise the bundled
`python -m venv` + `pip` path is used. Nothing is auto-installed onto your system
beyond that venv.

Then invoke the skill with a brief in ANY subject:

```
/creativity-amplifier:ideate names for a privacy-first calendar app
/creativity-amplifier:ideate research hypotheses for why week-2 retention dropped
```

If you run `/ideate` before the one-time setup has finished, the skill tells you it's
**setting up the engine** and continues automatically once it's ready — you never have
to run a setup step yourself. (If Python 3.11+ isn't found, it says so with a fix.)

## What happens when you run it

You don't need any of the math below to use the plugin — a session is just a
back-and-forth chat, and **you are the one choosing the ideas**. Here is what each
step looks like and what you do.

1. **You give a brief.** Run `/creativity-amplifier:ideate <your brief>` for whatever
   you want ideas about — product names, campaign angles, plot twists, research
   hypotheses, anything. That's the only command you have to remember.
2. **Claude confirms the "angles" — one quick question.** Before generating, Claude
   proposes a handful of *axes*: the dimensions it will deliberately spread ideas
   across (for names, say: tone, imagery, length, and how the name is built). It asks
   **one** short question to confirm them. Just reply "ok", or tweak in plain words —
   "make it edgier", "drop the length one". *(If you named a known domain it skips
   straight ahead.)*
3. **Claude shows you a varied slate.** Claude drafts a batch of ideas, quietly drops
   any that are off-brief or don't make sense, and the engine picks a few that are as
   *different from one another* as possible — not just the "best" ones. Each idea comes
   with a short note on why it counts as distinct, so you can see the spread. **You
   read them over.**
4. **You steer with a couple of quick comparisons.** Claude asks only the few most
   useful **A-vs-B** questions — "which points in a better direction, A or B?" (you can
   also say "neither"). You can **pin** any idea you like as a "stepping stone" to keep
   exploring from. Reacting and pinning is the main thing you do.
5. **Claude runs another round, building on your picks.** Using what you preferred and
   pinned, Claude generates a fresh batch — pushing for ideas that are genuinely *new*,
   not variations on the same theme. If things start looking samey, a built-in monitor
   notices and forces more variety the next round.
6. **Repeat until you're happy, then stop.** Loop as many rounds as you like; say
   "stop" (or "that's enough") when you have what you need. Your preferences are
   remembered for the next time you ideate in the same kind of domain.

In short: **you give a brief, confirm the angles once, then react to slates and pin
favorites while Claude keeps widening the search** until you're satisfied. The rest of
this README is for people who want to develop on it or understand the internals.

## Local development (fallback)

To hack on the plugin from a checkout instead of installing it:

```bash
# 1. Build the engine venv (Windows / macOS / Linux)
python3 skills/ideate/scripts/bootstrap.py     # Windows: python ... or py ...
#    or, equivalently:  bash skills/ideate/scripts/setup.sh

# 2. Load the plugin in Claude Code without installing it
claude --plugin-dir .

# 3. Invoke the skill
/creativity-amplifier:ideate names for a privacy-first calendar app
```

In this mode the venv lives at `skills/ideate/.venv/` and the engine is installed
editable, so source edits take effect without a rebuild. Validate the plugin at any
time:

```bash
claude plugin validate .          # or: claude plugin validate --strict .
```

### The one configuration choice (embedding provider)

By default the engine uses the **static** `model2vec` embedder
(`minishlab/potion-multilingual-128M`) — no API key, CPU-only, **multilingual**,
torch-free (~120 MB), downloaded once on first use. Select a different provider with
the `CREATIVITY_EMBEDDER` environment variable before launching Claude Code:

```bash
export CREATIVITY_EMBEDDER=local   # static | local | hash | api
```

- **`static`** (default) — `potion-multilingual-128M`, 256-dim, 101 languages,
  numpy-only inference.
- **`local`** — higher-fidelity **English-only** `BAAI/bge-small-en-v1.5` (384-dim).
  Pulls the ~2 GB PyTorch stack, so it's **opt-in**: install it with
  `pip install -r skills/ideate/scripts/requirements-local.txt`.
- **`hash`** — deterministic, dependency-light char-n-gram embedder used by the tests
  and the offline self-test (no model download).
- **`api`** — an **extension point** for a hosted provider (Voyage/OpenAI/Cohere). It
  is a stub: constructing it is cheap, but embedding raises until you wire a backend
  into `embed.py`.

> **Switching the default is breaking for existing projects.** `static` is 256-dim and
> `local` is 384-dim; the engine refuses to mix embedding widths within one project. A
> project created/persisted under one embedder can't be re-ingested under another —
> start a **new** project, or pin the original embedder (e.g.
> `export CREATIVITY_EMBEDDER=local`) and re-embed if you must switch.

## How a session works (under the hood)

The plain-language walkthrough is in [**What happens when you run it**](#what-happens-when-you-run-it)
above; this is the same loop with the internals. The skill follows
`skills/ideate/references/loop.md`. One cycle:

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
sentence under the real default `static` embedder), skipped cleanly when that
embedder can't be built or downloaded.

## Layout

```
creativity-amplifier/                  # plugin root
├── .claude-plugin/
│   ├── plugin.json                    # manifest
│   └── marketplace.json               # marketplace entry (for /plugin marketplace add)
├── hooks/
│   ├── hooks.json                     # SessionStart hook: auto-provision the venv
│   ├── provision.sh                   # POSIX launcher (sh / Git Bash / WSL)
│   └── provision.ps1                  # Windows-PowerShell launcher
├── skills/ideate/
│   ├── SKILL.md                       # model-invoked orchestration (concise)
│   ├── references/                    # loop, operators, judge rubric, axis inference
│   ├── config/domains/                # _schema.md, generic.yaml, examples/*.yaml
│   └── scripts/
│       ├── bootstrap.py               # cross-platform self-provisioning installer
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
