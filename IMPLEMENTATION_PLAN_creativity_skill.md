# Implementation Plan — Creativity Amplifier (Claude Code plugin)

A build plan for **Claude Code** to autonomously implement the lightweight,
local, server-less creativity-amplifier described in
`blueprint_creativity_skill_english.md`, packaged as a **Claude Code plugin
containing one model-invoked skill**.

The engine's deterministic parts (embeddings, MAP-Elites archive, geometric
novelty, DPP diverse selection, anti-collapse monitor, local state) run as a
bundled **Python CLI**. The LLM parts (variation operators, the judge prefilter)
are performed by the **agent itself** (Claude), so no extra chat-LLM key is
needed. The human stays in the loop only as the in-chat selector.

---

## Execution model (read first)

- **Autonomous, phase by phase.** Every phase ends with **machine-checkable
  acceptance criteria** (pytest, `claude plugin validate`, and a stubbed
  end-to-end `selftest`). Claude Code runs them itself and proceeds to the next
  phase without waiting for human sign-off.
- **One optional question, non-blocking.** In Phase 0 ask the user once which
  embedding provider to use. **Default to local `sentence-transformers` (no API
  key, CPU-only)** and continue immediately if there is no answer. No other
  human gate exists in this plan.
- **Determinism.** All randomized steps take a fixed `--seed`. Tests assert on
  seeded runs so they are reproducible.
- **Do not break these invariants** (from the blueprint): diversity/novelty is
  **decoupled from the judge** (geometry owns novelty; the judge only ranks
  within a niche and filters validity/on-brief); embeddings use a **different
  model family** from the agent; the anti-convergence machinery is **never**
  removed or bypassed.

---

## 0. Claude Code technical requirements (verified against official docs)

Sources: `https://code.claude.com/docs/en/plugins`, `https://code.claude.com/docs/en/skills`.

**Prerequisites**
- Claude Code installed and authenticated, latest version (so `/plugin` and
  `claude plugin validate` exist).
- Python 3.11+ available on PATH.

**Plugin shape**
- A plugin is a directory. The manifest is `.claude-plugin/plugin.json` with
  `name`, `description`, `version`, optional `author`. The `name` is the skill
  namespace (`/<plugin>:<skill>`).
- **Only `plugin.json` goes inside `.claude-plugin/`.** `skills/`, `config`,
  `bin/`, etc. live at the **plugin root**.
- Skills live in `skills/<skill-name>/SKILL.md`. The folder name is the skill
  name. Skills are **model-invoked**: Claude triggers them from the `description`
  frontmatter, so the description must be specific.

**SKILL.md**
- YAML frontmatter (all optional except a recommended `description`):
  `name`, `description`, `disable-model-invocation`, `allowed-tools`.
- `allowed-tools` is honored by the **Claude Code CLI** (our target). Scope it,
  e.g. `allowed-tools: Bash, Read, Write`.
- Body is Markdown instructions. `$ARGUMENTS` captures user input after the skill
  name. Keep SKILL.md concise (≈ under 5,000 words) and use **progressive
  disclosure**: put detailed operator prompts, the judge rubric, and the long
  loop in `references/` files that are read on demand.
- Bundled assets: `scripts/` (Python/Bash, executed via Bash, do **not** consume
  context until run) and `references/` (docs loaded on demand). Reference them
  with the substitution variable **`${CLAUDE_SKILL_DIR}`** (the absolute path of
  the directory containing this SKILL.md), so paths work regardless of the
  current working directory.

**Local development / testing**
- Load without installing: `claude --plugin-dir ./creativity-amplifier`.
- After edits: `/reload-plugins` (SKILL.md text hot-reloads; structural changes
  may need a reload).
- Validate: `claude plugin validate` (run before considering any phase done).

> Implementation note for the agent: confirm the exact current frontmatter field
> list and substitution-variable names against `/en/skills` and
> `/en/plugins-reference` at the start of Phase 0, in case the docs changed.

---

## 1. What we are building

A skill that, given a creative brief, runs the blind-variation → diverse-archive
→ human-selection loop and returns a **diverse, non-cliché slate** of campaign
**angles/concepts**, navigable and steerable in chat.

**Runtime data flow (one cycle)**

1. User invokes the skill with a brief. The skill loads the domain config and
   `recall`s recent preferences from local state.
2. **Variation (agent):** Claude applies operator prompt modules
   (`references/operators.md`) to produce N candidate angles, each with
   self-reported descriptor coordinates and genealogy (parent ids, operator id).
   Diversity pressure is in the prompt ("produce mechanisms unlike these …").
3. **Judge prefilter (agent):** Claude applies `references/judge_rubric.md`
   (skeptical, anti-cliché persona) to drop invalid / off-brief candidates only —
   **not** to judge novelty. Pairwise where helpful.
4. **Engine `ingest` (Python):** embeds survivors with a different-family
   embedder, dedups, places them into MAP-Elites niches, computes geometric
   novelty (k-NN), updates the archive (one elite per niche), runs **DPP** to
   pick a diverse slate, and runs the **anti-collapse monitor**. Returns the
   slate, the A-vs-B pairs worth asking, and monitor status as JSON.
5. **Human (chat):** Claude shows the slate and asks the flagged A-vs-B pairs.
   The user answers in the CLI; can pin "stepping stones".
6. **Engine `remember` / `parents`:** logs choices/pins to local preference
   memory; returns diverse parents. The loop repeats or stops on user command.
   If the monitor flags collapse, Claude raises diversity directives next round.

---

## 2. Repository / plugin layout

```
creativity-amplifier/                      # plugin root (pass to --plugin-dir)
├── .claude-plugin/
│   └── plugin.json                        # manifest (name/description/version/author)
├── skills/
│   └── ideate/
│       ├── SKILL.md                       # orchestration (model-invoked, concise)
│       ├── references/
│       │   ├── loop.md                    # full loop, read on demand
│       │   ├── operators.md               # operator prompt modules
│       │   └── judge_rubric.md            # anti-cliché judge persona + rubric
│       ├── config/
│       │   └── domains/
│       │       └── marketing.yaml         # example domain config (axes, unit, rubric ptr)
│       └── scripts/
│           ├── setup.sh                   # creates .venv, installs requirements
│           ├── requirements.txt
│           └── creativity_engine/         # the Python package (CLI)
│               ├── __init__.py
│               ├── __main__.py            # argparse CLI dispatch
│               ├── state.py               # file-based state (archive, niches, prefs, pins)
│               ├── config.py              # domain YAML loader
│               ├── embed.py               # pluggable embedder (local default)
│               ├── archive.py             # MAP-Elites placement + elites
│               ├── novelty.py             # k-NN geometric novelty
│               ├── diversity.py           # DPP / greedy diverse selection
│               ├── monitor.py             # entropy + mean-cosine anti-collapse
│               ├── memory.py              # preference memory + heuristic active learning
│               └── pipeline.py            # ingest/remember/parents/selftest orchestration
├── tests/                                 # pytest (dev-only)
│   ├── test_state.py
│   ├── test_embed.py
│   ├── test_archive_novelty.py
│   ├── test_diversity_monitor.py
│   ├── test_memory.py
│   └── test_selftest_e2e.py
├── README.md
└── .gitignore                             # ignore .venv/, state dirs
```

Runtime state is written **outside** the plugin (so reinstalls don't wipe it):
`~/.creativity-amplifier/<project>/...` as defined in the blueprint's data model.

---

## 3. Tech stack & dependencies

`scripts/requirements.txt`:

```
numpy>=1.26
scipy>=1.11            # determinant/eigh for DPP
scikit-learn>=1.4      # NearestNeighbors, KMeans/CVT niching
pyyaml>=6              # domain config
sentence-transformers>=2.7   # default local embedder (CPU)
pytest>=8              # dev/test only
```

- **Embeddings:** default `sentence-transformers` model `BAAI/bge-small-en-v1.5`
  (≈33M params, CPU, different family from the agent → satisfies the lineage
  hedge). Provider is pluggable behind `embed.Embedder`; an API provider
  (Voyage/Cohere/OpenAI) can be selected via env var without touching callers.
- **No chat-LLM SDK dependency:** generation and judging are done by the agent.
- The engine is invoked as `${CLAUDE_SKILL_DIR}/.venv/bin/python -m creativity_engine <cmd>`.

---

## 4. The engine CLI (contracts)

All commands read/write **JSON** (files or stdin/stdout) and take
`--project <id> [--domain <name>] [--seed <int>]`. Exit non-zero on error.

| Command | Input | Output | Does |
| :-- | :-- | :-- | :-- |
| `init-project` | `--project --domain` | `{ok, paths}` | create state dirs, snapshot config |
| `recall` | `--project --domain [--k]` | `{preferences:[...], pins:[...]}` | return memory for in-context injection |
| `ingest` | `--candidates file.json` | `{slate:[...], ask_pairs:[...], monitor:{...}, parents:[...]}` | embed → dedup → place → novelty → archive → DPP → monitor |
| `remember` | `--event file.json` | `{ok}` | append comparison/pin to preference memory |
| `parents` | `--project --k` | `{parents:[...]}` | diverse parents for next generation |
| `metrics` | `--project` | `{entropy, mean_cosine, coverage, n}` | current archive health |
| `selftest` | `[--live]` | exit 0/1 + report | full loop with stubbed LLM + stubbed human |

**Candidate object** (produced by the agent, consumed by `ingest`):

```json
{
  "id": "c-0007",
  "text": "Angle: ...",
  "descriptor": {"audience": "...", "emotional_register": "...",
                 "channel_format": "...", "risk_edginess": 0.7, "mechanism": "..."},
  "genealogy": {"parent_ids": ["c-0001"], "operator_id": "analogy"}
}
```

**`ingest` output `slate`** items add `niche_id`, `novelty`, `embedding_ref`, and
the `ask_pairs` are `[ [idA, idB, reason] ]` chosen by the heuristic active
learner (max judge-disagreement / high-novelty-uncertain).

---

## 5. The skill (`SKILL.md`)

Keep SKILL.md short; defer detail to `references/`.

```markdown
---
name: ideate
description: >
  Generate a diverse, non-cliché slate of campaign angles/concepts from a
  creative brief, using a blind-variation + diverse-archive loop with the user
  selecting in chat. Use when the user asks to brainstorm, ideate, find fresh
  angles, or escape clichéd marketing concepts.
allowed-tools: Bash, Read, Write
---

# Creativity Amplifier — ideate

Brief: $ARGUMENTS

Follow `${CLAUDE_SKILL_DIR}/references/loop.md` exactly. Summary:

1. Ensure the engine is ready: if `${CLAUDE_SKILL_DIR}/.venv` is missing, run
   `bash ${CLAUDE_SKILL_DIR}/scripts/setup.sh`.
2. Set ENGINE = `${CLAUDE_SKILL_DIR}/.venv/bin/python -m creativity_engine`.
3. `recall` preferences; load the domain config (default `marketing`).
4. GENERATE candidates yourself using `${CLAUDE_SKILL_DIR}/references/operators.md`
   (apply several operators; report each candidate's descriptor + genealogy).
   Push for variety: each new mechanism must differ from the ones already shown.
5. PREFILTER yourself using `${CLAUDE_SKILL_DIR}/references/judge_rubric.md`
   (kill invalid/off-brief only — NEVER judge novelty here).
6. Write survivors to a temp JSON and call `ENGINE ingest --candidates <file>`.
7. Present the returned `slate` to the user; ask only the `ask_pairs` as short
   A-vs-B questions. Honor pins.
8. Call `ENGINE remember` with the user's answers; call `ENGINE parents` and loop.
9. If `monitor.collapsing` is true, increase diversity directives next round.

Never select the final slate with the judge alone; geometry (the engine) owns
diversity. The user is the real selector.
```

`references/operators.md`, `references/judge_rubric.md`, `references/loop.md`,
and `config/domains/marketing.yaml` carry the detail.

`marketing.yaml` (example):

```yaml
domain: marketing
unit_of_generation: campaign_angle
axes:
  - {name: audience, type: categorical}
  - {name: emotional_register, type: categorical}
  - {name: channel_format, type: categorical}
  - {name: risk_edginess, type: continuous, range: [0, 1]}
  - {name: mechanism, type: open, primary_novelty: true}
judge_rubric: references/judge_rubric.md
slate_size: 6
candidates_per_generation: 24
```

---

## 6. Phased build plan (autonomous, self-verifying)

Each phase: **deliverables**, **automated acceptance** (the agent runs these and
proceeds on green). No human gate except the single optional Phase-0 question.

### Phase 0 — Scaffold, deps, manifest  *(one optional question)*
- **Deliverables:** plugin dir; `.claude-plugin/plugin.json`; empty
  `skills/ideate/SKILL.md`; `scripts/requirements.txt`; `scripts/setup.sh`
  (create `.venv`, `pip install -r requirements.txt`); `.gitignore`; `README.md`
  stub. `plugin.json`:
  ```json
  {"name":"creativity-amplifier","description":"Diverse, non-cliché idea generation with a human-in-the-loop diversity engine","version":"0.1.0","author":{"name":"Sergi"}}
  ```
- **Optional question (non-blocking):** "Embedding provider — local
  sentence-transformers (default, no key) or an API provider (give key/env var)?"
  If unanswered within a brief moment, default to local and continue.
- **Acceptance:** `bash scripts/setup.sh` exits 0; `.venv/bin/python -c "import numpy,scipy,sklearn,yaml,sentence_transformers"` exits 0; `claude plugin validate` passes.

### Phase 1 — State & config core
- **Deliverables:** `state.py`, `config.py`, dataclasses for Candidate/Niche;
  `init-project`, `recall` commands.
- **Acceptance:** `pytest tests/test_state.py` green — state round-trips
  (write → read → equal); `init-project` then `recall` returns empty-but-valid
  JSON; bad domain name exits non-zero with a clear message.

### Phase 2 — Embeddings + dedup
- **Deliverables:** `embed.py` with `Embedder` interface, local default impl,
  env-var-selected API impl stub, near-duplicate suppression (cosine > τ).
- **Acceptance:** `pytest tests/test_embed.py` green — embeddings have expected
  shape and are deterministic for fixed input; dedup removes a near-duplicate and
  keeps a distinct one; provider switch via env var loads without import errors.

### Phase 3 — Anti-convergence math (the core)
- **Deliverables:** `archive.py` (CVT/grid MAP-Elites placement, elite-per-niche),
  `novelty.py` (k-NN mean distance), `diversity.py` (DPP via `scipy` log-det /
  greedy fallback), `monitor.py` (Shannon entropy over niche occupancy +
  mean pairwise cosine), and `pipeline.ingest`.
- **Acceptance:** `pytest tests/test_archive_novelty.py tests/test_diversity_monitor.py`
  green, asserting the invariants:
  - novelty is **higher** for a point far from the set than for a near point;
  - DPP-selected k items have **measurably higher** mean pairwise distance
    (and Vendi score) than a random k from the same pool, on a seeded fixture;
  - the monitor's `collapsing` flag is **true** for an injected
    low-diversity (near-duplicate) stream and **false** for a diverse one;
  - one elite per niche is maintained; the judge is **not** called anywhere in
    these modules (diversity is judge-independent — assert by construction/import).

### Phase 4 — Skill orchestration + references
- **Deliverables:** final `SKILL.md`; `references/loop.md`, `operators.md`
  (mutation, analogy/combination, transformation, reframing, SCAMPER/
  morphological/TRIZ as prompt modules), `judge_rubric.md` (skeptical anti-cliché
  persona; validity/on-brief only); `config/domains/marketing.yaml`.
- **Acceptance:** `claude plugin validate` passes; `claude --plugin-dir .`
  lists `/creativity-amplifier:ideate`; `references/*` resolve via
  `${CLAUDE_SKILL_DIR}`; SKILL.md under the size limit.

### Phase 5 — Preference memory + heuristic active learning
- **Deliverables:** `memory.py` — `remember` (append comparisons/pins),
  in-context `recall`, and `ask_pairs` selection (max judge-disagreement /
  high-novelty-uncertain); `parents` (diverse parent sampling, honoring pins).
- **Acceptance:** `pytest tests/test_memory.py` green — comparison/pin
  round-trip; on a constructed pool the active learner returns the
  most-informative pair; `parents` never drops a pinned stepping stone.

### Phase 6 — End-to-end selftest + value gate
- **Deliverables:** `pipeline.selftest` — runs the full loop with a **stubbed
  LLM** (canned candidate generator) and a **stubbed human** (auto-pick = highest
  novelty within the slate), no interactive input, no live model; plus a
  **value-gate benchmark**: compare the engine's diverse slate against a
  single-shot baseline (canned non-diverse generator) on a fixed brief.
- **Acceptance:** `python -m creativity_engine selftest` exits 0, writes all
  state files, and the **value gate passes**: diverse-slate mean pairwise
  distance / Vendi / entropy **exceeds** the single-shot baseline by a set margin
  on the seeded fixture; the **induced-collapse reversal** test passes (after the
  monitor fires and diversity pressure rises, the next-gen diversity recovers).
  `pytest tests/test_selftest_e2e.py` green.

### Phase 7 — Docs, validation, packaging
- **Deliverables:** complete `README.md` (install, `--plugin-dir` usage, the one
  config question, how to add a domain, how to run `selftest`); ensure
  `.gitignore` excludes `.venv/` and state; final `claude plugin validate`.
- **Acceptance:** `claude plugin validate` passes; full suite
  `pytest -q` green; `python -m creativity_engine selftest` exits 0; README
  present and accurate. **Definition of Done met (Section 9).**

---

## 7. Testing & self-verification strategy

- **Unit:** per-module pytest (Phases 1–5).
- **Property/invariant:** novelty monotonicity, DPP-beats-random diversity,
  monitor true/false on injected streams, one-elite-per-niche, judge-independence
  of diversity (Phase 3).
- **End-to-end stubbed:** `selftest` runs the whole loop with no human and no live
  LLM, so the agent can verify autonomously (Phase 6).
- **Value gate:** the engine's diverse slate must beat single-shot on
  diversity metrics — the blueprint's core hypothesis, checked in CI-style by the
  agent (Phase 6).
- All randomized steps use `--seed`; tests assert on seeded runs.

---

## 8. Human interaction points (minimal, by design)

- **Build time:** exactly one optional, non-blocking question (Phase 0,
  embedding provider). Everything else is gated by automated acceptance the agent
  runs itself.
- **Run time:** the skill asks the *user* short A-vs-B questions in the CLI — that
  is the product's human-in-the-loop, not a development gate, and never blocks for
  more than a brief moment.

---

## 9. Definition of Done

1. `claude plugin validate` passes.
2. `claude --plugin-dir ./creativity-amplifier` loads `/creativity-amplifier:ideate`.
3. `pytest -q` fully green.
4. `python -m creativity_engine selftest` exits 0 and the value gate +
   induced-collapse reversal both pass.
5. A live interactive run on a sample brief returns a slate whose angles are
   visibly diverse and steerable, asking only short A-vs-B questions.
6. Invariants hold: diversity decoupled from the judge; embedder is a different
   family; anti-convergence machinery present and exercised.

---

## 10. Risks & notes (for the agent)

- **Descriptor/niche fidelity (top value risk):** with an open `mechanism` axis,
  CVT niches are Voronoi cells over embeddings, so "diverse" is embedding-diverse,
  which may not equal human-perceived distinctness. Add a sanity check in the
  value gate and surface niche coordinates in the slate so the user can judge.
- **Venv bundling:** Python deps must be installed once (`setup.sh`); SKILL.md
  self-heals if `.venv` is missing. Keep `.venv/` out of version control.
- **`${CLAUDE_SKILL_DIR}` and frontmatter:** re-verify exact names against
  `/en/skills` and `/en/plugins-reference` at Phase 0 before hard-coding.
- **Latency:** generation + embedding + scoring per generation should keep first
  results quick; cap `candidates_per_generation` and DPP shortlist (~100–200).
- **Graduation:** keep the engine API stable; the same operator+diversity core is
  reused if this later graduates to the trained-proxy SaaS in the other blueprint.

---

## References (official documentation)

- Create plugins — `https://code.claude.com/docs/en/plugins`
- Agent Skills — `https://code.claude.com/docs/en/skills`
- Plugins reference — `https://code.claude.com/docs/en/plugins-reference`
