---
name: ideate
description: >
  Generate a diverse, non-cliché slate of ideas/concepts from a brief in ANY
  domain, using a blind-variation + diverse-archive loop with the user selecting
  in chat. Use when the user asks to brainstorm, ideate, explore an idea space,
  find fresh/original angles, generate many distinct options, or escape clichéd
  or samey concepts — regardless of subject (marketing, product, research,
  naming, design, fiction, strategy, etc.). The deterministic diversity engine
  (embeddings, MAP-Elites, novelty, DPP, anti-collapse monitor) runs locally; you
  do the generating and judging.
allowed-tools: Bash, Read, Write
---

# Creativity Amplifier — ideate

Brief: $ARGUMENTS

You amplify creativity by pairing **your** generation + judgment with a local
**diversity engine** that owns the anti-convergence math. Diversity is decoupled
from quality: geometry (the engine) decides what is *new*; you only filter what
is *valid/on-brief* and rank *within* a niche. Never let the judge pick the final
slate. The user is the real selector.

Follow `${CLAUDE_SKILL_DIR}/references/loop.md` exactly. Summary of one session:

1. **Locate the engine interpreter.** The venv auto-provisions in the background when
   the plugin loads, so it is usually ready already. Read the interpreter pointer from
   the **first** of these that exists, and set `ENGINE = "<PYBIN>" -m creativity_engine`
   (quote `<PYBIN>` — Windows paths may contain spaces):
   - `${CLAUDE_PLUGIN_DATA}/venv/engine-python.txt`  (marketplace install)
   - `${CLAUDE_SKILL_DIR}/.venv/engine-python.txt`   (dev `--plugin-dir` / `setup.sh`)
2. **If neither pointer exists yet**, the one-time setup is still running, hasn't
   started, or **failed in the background**. First check the background log so you
   don't re-run blind — it sits next to the venv:
   `tail -n 40 "${CLAUDE_PLUGIN_DATA}/provision.log"` (marketplace) or
   `tail -n 40 "${CLAUDE_SKILL_DIR}/provision.log"` (dev); relay any real failure it
   shows. Then tell the user **once**: "⏳ Setting up the creativity engine — a one-time
   download of ML libraries and a small embedding model. This can take a few minutes;
   I'll continue automatically when it's ready." Then run the bootstrap in the
   foreground (idempotent; it waits for any in-progress background provision):
   `"<PY>" "${CLAUDE_SKILL_DIR}/scripts/bootstrap.py" --venv "${CLAUDE_PLUGIN_DATA}/venv"`
   where `<PY>` is `python3` (macOS/Linux/WSL) or `py`/`python` (Windows). When it
   finishes, re-read the pointer. If the venv still can't be built, show the fresh
   `provision.log` tail alongside the bootstrap's error and stop — together they are
   the actionable diagnosis (e.g. Python 3.11+ missing). Then choose a short `PROJECT`
   id for this session.
3. **Resolve axes for this session** (diversity is only meaningful relative to a
   set of descriptor axes). Cascade:
   - if the user named a domain that has a config in
     `${CLAUDE_SKILL_DIR}/config/domains/examples/`, load it; else
   - **infer** 4–6 descriptor axes from the brief using
     `${CLAUDE_SKILL_DIR}/references/axis_inference.md` (mark exactly one `open`
     axis as the primary novelty carrier) and **confirm them with ONE short
     question** the user can accept or tweak; else
   - load `${CLAUDE_SKILL_DIR}/config/domains/generic.yaml`.
   Resolve the per-project scratch dir with `ENGINE paths --project PROJECT` and
   use its `tmp` field as `$TMP` (inside the state home, never your cwd) for every
   hand-off file. Write the resolved axes to `$TMP/axes.json`, then run
   `ENGINE init-project --project PROJECT --axes $TMP/axes.json` and
   `ENGINE recall --project PROJECT`.
4. **Generate** candidates yourself using
   `${CLAUDE_SKILL_DIR}/references/operators.md`. Apply several different
   operators; for each candidate report its `descriptor` on the resolved axes and
   its `genealogy` (parent ids + operator id). Push for variety — each new
   approach must differ from the ones already shown.
5. **Prefilter** yourself using `${CLAUDE_SKILL_DIR}/references/judge_rubric.md`
   to drop only invalid / off-brief candidates. NEVER judge novelty here. You may
   attach a within-niche `fitness` (0–1); you may NOT use it to cut variety.
6. **Ingest.** Write survivors to `$TMP/candidates.json` and run
   `ENGINE ingest --project PROJECT --candidates $TMP/candidates.json --axes $TMP/axes.json`.
7. **Present** the returned `slate` (show each idea with its niche `coords` so the
   user can judge distinctness). Ask only the returned `ask_pairs` as short
   A-vs-B questions. Let the user pin "stepping stones". Note on the `novelty`
   field: it is **mean k-NN distance to this session's own ideas (elites + this
   batch)** — a *variety* proxy, NOT originality vs. prior art / the world. Read a
   high `novelty` as "unlike the other ideas in this run", and don't present it to
   the user as proof an idea is novel to the world.
8. **Record & continue.** For each answer/pin run `ENGINE remember`; then
   `ENGINE parents` to get diverse parents and loop from step 4, or stop on the
   user's command.
9. **React to the monitor.** If `monitor.collapsing` is true, raise diversity
   directives next round (new operators, forbid the crowded niches, demand
   distance from recent ideas) — never remove or bypass the monitor. If
   `monitor.under_generation` is true, you over-prefiltered: next round generate the
   full target and cut **only** invalid/off-brief ideas, never the merely unusual.

Read `${CLAUDE_SKILL_DIR}/references/loop.md` for exact JSON shapes, engine
contracts, and steering tactics. Never hard-code a domain — always use the axes
resolved in step 3.
