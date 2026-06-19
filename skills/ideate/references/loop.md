# The full ideate loop

This is the authoritative procedure. SKILL.md is the summary; when in doubt,
follow this file. Everything is domain-agnostic: the only domain knowledge is the
**axes** you resolve in step 1, per session.

Set once:

```
ENGINE = "<PYBIN>" -m creativity_engine    # <PYBIN> = contents of the engine-python.txt
                                           # pointer (see step 0 for its location)
PROJECT = <short slug for this session, e.g. "campaign-jun14">
```

All engine commands read/write JSON and take `--project PROJECT`. Randomized
steps take `--seed <int>` — reuse the same seed within a session for
reproducibility.

**Scratch files go in the project's `tmp/`, never your cwd.** Once the engine is
ready (step 0), resolve the per-project scratch dir and reuse it for every hand-off
file (`axes.json`, `candidates.json`, `event.json`):

```
ENGINE paths --project PROJECT      # -> {"root","meta","axes",...,"tmp"}; sets up the dir
TMP = <the "tmp" field of that output>   # e.g. ~/.creativity-amplifier/<project>/tmp
```

Writing scratch files under `$TMP` (inside the state home, `~/.creativity-amplifier`
or `$CREATIVITY_AMPLIFIER_HOME`) keeps the user's working directory clean and avoids
collisions across concurrent sessions. Never write these files to the cwd.

---

## 0. Ensure the engine

The engine venv **auto-provisions in the background** when the plugin loads (a
`SessionStart` hook runs `bootstrap.py`), so it is usually ready before you need it.
Find the interpreter pointer at the **first** path that exists:

- `${CLAUDE_PLUGIN_DATA}/venv/engine-python.txt`  — marketplace install (the venv
  lives in the plugin's persistent data dir, so it survives plugin updates);
- `${CLAUDE_SKILL_DIR}/.venv/engine-python.txt`    — dev install (`--plugin-dir .`
  or `bash scripts/setup.sh`).

Read that file; its contents are `<PYBIN>`.

**If neither pointer exists yet**, the one-time setup is still running, hasn't
started, **or failed in the background**. The detached worker logs to
`provision.log` next to the venv — check it **first** so you don't re-run blind:

```
# the log sits at <venv-parent>/provision.log, i.e. the FIRST that exists:
tail -n 40 "${CLAUDE_PLUGIN_DATA}/provision.log"   # marketplace install
tail -n 40 "${CLAUDE_SKILL_DIR}/provision.log"     # dev install
```

If the tail shows a real failure (e.g. a wheel won't compile, no network, Python
too old), relay it to the user — that is the actual diagnosis. Then tell the user
once that you're setting up the engine (a one-time, multi-minute download of ML
libraries + a small embedding model) and finish/await it in the foreground
(idempotent; waits for any in-progress background provision):

```
"<PY>" "${CLAUDE_SKILL_DIR}/scripts/bootstrap.py" --venv "${CLAUDE_PLUGIN_DATA}/venv"
# <PY> = python3 (macOS/Linux/WSL) or py / python (Windows)
```

`bootstrap.py` creates the venv (using `uv` if it is on PATH, else `python -m venv`)
and installs deps — the **static** multilingual CPU embedder
(`minishlab/potion-multilingual-128M`, no API key, ~120 MB, torch-free) by default.
Re-read the pointer afterwards. **If it still fails**, show the fresh tail of
`provision.log` again alongside the foreground error and stop — the combination is
the actionable diagnosis (e.g. Python 3.11+ missing, or a wheel that won't build on
this OS). For the higher-fidelity English-only embedder, the user installs
`requirements-local.txt` and sets `CREATIVITY_EMBEDDER=local`; a hosted embedder is
`CREATIVITY_EMBEDDER=api` plus provider env vars (a stub until wired up).

---

## 1. Resolve axes (per session)

Diversity is only meaningful relative to descriptor axes. Resolve them with this
cascade, then snapshot them.

**(a) Named domain.** If the user names a domain that has a config in
`${CLAUDE_SKILL_DIR}/config/domains/examples/` (e.g. they say "marketing
campaign"), read that YAML and use it.

**(b) Inferred + confirmed.** Otherwise infer 4–6 axes from the brief following
`axis_inference.md`. Mark exactly one `open` axis as `primary_novelty: true` —
this is the main novelty carrier. Then ask the user ONE short confirmation
question, e.g.:

> I'll spread ideas across these axes: **audience, register, format, edginess
> (0–1), and mechanism** (the main novelty axis). Want to change any?

Accept a quick tweak or an "ok". Do not block for long.

**(c) Generic fallback.** If you cannot infer well-separated axes, load
`${CLAUDE_SKILL_DIR}/config/domains/generic.yaml` and tell the user you used
neutral axes.

Write the resolved axes to `$TMP/axes.json` (the scratch dir from the top of this
file) in this shape:

```json
{
  "domain": "ad-hoc:campaign-ideas",
  "unit_of_generation": "concept",
  "axes": [
    {"name": "audience", "type": "categorical"},
    {"name": "register", "type": "categorical"},
    {"name": "format", "type": "categorical"},
    {"name": "edginess", "type": "continuous", "range": [0, 1]},
    {"name": "mechanism", "type": "open", "primary_novelty": true}
  ],
  "slate_size": 6,
  "candidates_per_generation": 12
}
```

Then:

```
ENGINE init-project --project PROJECT --axes $TMP/axes.json --seed 7
ENGINE recall --project PROJECT          # returns {domain, preferences, pins}
```

Inject the `recall` output into your context — it carries the user's prior
A-vs-B choices and pinned stepping stones for this domain.

---

## 2. Generate (you)

Use `operators.md`. Produce `candidates_per_generation` ideas by applying
**several different operators** (don't lean on one). Each candidate is an object:

```json
{
  "id": "c-0007",
  "text": "Idea: <the concept, concretely>",
  "descriptor": {"audience": "...", "register": "...", "format": "...",
                 "edginess": 0.7, "mechanism": "<the core how, in a few words>"},
  "genealogy": {"parent_ids": ["c-0001"], "operator_id": "analogy"}
}
```

Rules:
- `descriptor` keys must be exactly the resolved axis names. Continuous axes get a
  number in range; the `open`/primary axis gets a short free-text "how".
- Make each new idea differ from the ones already shown. If you have `parents`
  from a previous cycle, treat them as stepping stones to vary, not repeat.
- Concrete beats abstract. One vivid sentence is better than a vague category.

---

## 3. Prefilter (you, as the skeptical judge)

Apply `judge_rubric.md`. Drop only candidates that are **invalid, off-brief, or
incoherent**. Do NOT cut ideas for being weird, risky, or unlike the others —
that is exactly the variety the engine needs. Optionally attach `fitness` (0–1)
for within-niche ranking; never use it to reduce diversity. Pairwise comparison
is allowed where it sharpens a validity call.

Write survivors to `$TMP/candidates.json` (a JSON list, or `{"candidates": [...]}`).

---

## 4. Ingest (engine owns diversity)

```
ENGINE ingest --project PROJECT --candidates $TMP/candidates.json --axes $TMP/axes.json --seed 7
```

Returns:

```json
{
  "slate": [ {"id","text","descriptor","coords","niche_id","novelty","fitness","embedding_ref"}, ... ],
  "ask_pairs": [ ["idA","idB","why this pair is worth asking"], ... ],
  "monitor": {"collapsing": false, "mean_cosine": 0.18, "entropy": 2.1,
              "normalized_entropy": 0.88, "coverage": 9, "n": 12, "reasons": [],
              "submitted": 12, "target_candidates": 12, "under_generation": false},
  "parents": ["id", "..."],
  "open_axis": {"present": true, "frozen": false, "partition": "cold_start",
                "accumulated": 12, "freeze_threshold": 48, "progress": 0.25}
}
```

`submitted` / `target_candidates` / `under_generation` are the **prefilter guard**:
the engine sees only the candidates you submitted, so if you prefiltered away so many
that `under_generation` is `true`, you may be cutting variety under cover of validity
(see step 7).

`open_axis` is **observability only** (no action needed): it reports the data-adaptive
mechanism partition's progress toward its one-time freeze. Most sessions stay on the
`cold_start` partition (good on its own); it flips to `frozen` once `accumulated`
reaches `freeze_threshold`.

The engine embedded survivors with a **different model family** from you,
deduped near-duplicates, placed each into a MAP-Elites niche over the resolved
axes, scored geometric novelty, kept one elite per niche, and ran DPP to choose
the diverse `slate`. You did not pick the slate — geometry did.

---

## 5. Present + ask (human in the loop)

Show the `slate`. For each idea, show its `coords` (the niche buckets) so the user
can see *why* it is considered distinct — e.g. "mechanism=cell7, edginess=b3".
Surfacing coordinates is important: embedding-diversity is not always
human-perceived distinctness, so let the user judge.

Then ask ONLY the `ask_pairs`, as short A-vs-B questions:

> Which sharpens the direction better — **A** (…) or **B** (…)? (or "neither")

Keep the *comparison* questions to the returned pairs. But then **explicitly invite
the user to pin any idea(s) they want to keep exploring from — including ideas that
were not in the asked pairs**, e.g.:

> Want me to pin any of these as a stepping stone? Any of them — not just A/B.

Pinning is deliberately *not* limited to the asked pairs, and it is the user's main
lever. The `ask_pairs` only refine a **low-weight** preference signal (`fitness`,
clipped and weighted so it can never prune variety or pick the slate); a **pin** is
the strong, durable signal — it is always kept as a parent for the next generation
(step 6, `select_parents` never drops a pin) and recalled across sessions. So when
the user likes a slate idea you didn't ask about, prompt them to pin it instead of
letting it pass unrecorded. Pins **and** any free-form comparison the user volunteers
(beyond the two suggested pairs) are both recorded in step 6.

---

## 6. Remember + parents + loop

For each user answer, record it:

```
# a pairwise preference
echo '{"type":"comparison","winner":"idA","loser":"idB","context":"..."}' > $TMP/event.json
ENGINE remember --project PROJECT --event $TMP/event.json

# a pin
echo '{"type":"pin","id":"idA"}' > $TMP/event.json
ENGINE remember --project PROJECT --event $TMP/event.json
```

Then fetch diverse parents for the next generation (pins are always honored):

```
ENGINE parents --project PROJECT --k 4 --seed 7
```

Loop back to step 2 with those parents, or stop when the user is satisfied. At any
time, `ENGINE metrics --project PROJECT` reports `{entropy, mean_cosine,
coverage, n, open_axis}` for the archive (`open_axis` = mechanism-partition freeze
progress).

---

## 7. React to the monitor

**Collapse.** If `monitor.collapsing` is true (or `reasons` is non-empty), the search
is converging. Next round, raise diversity pressure:
- switch to operators you haven't used (analogy/biomimicry/constraint-randomizer);
- explicitly forbid the crowded niches ("no more X-mechanism ideas");
- demand each new idea be far from the recent set;
- widen an axis (e.g. push edginess to its extremes).

**Under-generation (prefilter guard).** If `monitor.under_generation` is true, far
fewer candidates reached `ingest` than the per-generation target — you likely
**over-prefiltered**. The engine deduplicates and judges novelty itself, so your only
job in the prefilter is to drop the *invalid / off-brief / incoherent*. Next round:
- generate the full `candidates_per_generation` and prefilter **less** — keep weird,
  risky, or off-beat ideas; they are the variety the geometry needs;
- only the genuinely invalid should be cut, never the merely unusual.

Never remove, bypass, or "trust the judge instead of" the monitor. The
anti-convergence machinery is the point of the tool.

---

## Steering tactics (optional, on user request)

- **Navigate the archive:** ask for ideas in a specific niche, or "more like the
  pinned one but bolder".
- **Constrain an axis:** lock `register=deadpan` and vary the rest.
- **Zoom on mechanism:** request several distinct mechanisms for the same surface
  idea (varies only the primary-novelty axis).
- **Stop conditions:** user says stop, or coverage/diversity plateaus across two
  cycles with no new niches.
