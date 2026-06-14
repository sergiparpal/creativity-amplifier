# The full ideate loop

This is the authoritative procedure. SKILL.md is the summary; when in doubt,
follow this file. Everything is domain-agnostic: the only domain knowledge is the
**axes** you resolve in step 1, per session.

Set once:

```
ENGINE = "<PYBIN>" -m creativity_engine    # <PYBIN> = contents of
                                           # ${CLAUDE_SKILL_DIR}/.venv/engine-python.txt
PROJECT = <short slug for this session, e.g. "campaign-jun14">
```

All engine commands read/write JSON and take `--project PROJECT`. Randomized
steps take `--seed <int>` — reuse the same seed within a session for
reproducibility.

---

## 0. Ensure the engine

If `${CLAUDE_SKILL_DIR}/.venv` does not exist, run once:

```
python3 ${CLAUDE_SKILL_DIR}/scripts/bootstrap.py   # Windows: python ... or py ...
```

Then read the interpreter path from `${CLAUDE_SKILL_DIR}/.venv/engine-python.txt`
and use it as `<PYBIN>` in the command above.

This creates the venv and installs deps (local CPU embedder, no API key). If the
user wants a hosted embedder, they set `CREATIVITY_EMBEDDER=api` plus the
provider env vars before launching; you do not need a key otherwise.

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

Write the resolved axes to a temp file `axes.json` in this shape:

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
ENGINE init-project --project PROJECT --axes axes.json --seed 7
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

Write survivors to `candidates.json` (a JSON list, or `{"candidates": [...]}`).

---

## 4. Ingest (engine owns diversity)

```
ENGINE ingest --project PROJECT --candidates candidates.json --axes axes.json --seed 7
```

Returns:

```json
{
  "slate": [ {"id","text","descriptor","coords","niche_id","novelty","fitness","embedding_ref"}, ... ],
  "ask_pairs": [ ["idA","idB","why this pair is worth asking"], ... ],
  "monitor": {"collapsing": false, "mean_cosine": 0.18, "entropy": 2.1,
              "normalized_entropy": 0.88, "coverage": 9, "n": 12, "reasons": []},
  "parents": ["id", "..."]
}
```

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

Keep it to the returned pairs. Offer that the user can **pin** any idea as a
stepping stone to keep exploring from.

---

## 6. Remember + parents + loop

For each user answer, record it:

```
# a pairwise preference
echo '{"type":"comparison","winner":"idA","loser":"idB","context":"..."}' > event.json
ENGINE remember --project PROJECT --event event.json

# a pin
echo '{"type":"pin","id":"idA"}' > event.json
ENGINE remember --project PROJECT --event event.json
```

Then fetch diverse parents for the next generation (pins are always honored):

```
ENGINE parents --project PROJECT --k 4 --seed 7
```

Loop back to step 2 with those parents, or stop when the user is satisfied. At any
time, `ENGINE metrics --project PROJECT` reports `{entropy, mean_cosine,
coverage, n}` for the archive.

---

## 7. React to collapse

If `monitor.collapsing` is true (or `reasons` is non-empty), the search is
converging. Next round, raise diversity pressure:
- switch to operators you haven't used (analogy/biomimicry/constraint-randomizer);
- explicitly forbid the crowded niches ("no more X-mechanism ideas");
- demand each new idea be far from the recent set;
- widen an axis (e.g. push edginess to its extremes).

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
