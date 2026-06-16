# Improvement Plan

A consolidated, prioritized plan for `creativity-amplifier`, derived from a full
review of the engine (`skills/ideate/scripts/creativity_engine/`), the skill
(`SKILL.md` + `references/`), and the packaging (`hooks/`, `bootstrap.py`,
`requirements.txt`).

The codebase is in good shape: the "geometry owns novelty, the judge owns only
validity" invariant is coherent end-to-end, the bootstrap is idempotent and
concurrency-safe, and the self-test as a correctness contract is strong. The
items below are improvements on a solid base, not patches.

Each claim was verified against the code; file/line references are given so the
work is actionable.

---

## Execution order (TL;DR)

1. **CI** — the safety net that protects everything else while it changes.
2. **Lightweight + multilingual embedder** — fixes language *and* install weight
   in one change; makes the "instant hash fallback" idea unnecessary.
3. **Operational hardening** — prefilter guard, visible provisioning failures,
   temp-file hygiene.
4. **Engine correctness / cleanup** — open-axis freeze threshold, `api` stub
   honesty, unbounded state growth.
5. **Conceptual notes to document** (not necessarily change) + **minors**.

---

## 1. Continuous Integration (do first)

**Status:** ✅ Done — `.github/workflows/ci.yml` (9-cell test matrix + best-effort
plugin-metadata validation).

**Why:** the project's headline promise is "works the same on Windows/WSL/macOS/
Linux," and it has a real correctness contract (`selftest`) plus a hermetic test
suite — but there is no `.github/` at all. CI is cheap and protects the matrix
the hooks work hard to cover, especially while the embedder is swapped (item 2).

**What:** add `.github/workflows/ci.yml`.

- Matrix: OS `[ubuntu-latest, macos-latest, windows-latest]` × Python
  `[3.11, 3.12, 3.13]`.
- Steps: checkout → setup-python → `pip install -r skills/ideate/scripts/requirements.txt`
  and `pip install -e skills/ideate/scripts --no-deps` → `pytest -q` →
  `python -m creativity_engine selftest`.
- The suite is hermetic: `tests/conftest.py` forces `CREATIVITY_EMBEDDER=hash`
  and an isolated `CREATIVITY_AMPLIFIER_HOME`, so **CI never downloads a model**.
- Best-effort job: validate `plugin.json` / `marketplace.json` are well-formed
  JSON; run `claude plugin validate --strict .` if the CLI is installable in CI,
  otherwise skip cleanly.

**Acceptance:** green on all 9 matrix cells; `selftest` exits 0; no network model
download in any job.

---

## 2. Lightweight + multilingual embedder (biggest user-facing win)

**Status:** ✅ Done — `static` (`model2vec` / `potion-multilingual-128M`, 256-dim,
101 langs) is the new default in `embed.py`; `local` (bge) is opt-in via
`requirements-local.txt`; dedup τ calibrated to `0.93`. Verified in a clean venv:
no torch, real EN+ES inference is numpy-only, a Spanish brief dedups/niches sanely
(9→8, 8 niches, no false collapse), and `selftest` stays green on `hash`. Docs +
migration caveat updated across README/CLAUDE/PAPER/loop.

### The two problems it solves at once

1. **Monolingual geometry.** The default `local` embedder is
   `BAAI/bge-small-en-v1.5` (`embed.py:28`), which is **English-only** (384-dim).
   Every guarantee of the tool rests on the embedder judging "what's new" well;
   on a Spanish (or any non-English) brief the geometry degrades — dedup,
   niching, and k-NN novelty stop correlating with human-perceived novelty. For
   a tool sold as domain-/language-agnostic this is a **correctness** issue.
2. **Install weight contradicts the "local, lightweight, server-less"
   positioning.** `sentence-transformers` (`requirements.txt:4`) pulls in
   PyTorch (~2 GB), which is what causes the multi-minute first-run download and
   what can fail on constrained machines (the README acknowledges the wait,
   `README.md:49-51`).

### The fix

Add a `StaticEmbedder` backed by **model2vec**, and make
**`minishlab/potion-multilingual-128M`** the new default. Keep `bge`
(`LocalEmbedder`) as an **opt-in high-fidelity** option.

Verified specs (HuggingFace model card + repo API):

- ~100M params, **256-dim**, **101 languages** (Spanish included), distilled
  from `BAAI/bge-m3`, **MIT** license.
- **Static** embeddings → **inference needs only numpy, no torch**.
- **Different model family from Claude** → preserves the lineage hedge.
- **Size — read this carefully:** the weights are **~120 MB** (`model.safetensors`,
  F32); the full HF repo is **~1 GB** because it also ships an ONNX copy. The
  Python path (`StaticModel.from_pretrained`) downloads safetensors + tokenizer,
  so the real footprint is **~120–140 MB**, not the full repo.
  **Do NOT write "~30 MB" anywhere** — that is the `potion-base-8M` figure, not
  this model. Confirm the exact number on the HF *Files* page before quoting it
  in the README.

Why the static-embedding fidelity trade-off is acceptable *here*: static models
lose most on retrieval/STS and least on classification/clustering; this pipeline
(niching, dedup, k-NN novelty, DPP) is essentially **similarity clustering**, so
the loss lands on the tasks we care about least. (Minor caveat: k-NN novelty and
DPP also rely on mid/far-range geometry, where static models are a bit noisier —
keep `bge` as the high-fidelity escape hatch.)

### Implementation notes

- New `StaticEmbedder(Embedder)` in `embed.py`: `_embed_raw` just returns
  `model.encode(texts)`. The base `Embedder.embed()` already L2-normalizes after
  `_embed_raw` (`embed.py:71-75`), so the unit-row contract is satisfied for
  free.
- Add a provider key (e.g. `static`) and set `DEFAULT_PROVIDER = "static"`. Keep
  `local` (= bge) selectable via `CREATIVITY_EMBEDDER=local`.
- `requirements.txt`: replace `sentence-transformers` with `model2vec`; move
  `sentence-transformers` to an **optional** extra installed only when the user
  opts into `local`. The default install then carries **no torch**.
- Add a per-embedder dedup τ for the new family in `DEDUP_TAU_BY_EMBEDDER`
  (`embed.py:37`). The static family's cosine scale differs from bge's, so the
  current `local: 0.94` must not be reused blindly — **calibrate τ** by measuring
  the near-duplicate cosine distribution on a sample, the way the existing values
  were chosen. (The monitor's similarity flag self-calibrates via its rolling
  baseline, so only the fixed dedup τ needs hand-setting.)

### Migration caveat (must document)

Switching the default from 384-dim (bge) to 256-dim (potion) is **breaking for
existing persisted projects**: the next `ingest` on an old project trips
`_guard_embedding_dim` (`pipeline.py:289`) with a (correct) hard error. Nothing
silently corrupts, but the README/CHANGELOG must state that the new default
applies cleanly only to **new** projects; existing ones must be re-embedded or
pinned to `CREATIVITY_EMBEDDER=local`.

### This obviates the "instant hash fallback" idea

An earlier proposal was to start round 1 in `hash` while `local` downloads. That
clashes with the engine: `hash` is 512-dim (`embed.py:29`) vs bge's 384-dim, so
`_guard_embedding_dim` would reject the switch, and per-family cosine scales
differ (hence per-embedder τ) — it is not a transparent upgrade. With a ~120 MB
torch-free model the first-run wait largely disappears, so the workaround is no
longer needed. **Dropped.**

**Acceptance:** default install pulls no torch; a Spanish brief yields sane
dedup/niching/novelty; `selftest` stays green (still on `hash`); README updated
with the real footprint and the migration note.

---

## 3. Operational hardening

**Status:** ✅ Done — all three. 3a: `ingest` emits a soft
`monitor.under_generation` flag (`submitted`/`target_candidates`) when fewer than
`engine.under_generation_ratio` (0.6) of the target reach it, advisory only (never
touches `collapsing`/calibration); SKILL/loop tell the agent to prefilter less. 3b:
the engine-not-ready branch now tails `provision.log` before/after the foreground
bootstrap. 3c: a per-project `tmp/` scratch dir (new `paths` command,
`State.ensure`) holds the skill's hand-off files inside the state home, never the
cwd.

### 3a. Prefilter guard (mechanical check on the load-bearing invariant)

**Why:** "the judge never prunes variety" is enforced only by prose in
`SKILL.md` / `loop.md`; it depends on a future Claude obeying it. The monitor
already covers the *generation* stage — it runs on the **raw, pre-dedup**
vectors (`pipeline.py:515`), so "the agent generated samey ideas" is caught by
`too_similar`. The blind spot is the **prefilter** stage: the agent dropping
candidates as "off-brief" and cutting variety under cover of validity — the one
stage the engine never sees.

**What:** in `pipeline.ingest`, compare the number of candidates **submitted to
`ingest`** (`len(cand_list)`, pre-dedup) against `candidates_per_generation`
(from `meta.json`). If it is well below target, surface a soft flag in the
`monitor` dict (e.g. `under_generation` / `possible_overprefilter`). Put the
sensor at the **submitted-vs-target** boundary — **not** post-dedup survivors,
since dedup is the engine's own job and must not count against the agent. Have
`SKILL.md` / `loop.md` instruct the agent to react (generate more / prefilter
less) when the flag is set.

### 3b. Make background provisioning failures visible

**Why:** `bootstrap.py` writes to `provision.log` (`<venv>.parent/provision.log`,
`bootstrap.py:318`), but nothing surfaces it. If the background build fails
(e.g. a wheel won't compile) the user only finds out when `/ideate` falls to the
foreground path, with no diagnostic.

**What:** in the skill's "engine not ready yet" branch (`SKILL.md:33-42`), read
and show the **tail of `provision.log`** before/after the foreground bootstrap,
instead of re-running blind.

### 3c. Temp-file hygiene in the skill

**Why:** `loop.md` tells the agent to write `axes.json`, `candidates.json`,
`event.json` as "temp files" without specifying where (`loop.md:76,138,189`).
They default to the user's cwd — clutter for a repo, and a collision risk across
concurrent sessions.

**What:** write them under a per-session temp dir inside the state home
(`~/.creativity-amplifier/<project>/tmp/`), and document the location in
`loop.md`.

---

## 4. Engine correctness / cleanup

**Status:** ✅ Done — all three. 4a: lowered `open_niche_freeze_factor` 4→2
(threshold 96→48, ~4–5 generations) so the data-adaptive partition actually
activates; `ingest`/`metrics` now expose an `open_axis` progress block; cold-start
validated good under the real embedder (8/12 niches), and the benign sklearn
"fewer distinct clusters than k" warning is silenced at the fit. 4b: `api` is
honestly documented as a stub/extension point everywhere (README/CLAUDE/loop/
embed.py) — chose the downgrade path over wiring a backend. 4c: `ingest` prunes
candidate records/embeddings nothing reads again once the store exceeds
`engine.state_prune_threshold` (default 2000, 0 disables), keeping exactly elites +
pins + comparison ids, so it's output-neutral.

### 4a. The open-axis freeze threshold is almost never reached

**Why:** the data-adaptive open-axis partition (the "fit-once-then-freeze"
k-means over mechanism embeddings — a headline feature) only freezes after
`OPEN_NICHE_FREEZE_FACTOR * OPEN_NICHES = 4 * 24 = 96` **survivor** mechanism
embeddings accumulate (`pipeline.py:45-46`; `accum` grows only by survivors per
cycle). With `candidates_per_generation: 12` and dedup, that is ~8–10 full
generations — more than most real brainstorming sessions. **In practice almost
every session runs entirely on the deterministic cold-start partition**, and the
flagship feature rarely activates.

**What:** either lower `freeze_factor`, make the threshold proportional to
session size, or — at minimum — explicitly validate that the cold-start
partition is good enough on its own, since that is what runs ~95% of the time.
Add a `metrics`/log line exposing accumulation progress toward the freeze so this
is observable.

### 4b. `api` embedder: documented but a stub

**Why:** `APIEmbedder._embed_raw` raises `NotImplementedError` (`embed.py:149-153`),
yet the README lists `CREATIVITY_EMBEDDER=api` as a real configuration option
(`README.md:131-139`). A user who follows the docs hits a crash.

**What (orthogonal to item 2 — a doc decision you make anyway):** either wire a
real backend (Voyage/OpenAI is ~20 lines), or downgrade the README to call it an
**extension point**, not a supported option. The error message already points
the way.

### 4c. Unbounded state growth

**Why:** `_persist_cycle` rewrites `stored_emb` and `cand_store` **whole** every
cycle (`pipeline.py:404+`); only the novelty *reference* is capped
(`NOVELTY_REF_CAP=500`), not the persisted dicts. Fine for short sessions, O(n)
rewrite cost for long ones.

**What (lower priority):** prune non-elite, non-pinned candidates + embeddings
for long sessions.

---

## 5. Conceptual notes — document, don't necessarily change

**Status:** ✅ Done. 5a: documented the precise division of labor in `CLAUDE.md`
(the invariant section) and `docs/PAPER.md` (§4 niching) — niche *placement* on
**every** axis runs on agent descriptors; the purity guarantee is exactly k-NN
novelty + dedup + DPP on the lineage-distinct idea-text embedding; kept the
descriptor-based open-axis niching deliberately (documented why). 5b: exposed the
`select_ask_pairs` weights as config knobs (`ask_sim_weight`/`…uncertainty`/
`…novelty`, defaults reproduce current behavior; `ask_sim_weight ≤ 0` flips
learn-preference → explore) and documented the unresolved policy tension in
`CLAUDE.md`/`docs/PAPER.md`.

### 5a. Open-axis placement reflects agent-authored descriptors

The open-axis niche cell is computed from the embedding of the agent's
descriptor text (`_open_axis_texts`, `pipeline.py:155`). Because there is one
elite per niche and the DPP pool is exactly those elites, the agent's word choice
indirectly influences which ideas become elite, and thus the slate pool.

Scope it correctly so the docs are precise:

- This is **not** special to the open axis — niche placement on **every** axis is
  driven by agent-authored descriptors by design (a categorical mislabel changes
  the niche and the elite competition just the same).
- The purity guarantee the design actually makes — and keeps — is over **k-NN
  novelty, dedup, and the DPP kernel**, all computed on the **idea-text**
  embedding.
- Even the open-axis placement runs the agent's words through the **independent
  embedder**, so what "leaks" is the agent's *word choice*, not Claude's
  geometry.

Action: state this division of labor explicitly in `CLAUDE.md` / `docs/PAPER.md`.
Optionally, for stricter purity, niche the open axis on the full idea-text
embedding instead of the descriptor text.

### 5b. `ask_pairs` weights similarity highest

`select_ask_pairs` uses `W_SIM = 0.5` as the top weight (`memory.py:26`) — it
asks the user to compare *similar* ideas. The decomposition: if `ask_pairs`
exists to **learn the preference function** (rank within niches / pick parents),
similar pairs are legitimate (max judge-disagreement at the decision boundary);
if it exists to **explore**, comparing near-identical ideas is the least useful
question. The policy is also **not tested by the value-gate**, so there is no
evidence it improves the downstream slate.

Action: measure it — do similar pairs vs. region-separating pairs produce better
parents next generation? — and/or expose the weighting as a config knob.

---

## 6. Minors

- **Dual hook noise:** `hooks.json` launches both `sh` and `powershell` on every
  `SessionStart`; on the "wrong" OS one always fails (harmless under `async`, but
  noisy in hook logs).
- **No `CHANGELOG`**, and `version` is pinned at `0.1.0` (`plugin.json:3`) — worth
  adding for a plugin that updates via the marketplace.
- **License GPL-3.0** (`plugin.json:9`): strong copyleft can deter adoption/forks
  vs. MIT/Apache for a plugin. Not a bug — a decision worth making consciously.
