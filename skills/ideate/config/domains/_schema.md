# Domain config schema

A domain config is a small YAML file that pins down the **descriptor axes** of an
idea space. Diversity is only meaningful relative to a set of axes; this file is
how a session declares them. Axes are resolved per session via a cascade:

1. **named** — the user names a domain that has a shipped config in
   `config/domains/examples/` → load it;
2. **inferred** — the agent infers 4–6 axes from the brief (see
   `references/axis_inference.md`) and confirms them with the user, writing an
   `axes.json` in the same shape as below;
3. **generic** — neither applies → fall back to `config/domains/generic.yaml`.

All three paths produce the same internal `AxesSpec`. The engine never assumes a
domain — the resolved axes are always passed in explicitly.

## Shape

```yaml
domain: <string>                 # human label for the namespace, e.g. "marketing"
unit_of_generation: <string>     # idea | concept | hypothesis | name | feature | ...
axes:                            # 4-6 axes recommended
  - name: <string>               # unique within the spec
    type: <categorical | continuous | open>
    range: [<lo>, <hi>]          # REQUIRED for continuous; ignored otherwise
    primary_novelty: <bool>      # at most ONE axis; the main novelty carrier
    bins: <int>                  # optional, continuous only (default 5)
judge_rubric: references/judge_rubric.md   # which rubric the agent prefilters with
slate_size: <int>                # how many ideas to surface per cycle (default 6)
candidates_per_generation: <int> # how many the agent drafts per cycle (default 12)
engine:                          # OPTIONAL engine tuning overrides (see below)
  open_niches: <int>
  quality_weight: <float>
```

## Engine tuning (`engine:`)

Every engine knob has a built-in default, so this block is **optional** — omitting
it reproduces the default behavior. Override only what a domain needs (e.g. a
domain with many distinct mechanisms might widen `open_niches`). The defaults are
calibrated for the bundled embedders; change them when an embedder or domain shifts
the natural scale of similarity.

| key | default | meaning |
| :-- | --: | :-- |
| `open_niches` | 24 | frozen Voronoi cells for the open (mechanism) axis |
| `open_niche_freeze_factor` | 2 | freeze the open-axis partition once `factor × open_niches` (= 48) mechanisms accumulate |
| `knn_k` | 5 | neighbours for geometric novelty |
| `dedup_tau` | `null` | near-duplicate cosine threshold; `null` ⇒ per-embedder default |
| `novelty_ref_cap` | 500 | cap on the dedup/novelty reference (most-novel elites) |
| `max_dpp_pool` | 200 | cap on the elite pool fed to the DPP |
| `quality_weight` | 0.3 | weight of the (bounded, [0.7–1.3]-clipped) judge fitness in the DPP slate; 0 ⇒ pure diversity |
| `monitor_cos_threshold` | 0.55 | absolute similarity fallback (until the baseline window fills) |
| `monitor_entropy_threshold` | 0.50 | concentration-collapse threshold (≥3 occupied niches) |
| `monitor_margin` | 0.15 | relative similarity flag: `baseline + margin` |
| `monitor_cos_ceiling` | 0.80 | absolute safety ceiling for the similarity flag |
| `monitor_window` | 5 | rolling-baseline size (recent generations' mean cosine) |
| `monitor_min_baseline` | 2 | samples needed before the relative rule applies |
| `under_generation_ratio` | 0.6 | prefilter guard: flag `under_generation` below this fraction of the per-gen target |
| `state_prune_threshold` | 2000 | candidate-store size above which unreferenced records/embeddings are pruned (0 disables) |
| `ask_sim_weight` | 0.5 | active-learning pair score: embedding-similarity weight (≤ 0 ⇒ compare region-separating pairs / explore) |
| `ask_uncertainty_weight` | 0.3 | active-learning pair score: fitness-uncertainty weight |
| `ask_novelty_weight` | 0.2 | active-learning pair score: novelty weight |
| `explore_until_generation` | 0 | ask-policy schedule: first N generations ask region-separating (explore) pairs, then refine; `0` disables (flat `ask_sim_weight`) |
| `erosion_window` | 5 | variety-erosion sensor (advisory): generations of survivor novelty used to estimate the decay slope |
| `erosion_accel_ratio` | 0.5 | erosion fires when recent decay slope ≥ (1+ρ)× the earlier slope (i.e. ≥ 1.5×) |
| `erosion_persist` | 2 | consecutive accelerating generations before flagging `variety_eroding` |
| `gap_probe` | `false` | advisory surface/mechanism gap measurement: when true, `ingest` emits a `surface_mechanism_gap` record on the slate and appends it to a bounded `meta["gap_log"]` (also surfaced by `metrics`). Never affects selection or any gate |

## Axis types

| type | meaning | niching |
| :-- | :-- | :-- |
| `categorical` | a discrete label (audience, register, format…) | one bucket per distinct value |
| `continuous` | a number in `range` (edginess, boldness…) | `range` split into `bins` cells |
| `open` | a free-text "how" / mechanism / approach | data-adaptive Voronoi cells over embeddings (fit-once-then-freeze k-means; deterministic cold start) |

Mark exactly one axis — usually an `open` one — with `primary_novelty: true`. It
is the main carrier of novelty and is niched geometrically over embeddings, so
the engine can keep one elite per *kind of approach* rather than per surface
label.

## Validation

`load_axes` rejects, with a clear message and non-zero exit:

* an empty or missing `axes` list,
* an axis missing `name` or with a `type` outside the three allowed,
* a `continuous` axis without a valid `[lo, hi]` range (or `hi <= lo`),
* duplicate axis names,
* more than one `primary_novelty: true`,
* an out-of-range `engine:` override (e.g. a non-positive count, a `quality_weight`
  outside `[0, 1]`, or a `dedup_tau` outside `(0, 1]`).
