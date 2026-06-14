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
```

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
* more than one `primary_novelty: true`.
