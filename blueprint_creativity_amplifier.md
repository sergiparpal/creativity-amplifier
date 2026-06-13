# Blueprint: human-in-the-loop creativity amplifier

*Design document for systematizing idea generation in software (2026 state of the art). It consolidates the problem diagnosis, the proposed architecture, the division of labor between machine and human, the build order, the instrumentation, and the honest limits.*

---

## 0. Design thesis

You don't build a creative machine: you build a **creativity amplifier** that turns scarce human judgment into broad, diverse, well-explored search, and that is honest about what it cannot automate.

- The **machine** owns breadth: massive but informed variation, cheap pre-filtering, maintenance of a diverse archive, and the verifiable part of evaluation.
- The **human** supplies the grounded signal no automatic layer provides: value in subjective domains, the "this is interesting" judgment at bifurcation points, re-anchoring against Goodhart, the choice of diversity axes, and re-valuation when the space transforms.
- Almost all the engineering in between consists of **spending human attention optimally** and **preventing collapse to the mean**.
- The same skeleton serves both domain regimes: the human dial turns up when the value oracle is weak and down when an automatic verifier is available.

---

## 1. The engine: blind variation + selective retention

Every new idea arises from a two-stroke engine running in a loop: **generate variants → select the valuable ones → retain → feed back**. Conceptual basis: blind variation and selective retention (Campbell; Simonton) and the Geneplore model (Finke, Ward, Smith).

Four properties that shape the design:

1. **Meaning is assigned at evaluation**, not before generation. Sense is constructed retrospectively.
2. **Variation is blind, not random**: it is biased by prior knowledge. "Blind" only means that you don't know in advance which variant will succeed.
3. **Equal-odds rule** (Simonton): hits are proportional to the total number of attempts. There is no shortcut to identifying the good idea in advance; you must generate volume and let selection filter.
4. It is a **loop, not a sequence**: generation is biased by the sense already accumulated, and evaluation produces the new sense.

---

## 2. The two bottlenecks

- **Evaluation (the fitness function).** Recognizing which variants are worth it. This is the real bottleneck.
- **Diversity / novelty.** Preventing the search from collapsing toward a single conventional optimum.

Generating variants is cheap, especially with LLMs as operators. Design effort concentrates on these two points.

---

## 3. The split by domain regime

The strategy depends entirely on the quality of the available value oracle.

- **Regime A — verifiable, cheap, hard-to-hack oracle.** Mathematics, code, molecules, circuits, robots in simulation, game strategies. Evaluation is solved; the system is near-autonomous. (Demonstrated: FunSearch beat the best known cap set, 512 vs 496; Eureka outperformed human experts on 83% of tasks; Cully's hexapod generated ~13,000 gaits.)
- **Regime B — subjective, cultural, or contextual value.** Art, writing, business strategy, product ideas, research direction. No reliable oracle exists; the human is indispensable.

**Rule of thumb:** classify your domain first by oracle quality. That decides everything else.

---

## 4. Limitations to mitigate (catalog)

| Limitation | Regime A (oracle) | Regime B (subjective) |
|---|---|---|
| Value evaluation | Solved (verifier) | **Open** (central problem) |
| Convergence to the mean | Mitigated (QD/islands/DPP) | Partial (judge biases toward the typical) |
| Diversity / mode collapse | Mitigated (near-solved) | Mitigated (manual axes) |
| Transformational ceiling | Partial (meta-search) | **Open** (co-evolve E with R) |
| Reward hacking / Goodhart | Mitigated (robust oracle) | **Open** (every proxy degrades) |

New limitations that the solutions themselves surface:

- **Descriptor design:** QD shifts the human work from "judging outputs" to "choosing the diversity axes," which determine the result.
- **Evaluator regress:** co-evolving E with R opens an infinite regress ("who evaluates the new evaluator?").
- **Degradation of the model of interestingness:** using a foundation model as a proxy for cultural criteria is itself subject to Goodhart.
- **Judge monoculture:** ensembling judges only helps if they are of different lineages; if they share a base model, self-preference is systemic.
- **Stepping-stone destruction:** setting the desired output as an explicit objective destroys the path toward it (evidence: Picbreeder). Tightening evaluation can worsen openness.

---

## 5. The architecture

```
                  ┌──────────────────────────────┐
   human   ──────▶│  Framing and seeds            │  problem + diversity axes
      │           └───────────────┬──────────────┘
      │                           ▼
      │           ┌──────────────────────────────┐
      │           │  Operator bank                │  composable and rewritable
      │           └───────────────┬──────────────┘
      │                           ▼
      │           ┌──────────────────────────────┐
      │           │  Candidates                   │  blind but informed variation
      │           └───────────────┬──────────────┘
      │                           ▼
      │           ┌──────────────────────────────┐
   human   ──────▶│  Layered evaluation           │  verifiable · proxy · human
                  └───────────────┬──────────────┘
                                  ▼
                  ┌──────────────────────────────┐     ┌─────────────────────┐
                  │  Diverse archive              │◀────│ anti-collapse monitor│
                  │  MAP-Elites + DPP             │     │ entropy, diversity   │
                  └───────────────┬──────────────┘     └─────────────────────┘
                                  │
       feedback   ◀───────────────┘   (re-seeds the operators)

  meta-level (outer loop): rewrites the operators (T) and co-evolves the evaluator (E with R)
```

### 5.1 Framing and seeds
The system's input and the **first point of maximum human leverage**. The human poses the problem and, above all, defines the **diversity axes** (the behavior space the archive will illuminate). A single decision configures the entire search: cheap in attention, enormous in effect.
*Mitigates:* descriptor design (handed to whoever has the criteria) and problem reframing.

### 5.2 Operator bank (variation)
Composable operators, each with its own contract, driven by an LLM so that variation is **blind but informed** rather than noise:
- **Mutation** — local nudge; perturb an element to divert its trajectory.
- **Analogy / combination** — import structure from a distant domain (Gentner's structure mapping; Fauconnier-Turner conceptual blending; Koestler's bisociation).
- **Transformation** — alter the rules of the space (Boden's transformational creativity).
- **Reframing** — inversion, abstraction ladder, first principles.
- **Systematic** — SCAMPER, morphological analysis (Zwicky), TRIZ (Altshuller; resolving contradictions).

The operators are **rewritable** by the meta-level (see 5.7).

### 5.3 Candidates
Pool of unevaluated variants. The part the software already does better than the human (scale, speed, coverage). No bottleneck here.

### 5.4 Layered evaluation (dual regime)
The heart of the system. It branches by domain.

- **Regime A:** the automatic verifier *is* the fitness. Evaluation solved.
- **Regime B:** a three-layer funnel.
  1. **Cheap automatic filters** — validity, deduplication, constraints. Kill the obvious.
  2. **Learned preference proxy** — calibrated from human comparisons; scores the bulk to provide breadth.
  3. **Human via active learning** — queried only where the proxy is uncertain or candidates disagree, and at bifurcation points.

The human's roles in this layer: supply grounded value, re-inject the novelty premium that the low-perplexity proxy penalizes (anti-convergence), and re-anchor the proxy against Goodhart drift.
*Mitigates:* subjective value evaluation, convergence to the mean, and reward hacking.

### 5.5 Diverse archive (retention + diversity)
A MAP-Elites archive indexed by the axes chosen in 5.1, plus a **DPP** repulsion term in embedding space. It keeps **one elite per niche**, not a single best: this is the structural defense against mode collapse. Useful variants: CVT-MAP-Elites for high-dimensional descriptors, CMA-ME, Novelty Search with Local Competition.
*Mitigates:* diversity / mode collapse (near-solved in engineering) and convergence to the mean (structural).

### 5.6 Feedback
The winners seed the next round of variation, biased toward fertile but diverse regions. Because the archive rewards novelty and the human selects by "this is interesting" at the bifurcation points (Picbreeder mode), the system **preserves the stepping stones** that purely objective search would destroy.
*Mitigates:* stepping-stone destruction (the deepest problem).

### 5.7 Meta-level (outer loop)
Periodically the system rewrites itself:
- **Evolves its own operators** Promptbreeder-style (self-referential: it also mutates the mutation prompts).
- **Co-evolves the evaluator E along with the rules R** that define the space (the real difficulty: a transformed space needs a transformed evaluator).

The human is the anchor: when the space transforms, the human supplies the new sense of value, which resolves the "who evaluates the new evaluator?" regress as a grounding anchor.
*Mitigates (partially):* the transformational ceiling. Full transformational autonomy remains undemonstrated.

---

## 6. The human's role (governing principle)

Goal: **minimize queries to the human while maximizing their information value.** Three levers:

1. **Active learning** — query where the proxy is most uncertain or where candidates disagree most.
2. **Amplification** — the human trains and re-anchors a proxy that handles the easy 95%; the human reserves themselves for the ambiguous 5% and for drift detection.
3. **High-leverage placement** — place the human at the framing, the axes, the bifurcations, the paradigm shifts, and the Goodhart re-anchoring, not at every candidate.

Traps to avoid: human fatigue and inconsistency (the classic failure mode of interactive evolution) and the fact that the human **also** drifts toward the familiar (mere-exposure effect), so it pays to show diverse slates and reward exploration.

---

## 7. Build order

From lowest to highest risk, each piece as a node with its own contract for incremental deployment:

1. **Full Regime A** — LLM operators + island or MAP-Elites archive + automatic verifier. Works today and yields verifiable results. Progress metric: does it beat the best known result on a closed benchmark?
2. **Subjective layer** — cheap filters → preference proxy → human via active learning, with the anti-collapse monitor active.
3. **Meta-level** — start cheap with Promptbreeder (rewriting operators) before getting into evaluator co-evolution (POET / AI-GAs).

---

## 8. Instrumentation and metrics

- **Anti-collapse:** measure output entropy and the mean cosine similarity of the archive across generations. If entropy falls and similarity rises, there is mode collapse → raise diversity pressure or trigger more human queries.
- **Goodhart / over-optimization:** keep a "gold" set and watch the proxy-vs-gold curve; stop when performance against the gold begins to drop (Gao et al. over-optimization scaling laws).
- **Quality-diversity:** archive coverage and QD-score.
- **Judge-human agreement:** audit blind; if it falls below the acceptable threshold, reduce the proxy's weight and increase human sampling.
- **Creativity audit:** use external rubrics (Boden's novelty-value-surprise criteria, SPECS, FACE) as an auditing instrument, **not** as an optimizable fitness, so the system doesn't game them.

---

## 9. What it solves, what it mitigates, what stays open

- **Solved (conditional on Regime A):** evaluation, via the verifier, and with it the deployable creative loop. Caveat: it didn't solve evaluation, it sidestepped it by restricting to domains where evaluation is free.
- **Mitigated:** convergence to the mean (decoupling novelty from quality), diversity / mode collapse (near-solved), transformational ceiling (partial, via meta-search), Goodhart (bounded by human re-anchoring).
- **Open:** formalizing subjective/cultural value, fully autonomous transformational creativity, the human bandwidth ceiling, and the human's own biases.

---

## 10. Risks and honest limits

- The architecture **does not solve** the underlying problem; it **relocates** it to a tractable terrain (the optimal allocation of human attention), where tools do exist (active learning, preference learning, interactive evolution).
- **Bandwidth ceiling:** the human is slow and expensive; amplifying them with a proxy reintroduces their biases, so the proxy must be re-anchored relentlessly.
- **Computational cost** is high in the open-endedness pieces (POET, AI-GAs).
- **Autonomous transformation** is not achieved with the current state of the art.
- **The deep point:** human value may not be a separable algorithm waiting to be discovered, but a feature of a form of life — of being the kind of thing that can be bored, can fail, and has something at stake. If so, the irreducible core stays with the human; what this architecture achieves is using that core with surgical precision rather than blindly.

---

## Conceptual anchors and reference systems

- **Engine and theory:** blind variation and selective retention (Campbell, Simonton); Geneplore (Finke, Ward, Smith); Boden's three types of creativity (combinational, exploratory, transformational); Wiggins's R/T/E formalization and meta-level.
- **Operations:** structure mapping (Gentner); conceptual blending (Fauconnier, Turner); bisociation (Koestler); SCAMPER; morphological analysis (Zwicky); TRIZ (Altshuller).
- **Anti-convergence and diversity:** Novelty Search (Lehman, Stanley); MAP-Elites and Quality-Diversity (Mouret, Clune; Cully et al.); DPP (determinantal point processes).
- **LLM as operator in evolutionary loops:** FunSearch and AlphaEvolve (DeepMind); Evolution through Large Models (Lehman et al.); Eureka (NVIDIA et al.); QDAIF; Promptbreeder.
- **Open-endedness and meta-level:** POET and Enhanced POET; OMNI and OMNI-EPIC; AI-GAs (Clune).
- **Evaluation and its pathologies:** LLM-as-a-judge (position, verbosity, and low-perplexity self-preference biases); reward hacking and Goodhart's law (Gao, Schulman, Hilton); creativity rubrics (Boden; Lovelace Test, Bringsjord et al.; FACE; SPECS, Jordanous).
- **Human in the loop:** interactive evolution (Dawkins's Biomorphs; Picbreeder, Secretan, Stanley et al.).
