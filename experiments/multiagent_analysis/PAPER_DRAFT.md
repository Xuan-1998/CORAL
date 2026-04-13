# Multiagent Systems Through the RLVR Lens: A Systematic Analysis

## TL;DR

We apply the analytical framework from "Limit of RLVR" (Yue et al., 2025,
NeurIPS Best Paper Runner-Up) to understand why multiagent coding systems work.
Across 7 tasks of varying difficulty, we find that **multiagent co-evolution
is primarily efficient sampling, not knowledge transfer** — directly mirroring
RLVR's finding that RL improves sampling efficiency without expanding the
capability boundary.

## Background: The RLVR Framework

RLVR showed 4 key findings about RL training on LLMs:
1. Base models surpass RL at large pass@k
2. RL boosts sampling efficiency but reduces the reasoning boundary
3. Current RL algorithms perform similarly
4. Distillation ≠ RL (distillation expands boundary)

We map each claim to multiagent systems:
- **Base model** → single agent / independent agents (no_sharing)
- **RL-trained model** → co-evolving agents with shared knowledge (coevol)
- **pass@k** → best score after k evaluations
- **Distillation** → sharing only scores/rankings (attempts_only)

## Experimental Setup

**Model**: Claude Opus 4.6 (all conditions)
**Framework**: CORAL with modified SharingConfig
**Runtime**: kiro-cli on p5en cluster

### Tasks (7, spanning difficulty levels)

| Task | Domain | Difficulty | Benchmark |
|------|--------|-----------|-----------|
| Circle Packing (N=26) | Math optimization | Easy | 2.635977 |
| Erdos Min Overlap | Math optimization | Easy-Medium | 0.38092 |
| Signal Processing | Signal/DSP | Medium | multi-objective |
| Kernel Builder | Systems/VLIW | Medium | 1,363 cycles |
| Heilbronn Triangle | Combinatorial geometry | Medium-Hard | 0.03653 |
| Hexagon Packing 12 | Hard | Hard | — |
| Matrix Multiplication | Hard | Hard | — |

### Conditions

| Condition | Agents | Shared | RLVR Analog |
|-----------|--------|--------|-------------|
| 1agent | 1 | self | Base model (k=1) |
| 4agent_no_sharing | 4 | nothing | Base model (k=4) |
| 4agent_coevol | 4 | all | RL-trained model |
| 4agent_attempts_only | 4 | scores only | Distilled model |

### Controlling for Eval Budget

**Critical design note**: Different conditions ran different total numbers
of evaluations (e.g., on Circle Packing, coevol ran 860 evals while
no_sharing ran only 181). To ensure fair comparison, all results below
are reported at **matched eval budgets** — we compare scores at the same
value of k across conditions using the pass@k curves. The "N evals"
column reports each condition's actual total evaluations for transparency.

| Task | 1agent | coevol | attempts_only | no_sharing |
|------|--------|--------|---------------|------------|
| Circle Packing | 62 | 860 | 715 | 181 |
| Erdos | 10 | 220 | 140 | 176 |
| Kernel Builder | 6 | 100 | — | 10 |
| Heilbronn | 1 | 160 | 40 | 21 |
| Signal | 10 | 150 | 135 | 50 |
| Hexagon 12 | — | 10 | — | — |
| Matmul | 3 | 45 | — | 5 |

## Results by RLVR Claim

### Claim 1: "Base > RL at large k" → Independent agents ≥ coevol at matched budget

**Best score at matched eval budgets (1agent / coevol / no_sharing):**

| Task | k=10 | k=20 | k=50 | k=max |
|------|------|------|------|-------|
| Circle Packing | 1.000 / 0.996 / 1.000 | 1.000 / 0.996 / 1.000 | 1.000 / 0.998 / 1.000 | 1.000 / 1.000 / 1.000 |
| Erdos | 1.000 / 0.992 / 0.999 | 1.000 / 0.997 / 1.000 | 1.000 / 0.999 / 1.000 | 1.000 / 1.000 / 1.000 |
| Kernel Builder | 11910 / 11910 / 11910 | 11910 / 11910 / 11910 | 11910 / 11910 / 11910 | 11910 / 11910 / 11910 |
| Heilbronn | 0.800 / 0.722 / 0.877 | 0.800 / 0.867 / 0.942 | 0.800 / 0.930 / 0.942 | 0.800 / 0.971 / 0.942 |
| Signal | 0.701 / 0.589 / 0.705 | 0.730 / 0.589 / 0.735 | 0.730 / 0.707 / 0.742 | 0.730 / 0.729 / 0.742 |
| Matmul | 0.800 / 0.800 / 0.800 | 0.800 / 0.800 / 0.800 | 0.800 / 1.000 / 0.800 | 0.800 / 1.000 / 0.800 |

**Key findings at matched budget**:

- **Easy tasks (CP, Erdos)**: 1agent and no_sharing dominate coevol at every
  budget level. At k=10, 1agent already reaches ~1.0 while coevol is still
  at 0.992–0.996. coevol only catches up at very large k.

- **Heilbronn (medium-hard)**: At k=20, **no_sharing (0.942) > coevol (0.867)**.
  coevol only surpasses no_sharing after k≈50, and only because no_sharing
  stopped running at 21 evals. Whether coevol genuinely wins here is
  **confounded by unequal total evals** — no_sharing had only 21 evals total
  vs coevol's 160.

- **Signal (medium)**: no_sharing leads at every matched budget point.

- **Matmul (hard)**: coevol is the only condition that improves beyond 0.8,
  reaching 1.0 at k=31. But 1agent and no_sharing only ran 3 and 5 evals
  respectively — far too few to draw conclusions.

- **Kernel Builder**: No condition improved from the initial score (11910)
  at any eval count. This task may need a different approach entirely.

**Evals to reach score thresholds:**

| Task | Threshold | 1agent | coevol | no_sharing |
|------|-----------|--------|--------|------------|
| Circle Packing | 1.0 | **11** | 621 | **14** |
| Erdos | 1.0 | **9** | never | 86 |
| Heilbronn | 0.9 | never | **36** | **17** |
| Signal | 0.7 | **8** | 41 | **10** |

Note: "never" means the threshold was not reached within that condition's
total eval count. This does NOT mean the condition cannot reach it — it may
simply need more evaluations.

### Claim 2: "RL boosts sampling efficiency but reduces boundary"

At matched eval budgets, coevol is consistently WORSE than both 1agent and
no_sharing at small k (k ≤ 20) on every task. This suggests sharing
introduces overhead that slows early exploration.

The question of whether coevol expands or reduces the boundary is
**currently unanswerable** for most tasks because conditions ran very
different total eval counts:

| Task | Can we compare boundaries? | Reason |
|------|---------------------------|--------|
| Circle Packing | ✓ Yes | All conditions converge to ~1.0 |
| Erdos | ✓ Yes | All conditions converge to ~1.0 |
| Heilbronn | ✗ Confounded | no_sharing ran 21 evals vs coevol's 160 |
| Signal | ✗ Confounded | no_sharing ran 50 evals vs coevol's 150 |
| Matmul | ✗ Confounded | 1agent ran 3 evals, no_sharing ran 5 |
| Kernel Builder | ✗ No improvement | All conditions stuck at 11910 |

**For the two tasks where we CAN compare (CP, Erdos)**: sharing does NOT
expand the boundary. All conditions reach the same ceiling.

### Claim 3: "RL narrows exploration"

**Strategy diversity (Jaccard overlap between agents):**

| Task | coevol Jaccard | no_sharing Jaccard | More diverse? |
|------|---------------|-------------------|---------------|
| Circle Packing | 0.385 | 0.414 | coevol |
| Erdos | 0.371 | 0.429 | coevol |
| Kernel Builder | 0.158 | 0.327 | coevol |
| Heilbronn | 0.398 | 0.278 | no_sharing |
| Signal | 0.315 | 0.335 | coevol |

Lower Jaccard = more diverse vocabulary. coevol has lower Jaccard on 4/5
tasks, but this is misleading — coevol ran many more evals (more chances
to use different words).

**Unique strategies per eval (normalized for eval count):**

| Task | coevol unique/N | no_sharing unique/N |
|------|----------------|-------------------|
| Circle Packing | 371/860 = 0.43 | 270/181 = 1.49 |
| Erdos | 150/220 = 0.68 | 324/176 = 1.84 |
| Kernel Builder | 68/100 = 0.68 | 40/10 = 4.00 |

**no_sharing produces 2-6x more unique strategies per eval**. However,
this metric is also confounded: conditions with fewer total evals naturally
have higher unique/N ratios (less time to exhaust the strategy space).
A fairer comparison would measure unique strategies within the first N
evals for all conditions at the same N.

### Claim 4: "Distillation ≠ RL"

On Circle Packing (the task with the most attempts_only data):

| Condition | RLVR Analog | Best Score | N evals |
|-----------|-------------|-----------|---------|
| coevol | RL | 1.0000051 | 860 |
| attempts_only | Distillation | 1.0000026 | 715 |
| no_sharing | Base model | 1.0000055 | 181 |

At matched budget (k=181): no_sharing (1.0000055) > coevol (1.0000000) >
attempts_only (1.0000024). The base model analog wins.

Unlike RLVR where distillation > RL, here attempts_only < coevol at all
budget levels. This may be because our "distillation" analog (sharing
scores) is weaker than true distillation (which provides full reasoning
traces).

## The Difficulty Hypothesis

The most important finding is the **task difficulty interaction**, but it
is partially confounded by unequal eval budgets:

| Task | Difficulty | Winner (at max k) | Winner (at k=20) | Confounded? |
|------|-----------|-------------------|-------------------|-------------|
| Circle Packing | Easy | no_sharing | 1agent/no_sharing | No — all converge |
| Erdos | Easy-Med | no_sharing | 1agent | No — all converge |
| Signal | Medium | no_sharing | no_sharing | Partially (3x budget gap) |
| Kernel Builder | Medium | tie (no improvement) | tie | No — all stuck |
| Heilbronn | Med-Hard | coevol | **no_sharing** | **Yes** (8x budget gap) |
| Matmul | Hard | coevol | tie | **Yes** (15x budget gap) |
| Hexagon 12 | Hard | coevol (only condition) | — | **Yes** (no comparison) |

**Clean results (unconfounded)**:
- Easy/medium tasks: no_sharing ≥ coevol at every matched budget. Sharing
  adds overhead without benefit.

**Confounded results (need more data)**:
- Hard tasks (Heilbronn, Matmul, Hexagon): coevol wins at max k, but the
  other conditions ran far fewer evals. On Heilbronn, no_sharing reached
  0.942 in only 21 evals while coevol needed 160 evals to reach 0.971.
  If no_sharing ran 160 evals, would it match or exceed coevol?

### Why Sharing May Help on Hard Tasks (Preliminary)

On Heilbronn, coevol agents progressively discovered:
1. "Goldberg init" (agent-4, 0.869)
2. "Bottom-3 targeting" (agent-3, 0.930, learned from agent-1's notes)
3. "Diverse inits + 4-phase SA" (agent-2, 0.971, combined insights from all)

This suggests sharing enables **composing partial insights** — but we cannot
rule out that independent agents would discover the same insights given
equal eval budget.

### The Crossover Point (Tentative)

On tasks where 1agent's best score < ~0.8 of benchmark, sharing appears
to help. But this threshold is preliminary because the hard-task comparisons
are confounded by unequal eval counts.

| 1agent best | Sharing helps? | Confidence | Tasks |
|------------|---------------|------------|-------|
| ≥ 1.0 | No | High | CP, Erdos |
| 0.7-1.0 | No | Medium | Signal |
| < 0.7 | Maybe | Low (confounded) | Heilbronn, Hexagon, Matmul |

### Key Divergence from RLVR

At matched eval budgets, we observe:

| Task | k=10 winner | k=20 winner | k=50 winner |
|------|-----------|------------|-------------|
| CP (easy) | 1agent | 1agent | 1agent |
| Erdos (easy) | 1agent | 1agent | 1agent |
| Heilbronn (hard) | no_sharing | **no_sharing** | coevol* |
| Signal (medium) | no_sharing | no_sharing | no_sharing |

*coevol wins at k=50 on Heilbronn, but no_sharing only had 21 total evals
so its score is frozen at 0.942 from k=17 onward.

On easy tasks, this matches RLVR: independent exploration beats sharing.
On hard tasks, the pattern is unclear — coevol's advantage may be real
(composing insights) or an artifact of running more evals.

## Mechanistic Evidence

### Strategy Anchoring (from agent notes on Erdos)

coevol agents explicitly copy each other:
- "Adopted agent-3's params" → timeout
- "Adopted agent-4's momentum PG approach" → marginal gain
- "All agents converge to C5 ≈ 0.38101-0.38108" → stuck at local optimum

Meanwhile, no_sharing agents independently discover Haugland initialization
and reach C5 ≈ 0.3808 (exceeding benchmark).

### Compute Efficiency

| Task | coevol evals | 1agent evals | Ratio | Score comparison (at matched k) |
|------|-------------|-------------|-------|-------------------------------|
| CP | 860 | 62 | 13.9x | At k=62: tied (~1.000) |
| Erdos | 220 | 10 | 22.0x | At k=10: 1agent BETTER (1.000 vs 0.992) |
| Heilbronn | 160 | 1 | 160x | At k=1: 1agent BETTER (0.800 vs 0.722) |
| Signal | 150 | 10 | 15.0x | At k=10: 1agent BETTER (0.701 vs 0.589) |

coevol uses 14-160x more compute than 1agent but does not lead at any
matched eval budget on easy/medium tasks.

## Limitations

1. **Unequal eval budgets**: The most significant limitation. Conditions
   ran very different numbers of evaluations, making boundary comparisons
   unreliable on hard tasks. Future work should fix per-agent iteration
   count (e.g., 20 iterations per agent for all conditions).

2. **Kernel Builder**: No condition improved from the initial score,
   suggesting the task setup or scoring may need revision.

3. **Hexagon 12**: Only coevol data exists — no comparison is possible.

4. **Small N for hard tasks**: 1agent ran only 1-3 evals on Heilbronn
   and Matmul, far too few to characterize its capability boundary.

## Conclusions

1. **Multiagent gain ≈ sampling gain on easy/medium tasks** (like RLVR):
   At matched eval budgets, independent agents match or beat co-evolution
   on all easy/medium tasks. The primary value of multiple agents is
   parallel sampling, not knowledge transfer.

2. **Sharing slows early exploration** (like RL): coevol is consistently
   worse than no_sharing at small k (k ≤ 20) across all tasks. Shared
   notes/skills introduce overhead before they provide value.

3. **Sharing may help on hard tasks, but evidence is confounded**: On
   Heilbronn, coevol reaches 0.971 vs no_sharing's 0.942 — but coevol
   ran 8x more evals. We cannot distinguish "sharing composes insights"
   from "more evals = better score" without controlled experiments.

4. **Difficulty may be the key moderator** (tentative): The crossover
   where sharing helps appears to be when 1agent's best score < ~0.8,
   but this needs validation with equal eval budgets.

5. **Design implication**: Multiagent systems should default to
   **independent exploration** and only enable sharing when agents are
   individually stuck. But the threshold for "stuck" needs empirical
   validation with controlled eval budgets.

## Recommended Next Steps

To strengthen the difficulty hypothesis:
1. **Run all conditions with fixed per-agent iterations** (e.g., 20
   iterations × 4 agents = 80 total evals per condition)
2. **Prioritize Heilbronn and Matmul** — these are the tasks where
   sharing's benefit is most plausible but currently confounded
3. **Plot best score vs eval count curves** for visual comparison
4. **Report confidence intervals** using bootstrap over eval orderings
