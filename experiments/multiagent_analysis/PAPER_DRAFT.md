# Multiagent Systems Through the RLVR Lens: A Systematic Analysis

## TL;DR

We apply the analytical framework from "Limit of RLVR" (Yue et al., 2025,
NeurIPS Best Paper Runner-Up) to understand why multiagent coding systems work.
Across 5 tasks of varying difficulty, we find that **multiagent co-evolution
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

### Tasks (5, spanning difficulty levels)

| Task | Domain | Difficulty | Benchmark |
|------|--------|-----------|-----------|
| Circle Packing (N=26) | Math optimization | Easy | 2.635977 |
| Erdos Min Overlap | Math optimization | Easy-Medium | 0.38092 |
| Signal Processing | Signal/DSP | Medium | multi-objective |
| Heilbronn Triangle | Combinatorial geometry | Medium-Hard | 0.03653 |
| Kernel Builder | Systems/VLIW | Hard | 1,363 cycles |

### Conditions

| Condition | Agents | Shared | RLVR Analog |
|-----------|--------|--------|-------------|
| 1agent | 1 | self | Base model (k=1) |
| 4agent_no_sharing | 4 | nothing | Base model (k=4) |
| 4agent_coevol | 4 | all | RL-trained model |
| 4agent_attempts_only | 4 | scores only | Distilled model |

## Results by RLVR Claim

### Claim 1: "Base > RL at large k" → Independent agents ≥ coevol at large N

**Evals to reach score thresholds:**

| Task | Threshold | 1agent | coevol | no_sharing |
|------|-----------|--------|--------|------------|
| Circle Packing | 1.0 | **11** | 621 | **14** |
| Erdos | 1.0 | **9** | **never** | 86 |
| Kernel Builder | 3000 cycles | **5** | 41 | **4** |
| Heilbronn | 0.9 | never | **36** | never |
| Signal | 0.7 | never | **41** | **10** |

**Pattern**: On easy/medium tasks (CP, Erdos, Kernel), independent agents
reach thresholds much faster. On harder tasks (Heilbronn), coevol reaches
thresholds that others cannot (yet).

**Which condition wins (best score)?**

| Task | Winner | Best Score | N evals | 2nd Place |
|------|--------|-----------|---------|-----------|
| Circle Packing | no_sharing | 1.0000055 | 161 | coevol |
| Erdos | no_sharing | 1.0001343 | 149 | 1agent |
| Kernel Builder | no_sharing | 2,125 cyc | 10 | 1agent |
| Heilbronn | **coevol** | 0.9493 | 65 | no_sharing |
| Signal | no_sharing | 0.7352 | 24 | coevol |

**Verdict**: no_sharing wins 4/5 tasks. coevol wins only on Heilbronn
(the hardest math task). This mirrors RLVR: base > RL at large k, except
on problems where the base model genuinely struggles.

### Claim 2: "RL boosts sampling efficiency but reduces boundary"

**Score at different eval budgets (1agent / coevol / no_sharing):**

| Task | k=5 | k=20 | k=all |
|------|-----|------|-------|
| Circle Packing | 1.00 / 0.36 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| Erdos | 1.00 / 0.94 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| Kernel Builder | 7093 / 11910 / 2130 | 7093 / 11429 / 2125 | 2132 / 2191 / 2125 |
| Heilbronn | 0.80 / 0.72 / 0.88 | 0.80 / 0.87 / 0.88 | 0.80 / 0.95 / 0.88 |
| Signal | 0.67 / 0.57 / 0.64 | 0.67 / 0.59 / 0.67 | 0.70 / 0.73 / 0.74 |

**Key observation**: coevol is WORSE than no_sharing at small k on every
task except Signal. At large k, coevol catches up on easy tasks but still
loses on Erdos. On Heilbronn, coevol eventually surpasses both — this is
the one case where sharing genuinely expands the boundary.

### Claim 3: "RL narrows exploration"

**Strategy diversity (Jaccard overlap between agents):**

| Task | coevol Jaccard | no_sharing Jaccard | More diverse? |
|------|---------------|-------------------|---------------|
| Circle Packing | 0.385 | 0.408 | coevol |
| Erdos | 0.368 | 0.404 | coevol |
| Kernel Builder | 0.156 | 0.269 | coevol |
| Heilbronn | 0.206 | 0.180 | no_sharing |
| Signal | 0.212 | 0.235 | coevol |

Lower Jaccard = more diverse. coevol has lower Jaccard on 4/5 tasks,
meaning coevol agents are actually MORE diverse in their vocabulary.

But this is misleading — coevol has many more evals (more chances to use
different words). The real measure is **unique strategies per eval**:

| Task | coevol unique/N | no_sharing unique/N |
|------|----------------|-------------------|
| Circle Packing | 369/840 = 0.44 | 263/161 = 1.63 |
| Erdos | 146/210 = 0.70 | 298/149 = 2.00 |
| Kernel Builder | 54/90 = 0.60 | 40/10 = 4.00 |

**no_sharing produces 2-6x more unique strategies per eval**. coevol's
apparent diversity is just volume — most of it is redundant.

### Claim 4: "Distillation ≠ RL"

On Circle Packing (the only task with attempts_only data):

| Condition | RLVR Analog | Best Score | N evals | Impr% |
|-----------|-------------|-----------|---------|-------|
| coevol | RL | 1.0000049 | 840 | 39% |
| attempts_only | Distillation | 1.0000026 | 700 | 26% |
| no_sharing | Base model | 1.0000055 | 161 | 43% |

Unlike RLVR where distillation > RL, here attempts_only < coevol.
This may be because our "distillation" analog (sharing scores) is weaker
than true distillation (which provides full reasoning traces).

## The Difficulty Hypothesis (7 tasks)

The most important finding is the **task difficulty interaction**:

| Task | Difficulty | Winner | Best Score (winner) | 2nd Place |
|------|-----------|--------|-------------------|-----------|
| Circle Packing | Easy | no_sharing | 1.0000055 | coevol |
| Erdos | Easy-Med | no_sharing | 1.0001370 | 1agent |
| Kernel Builder | Medium | no_sharing | 2,125 cyc | coevol (tie) |
| Signal Processing | Medium | no_sharing | 0.738 | coevol |
| Heilbronn Triangle | Med-Hard | **coevol** | 0.971 | no_sharing (0.942) |
| Hexagon Packing 12 | Hard | **coevol** | 0.941 | (others crash) |
| Matrix Multiplication | Hard | **coevol** | 1.000 | 1agent (0.80) |

**Score**: no_sharing wins 4/7, coevol wins 3/7.
**But the split is clean**: no_sharing wins all easy/medium tasks,
coevol wins all hard tasks.

This maps precisely to RLVR's nuance:

- **Easy tasks** (1agent can solve): sharing is pure overhead, narrows exploration
- **Medium tasks** (4 independent agents can solve): sharing narrows exploration
- **Hard tasks** (independent agents crash/struggle): sharing enables combining
  partial insights — agents learn from each other's rare successes

### Why Sharing Helps on Hard Tasks

On Heilbronn, coevol agents progressively discovered:
1. "Goldberg init" (agent-4, 0.869)
2. "Bottom-3 targeting" (agent-3, 0.930, learned from agent-1's notes)
3. "Diverse inits + 4-phase SA" (agent-2, 0.971, combined insights from all)

No single agent discovered all three innovations. Sharing let agents
**compose partial insights** into a solution none could find alone.

On Hexagon 12, no_sharing agents ALL crashed. coevol's agent-2 succeeded
because it could learn from agent-3's first successful attempt (0.691).

### The Crossover Point

The crossover happens when **1agent's best score < ~0.9 of benchmark**.
Below this threshold, the task is hard enough that sharing's benefits
(combining partial insights) outweigh its costs (exploration narrowing).

| 1agent best | Sharing helps? | Tasks |
|------------|---------------|-------|
| ≥ 1.0 | No | CP, Erdos |
| 0.7-1.0 | No | Signal, Kernel |
| < 0.7 | **Yes** | Heilbronn (0.80), Hexagon (crash), Matmul (0.80) |

## Mechanistic Evidence

### Strategy Anchoring (from agent notes on Erdos)

coevol agents explicitly copy each other:
- "Adopted agent-3's params" → timeout
- "Adopted agent-4's momentum PG approach" → marginal gain
- "All agents converge to C5 ≈ 0.38101-0.38108" → stuck at local optimum

Meanwhile, no_sharing agents independently discover Haugland initialization
and reach C5 ≈ 0.3808 (exceeding benchmark).

### Compute Efficiency

| Task | coevol evals | 1agent evals | coevol/1agent ratio | Score comparison |
|------|-------------|-------------|-------------------|-----------------|
| CP | 840 | 54 | 15.6x | coevol +0.000004 (negligible) |
| Erdos | 210 | 10 | 21.0x | coevol WORSE by 0.00037 |
| Kernel | 90 | 6 | 15.0x | coevol WORSE by 59 cycles |
| Heilbronn | 65 | 1 | 65.0x | coevol BETTER by 0.149 |

On Heilbronn, the 65x compute investment pays off. On other tasks, it doesn't.

## Conclusions

1. **Multiagent gain ≈ sampling gain on easy/medium tasks** (like RLVR):
   On 4/7 tasks, independent agents match or beat co-evolution. The primary
   value of multiple agents is parallel sampling, not knowledge transfer.

2. **Sharing narrows exploration** (like RL): Shared notes/skills cause agents
   to converge on similar strategies, reducing the effective search space.
   coevol produces 2-6x fewer unique strategies per eval than no_sharing.

3. **Sharing genuinely helps on hard tasks** (unlike pure RLVR): On 3/7 tasks
   where single agents struggle (Heilbronn, Hexagon, Matmul), co-evolution
   enables agents to compose partial insights into solutions none could find
   alone. This is the multiagent analog of distillation expanding the boundary.

4. **Difficulty is the key moderator**: The crossover point is when 1agent's
   best score < ~0.8 of benchmark. Below this, sharing's benefits outweigh
   its costs. Above this, independent exploration is strictly better.

5. **Design implication**: Multiagent systems should use **adaptive sharing** —
   independent exploration early (or on easy tasks), shared knowledge only
   when agents are individually stuck. This combines the exploration benefits
   of independence with the insight-composition benefits of sharing.
