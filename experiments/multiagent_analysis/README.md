# Multiagent Gain Source Analysis

Inspired by "Limit of RLVR" (Yue et al., 2025), which showed that RLVR improves
sampling efficiency without expanding the capability boundary of base models,
we analyze **why multiagent systems work** through the same lens.

## Core Question

Is multiagent co-evolution just efficient sampling (like RLVR), or does it
genuinely expand the capability boundary (like distillation)?

## Experimental Design

### Exp A: Sampling Efficiency — Is multiagent just best-of-N?

| Condition | Agents | Sharing | Total Compute |
|-----------|--------|---------|---------------|
| `1agent_Nruns` | 1 agent × N independent runs | None | N × budget |
| `Nagent_coevol` | N agents co-evolving | Full | N × budget |

**Metric**: Compare best score at matched total eval count.
If similar → multiagent gain is mostly sampling.

### Exp B: Capability Boundary — Can multiagent solve unsolvable problems?

| Condition | Agents | Sharing | Budget |
|-----------|--------|---------|--------|
| `1agent_long` | 1 agent, very long run | Self-only | 4× budget |
| `4agent_coevol` | 4 agents co-evolving | Full | 1× budget each |

**Metric**: Track the set of "solved sub-problems" or score thresholds reached.
Does co-evolution reach scores that 1-agent never reaches even with 4× compute?
(Analogous to RLVR's pass@k analysis)

### Exp C: Knowledge Transfer — Is shared knowledge distillation or RL?

| Condition | Agents | Notes | Skills | Attempts |
|-----------|--------|-------|--------|----------|
| `full_sharing` | 4 | ✓ | ✓ | ✓ |
| `no_knowledge` | 4 | ✗ | ✗ | ✓ |
| `no_sharing` | 4 | ✗ | ✗ | ✗ |
| `attempts_only` | 4 | ✗ | ✗ | ✓ |

**Metric**: Score trajectory + exploration diversity.
- If `no_knowledge` ≈ `full_sharing` → knowledge doesn't help (pure sampling)
- If `full_sharing` >> `no_knowledge` → knowledge acts as distillation
- If `no_sharing` ≈ `1agent_Nruns` → multiagent gain is purely from sampling

### Exp D: Exploration Narrowing — Does shared knowledge reduce diversity?

Track per-agent strategy diversity (Jaccard similarity of attempt titles)
across conditions. If sharing increases similarity → exploration narrowing
(analogous to RLVR narrowing the base model's exploration space).

## Tasks

We use tasks from CORAL's existing benchmarks:
1. **circle_packing** (math, moderate difficulty)
2. **kernel_builder** (systems, high difficulty)
3. **tsp** (optimization, easy to set up)

## Running on b300

```bash
# From b300 head node
cd /fsx/xuanj/CORAL
uv sync
./experiments/multiagent_analysis/run_all.sh
```
