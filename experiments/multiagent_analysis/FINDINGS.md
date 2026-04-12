# Why Multiagent Works: An Analysis Through the RLVR Lens

## Motivation

"Limit of RLVR" (Yue et al., 2025) showed that Reinforcement Learning with
Verifiable Rewards improves **sampling efficiency** without expanding the
**capability boundary** of base models. We apply this same analytical framework
to understand why multiagent systems (specifically CORAL) outperform single agents.

## Core Question

Is multiagent co-evolution just efficient sampling (like RLVR), or does it
genuinely expand the capability boundary (like distillation)?

## Experimental Setup

**Task**: Circle Packing (N=26, maximize sum of radii, benchmark=2.635977)

**Conditions** (all using kiro-cli as agent runtime):

| Condition | Agents | Shared Attempts | Shared Notes/Skills |
|-----------|--------|----------------|-------------------|
| 1agent_baseline | 1 | self only | self only |
| 4agent_coevol | 4 | ✓ | ✓ |
| 4agent_attempts_only | 4 | ✓ | ✗ |
| 4agent_no_sharing | 4 | ✗ | ✗ |

**Infrastructure**: CORAL framework with modified `SharingConfig` to control
per-item sharing. Ran on p5en cluster for ~9 hours.

## Key Results

### 1. Convergence Speed

Evals needed to reach score thresholds:

| Threshold | 1agent | coevol | attempts_only | no_sharing |
|-----------|--------|--------|--------------|------------|
| 0.99 | 15 | 26 | 16 | **2** |
| 0.999 | 17 | 106 | 36 | **6** |
| 1.0 | **17** | 261 | 516 | **14** |

**Finding**: Independent agents (no_sharing) converge 18x faster than
co-evolution to reach the benchmark. Even a single agent reaches 1.0 in
17 evals — faster than 4 co-evolving agents (261 evals).

### 2. Final Scores (after ~9 hours)

| Condition | Total Evals | Best Score | Improvement Rate |
|-----------|------------|-----------|-----------------|
| 1agent_baseline | 56 | 1.0000004 | 53.6% |
| 4agent_coevol | 645 | 1.0000023 | 40.3% |
| 4agent_attempts_only | 640 | 1.0000025 | 39.1% |
| 4agent_no_sharing | 141 | 1.0000024 | 54.6% |

**Finding**: All conditions eventually reach similar final scores (~1.000002),
but with vastly different efficiency. no_sharing achieves this with 141 evals
while coevol needs 645 — a 4.6x overhead.

### 3. Exploration Diversity

| Condition | Unique Strategies | Jaccard Similarity |
|-----------|------------------|-------------------|
| 4agent_coevol | 337 | 0.383 |
| 4agent_attempts_only | 361 | 0.405 |
| 4agent_no_sharing | 387 | 0.412 |

**Finding**: Surprisingly, no_sharing has the MOST unique strategies despite
having fewer total evals. Shared knowledge causes agents to converge on
similar approaches rather than exploring independently.

## RLVR Mapping

| RLVR Concept | Multiagent Analog | Evidence |
|-------------|-------------------|----------|
| **Sampling efficiency** (RL improves pass@1) | Sharing helps early convergence slightly | coevol reaches 0.95 at eval 11 vs no_sharing at eval 2 — no_sharing is actually faster |
| **Capability boundary** (base > RL at large k) | Independent agents match co-evolution | All conditions reach ~1.000002; sharing doesn't unlock higher scores |
| **Exploration narrowing** (RL shrinks solution space) | Shared notes/skills reduce diversity | coevol: 337 strategies vs no_sharing: 387 strategies |
| **Distillation ≠ RL** (distillation expands boundary) | Attempt sharing ≠ knowledge sharing | attempts_only eventually beats coevol on final score |

## Interpretation

### The Multiagent Gain is Primarily Sampling

Like RLVR, multiagent co-evolution's gains come from **running more
independent attempts**, not from knowledge transfer between agents.
The evidence:

1. **no_sharing matches or beats coevol** on both convergence speed and
   final score, despite having no inter-agent communication
2. **Improvement rate is highest for no_sharing** (54.6%) and lowest for
   coevol (40.3%) — shared knowledge causes agents to waste evals on
   approaches that look promising based on others' notes but don't
   actually improve their own score
3. **1 agent with 56 evals reaches the same ceiling** as 4 co-evolving
   agents with 645 evals — the capability boundary is set by the
   base model (LLM), not by the collaboration protocol

### Knowledge Sharing Can Hurt

This is the multiagent analog of RLVR's exploration narrowing:

- Agents reading shared notes converge on "proven" strategies
- This reduces exploration diversity (337 vs 387 unique strategies)
- The overhead of reading/writing shared knowledge slows iteration speed
- On well-defined optimization tasks, independent exploration is more efficient

### When Might Sharing Help?

Based on CORAL's own paper (Table 3), knowledge sharing helps most on
**complex, long-horizon tasks** (Kernel Engineering: 1350→1601 without
knowledge). Our circle_packing task may be "too easy" — a single agent
can solve it in 17 evals. The value of shared knowledge likely increases
with task complexity, similar to how distillation (vs RL) becomes more
valuable for harder problems.

## Cross-Task Validation: Erdos Minimum Overlap

To verify findings aren't task-specific, we ran the same ablation on Erdos
Minimum Overlap (minimize C5 upper bound, benchmark=0.38092).

### Erdos Results (~40 min)

| Condition | Evals | Best Score | Improvement Rate |
|-----------|-------|-----------|-----------------|
| 1agent | 3 | 0.99999 | 100% |
| 4agent_coevol | 21 | 0.99842 | 71.4% |
| 4agent_no_sharing | 22 | 0.99991 | 81.8% |

**Same pattern**: 1agent reaches near-benchmark in just 3 evals. Co-evolution
with 21 evals still hasn't caught up. no_sharing outperforms coevol.

### Cross-Task Consistency

| Metric | Circle Packing | Erdos |
|--------|---------------|-------|
| Most efficient | no_sharing (50.7%) | 1agent (100%) |
| Least efficient | coevol (33.5%) | coevol (71.4%) |
| Fastest to benchmark | no_sharing (14 evals) | 1agent (3 evals) |
| Slowest to benchmark | coevol (261 evals) | coevol (not reached) |

Co-evolution is consistently the least efficient condition across both tasks.

### Temporal Analysis: When Does Sharing Help?

Cross-agent code transfer only appears in the late phase of optimization,
but by then improvement rates have collapsed:

| Task | Phase | Cross-Agent % | Improvement Rate |
|------|-------|--------------|-----------------|
| Circle Packing | Early (Q1) | 0% | 69% |
| Circle Packing | Late (Q4) | 42% | 4% |
| Erdos | Early (Q1) | 0% | 93% |
| Erdos | Late (Q4) | 45% | 43% |

Agents start referencing each other's code heavily in the late phase, but
these references rarely lead to improvements. This mirrors RLVR's finding
that RL concentrates probability on known rewarded paths that have already
been fully explored.

### Technique Convergence (Erdos)

| Condition | Technique Overlap (Jaccard) |
|-----------|---------------------------|
| coevol | 0.700 (more convergent) |
| no_sharing | 0.655 (more diverse) |

Shared notes/skills cause agents to converge on similar optimization
techniques (Adam + perturbation in Erdos), while independent agents
discover more diverse approaches (multi-resolution, Haugland initialization).

## Next Steps

1. **Difficulty scaling**: Test on tasks of varying difficulty to find the
   crossover point where sharing starts helping
2. **Temporal analysis**: Track when cross-agent transfer becomes beneficial
   (early vs late in the optimization trajectory)
3. **Knowledge quality**: Analyze whether the content of shared notes/skills
   is actually useful or just noise

## Code

All experiment code is in `experiments/multiagent_analysis/`:
- `configs/` — YAML configs for each condition
- `analyze_gains.py` — General analysis script
- `deep_report.py` — Detailed convergence analysis
- `quick_report.py` — Quick summary
- `run_all.sh` — Batch runner

CORAL modifications (branch `ablation-study`):
- `coral/workspace/worktree.py` — `SharingConfig` now controls symlinks
- `coral/agent/builtin/kiro.py` — Kiro runtime prepends KIRO.md to prompts
- `coral/agent/manager.py` — Passes sharing flags to workspace setup
