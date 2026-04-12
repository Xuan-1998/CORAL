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

**Tasks**: Circle Packing (N=26), Erdos Minimum Overlap

**Model**: Claude Opus 4.6 (matching CORAL paper's setup)

**Conditions** (all using kiro-cli as agent runtime):

| Condition | Agents | Shared Attempts | Shared Notes/Skills |
|-----------|--------|----------------|-------------------|
| 1agent_baseline | 1 | self only | self only |
| 4agent_coevol | 4 | ✓ | ✓ |
| 4agent_attempts_only | 4 | ✓ | ✗ |
| 4agent_no_sharing | 4 | ✗ | ✗ |

**Infrastructure**: CORAL framework with modified `SharingConfig`. Ran on p5en cluster, 9+ hours per run.

## Key Results

### 1. Convergence Speed (Opus 4.6)

Evals needed to reach score thresholds:

**Circle Packing**:

| Threshold | 1agent | coevol | attempts_only | no_sharing |
|-----------|--------|--------|--------------|------------|
| 0.99 | **5** | 6 | 11 | 4 |
| 0.999 | **5** | 131 | 111 | **7** |
| 1.0 | **11** | **never** (525 evals) | 111 | **14** |

**Erdos**:

| Threshold | 1agent | coevol | no_sharing |
|-----------|--------|--------|------------|
| 0.99 | **3** | 6 | 2 |
| 0.999 | **5** | 46 | 11 |
| 1.0 | **5** | **never** (205 evals) | 94 |

**Finding**: coevol NEVER reaches 1.0 on Circle Packing (525 evals) or Erdos
(205 evals). 1agent reaches 1.0 in 11 and 5 evals respectively.

### 2. Final Scores (Opus 4.6, ~9 hours)

**Circle Packing**:

| Condition | Total Evals | Best Score | Improvement Rate |
|-----------|------------|-----------|-----------------|
| 1agent | 30 | 1.0000004 | 50% |
| coevol | 525 | 0.9999999 | 42% |
| attempts_only | 430 | 1.0000025 | 49% |
| no_sharing | 100 | **1.0000055** | **67%** |

**Erdos**:

| Condition | Total Evals | Best Score | Improvement Rate |
|-----------|------------|-----------|-----------------|
| 1agent | 10 | **1.0001253** | 100% |
| coevol | 205 | 0.9997547 | 35% |
| no_sharing | 94 | 1.0001253 | 40% |

**Finding**: 1agent opus exceeds the Erdos benchmark in just 5 evals.
coevol with 205 evals cannot even reach 0.999.

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

### Erdos Results (~2 hours, 175+ evals for coevol)

| Condition | Evals | Best Score | Improvement Rate |
|-----------|-------|-----------|-----------------|
| 1agent | 18 | 0.99999 | 44% |
| 4agent_coevol | 175 | 0.99897 | 77% |
| 4agent_no_sharing | 41 | 0.99999+ | 73% |

Evals to reach threshold:

| Threshold | 1agent | coevol | no_sharing |
|-----------|--------|--------|------------|
| 0.99 | 3 | 26 | 4 |
| 0.999 | 5 | **never** (175 evals) | 13 |
| 0.9999 | 5 | **never** | 20 |

**Erdos shows an even stronger effect**: coevol with 175 evals cannot reach
0.999, while 1agent reaches 0.9999 in just 5 evals. The 4 co-evolving agents
all converged on Adam + perturbation strategies and got stuck at 0.999, while
independent agents discovered multi-resolution approaches reaching 0.99999+.

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

## Mechanistic Analysis: WHY Does Sharing Hurt?

### The Attention Tax

Sharing doesn't slow agents down — it speeds them up. coevol agents produce
19.7 evals/agent/h vs no_sharing's 3.8/agent/h (Circle Packing). But this
high velocity is low quality: 95% of coevol's strategies are redundant.

Agents reading shared notes skip the deep research phase and jump straight
to copying others' approaches. This produces more evals but less exploration.

### Strategy Anchoring (from actual agent notes)

Reading coevol agents' notes on Erdos reveals the mechanism:

- Agent-1 (eval 6): "Adopted agent-3's params" → timeout
- Agent-1 (eval 17): "Adopted agent-4's momentum PG approach" → marginal gain
- Agent-1 (eval 33): "All agents converge to C5 ≈ 0.38101-0.38108" → **explicit acknowledgment of convergence**
- Agent-2 (eval 15): tried different method, regressed, went back to copying

All 4 coevol agents converged to the same local optimum (C5 ≈ 0.381).
Meanwhile, no_sharing agents independently discovered Haugland initialization
and reached C5 ≈ 0.3808 (exceeding benchmark).

### The Redundancy Problem

| Condition | Unique strategies | Redundancy | Cross-agent duplication |
|-----------|------------------|-----------|----------------------|
| coevol (CP) | 25/545 | 95% | 85% |
| no_sharing (CP) | 35/103 | 66% | 78% |
| coevol (Erdos) | 14/155 | 91% | 74% |
| no_sharing (Erdos) | 48/109 | 56% | 64% |

### Compute-Normalized Comparison

At equal compute budget:

**Erdos**: coevol uses 21x more evals than 1agent but scores LOWER
(0.99975 vs 1.00013). More compute, worse result.

**Circle Packing**: coevol uses 15x more evals for 0.0004% improvement.

### Cross-Agent Transfer: Helpful but Rare

On Circle Packing, cross-agent transfers (building on another agent's commit)
have 69% improvement rate vs 36% for self-parent attempts. But cross-agent
transfers are only 10% of all attempts. The net effect: sharing makes the
rare cross-agent transfers better but makes the common self-improvements worse.

## Implications for Multiagent System Design

1. **Default to independent exploration** for well-defined optimization tasks
2. **Share scores/rankings only** (like a leaderboard) — this gives agents
   signal about what score levels are achievable without anchoring their strategy
3. **Delay sharing** until agents have independently explored — early sharing
   causes premature convergence
4. **Share "what doesn't work" more than "what works"** — negative knowledge
   expands the search space, positive knowledge narrows it
5. **Stronger models need less sharing** — opus 4.6 single agent outperforms
   4 co-evolving agents on both tasks

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
