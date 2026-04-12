# Systematic Experiment Plan: RLVR Lens on Multiagent

## RLVR Claims → Multiagent Analogs → Required Evidence

### Claim 1: "Base > RL at large pass@k" → "1agent/no_sharing ≥ coevol at large N"

**What RLVR did**: pass@k curves across k=1..1024 for base vs RL models
**Our analog**: best_score@N curves across N=1..max for each condition
**Current coverage**: CP ✅, Erdos ✅, Kernel ⚠️ (early)
**Gap**: Need more tasks spanning difficulty levels. Need proper pass@k analog
(not just running best — need "probability of reaching threshold at eval k")

### Claim 2: "RL boosts sampling efficiency but reduces boundary"

**What RLVR did**: Showed RL models solve fewer unique problems at large k
**Our analog**: Count unique score thresholds reached by each condition
**Current coverage**: ✅ Have capability boundary tables
**Gap**: Need to formalize as "problem coverage" — at each difficulty level,
which condition can solve it? This maps directly to RLVR's per-problem analysis.

### Claim 3: "RL algorithms perform similarly"

**What RLVR did**: Compared PPO, GRPO, Reinforce++
**Our analog**: Compare different sharing mechanisms
**Current coverage**: ⚠️ Only tested full-sharing vs attempts-only vs none
**Gap**: Need more sharing variants:
- Share notes only (no attempts, no skills)
- Share skills only
- Delayed sharing (independent first N evals, then share)
- Leaderboard only (share scores but not code/notes)

### Claim 4: "Distillation ≠ RL" (distillation expands boundary)

**What RLVR did**: Showed distilled models have wider capability than RL models
**Our analog**: attempts_only (see others' scores) vs coevol (full knowledge)
**Current coverage**: ✅ attempts_only > coevol on CP long-term
**Gap**: Need cleaner distillation analog. Maybe: "warm-start from best
solution" vs "co-evolve from scratch"

## Task Selection for Comprehensive Coverage

### Difficulty Tiers (by single-agent solvability)

**Easy** (1agent reaches benchmark in <20 evals):
- Circle Packing ✅ (running)
- Erdos Min Overlap ✅ (running)

**Medium** (1agent reaches >0.9 but not benchmark):
- Heilbronn Triangle
- Signal Processing
- First Autocorrelation Inequality

**Hard** (1agent struggles to reach 0.5):
- Kernel Builder ✅ (running)
- Matrix Multiplication Tensor Decomposition
- Hexagon Packing N=12

**Domain diversity**:
- Math optimization: CP, Erdos, Heilbronn ✅
- Kernel/systems: Kernel Builder ✅
- ML/data science: MNIST, Spaceship Titanic
- Biology: DNA Enhancer, Drug Design

## Batch Experiment Design

For each task, run 3 conditions (minimum):
1. `1agent` — single agent baseline
2. `4agent_coevol` — full sharing (RL analog)
3. `4agent_no_sharing` — independent (base model analog)

Optional 4th: `4agent_attempts_only` — score sharing (distillation analog)

All with: opus 4.6, max_turns=50, kiro runtime

## Priority Order

1. **Kernel Builder** — already running, critical for "hard task" hypothesis
2. **Heilbronn Triangle** — medium difficulty math, good diversity
3. **Signal Processing** — different problem structure
4. **MNIST** — ML task, very different from optimization
5. **Matrix Multiplication** — hardest math task

## Analysis Deliverables

1. **pass@k curves** per task (Figure 1 analog from RLVR)
2. **Capability boundary table** — which conditions reach which thresholds
3. **Sampling efficiency gap (ΔSE)** — how far is coevol from optimal?
4. **Exploration diversity** — strategy overlap per condition
5. **Cross-task summary** — does the pattern hold across difficulty levels?
