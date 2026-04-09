# CORAL TTT Experiment Report

## Date: 2026-04-09

## Objective
Integrate test-time training (RL) into CORAL's autonomous agent framework, using an open-source model (Qwen3) served via vLLM, with GRPO weight updates.

## What Worked

### Vanilla CORAL + kiro-cli + Claude API ✅
- CORAL agent (kiro-cli runtime) successfully started and autonomously solved circle packing
- Agent read task description, searched literature, designed optimization strategy, wrote code, submitted evals
- Best score: **0.364** (sum_radii / 2.635977) in ~6 evals
- Proves CORAL framework works on p5en cluster

### Vanilla CORAL + OpenCode + Qwen3-8B via vLLM ✅ (partial)
- vLLM successfully served Qwen3-8B with 131K context on H200
- OpenCode agent connected to vLLM and generated code
- Agent produced 2 evals with score **0.364**
- Proves the vLLM → OpenCode → CORAL pipeline works

### Infrastructure ✅
- CORAL installed and working on p5en (`/fsx/xuanj/CORAL-xuan/`)
- kiro-cli, Claude Code, OpenCode all installed
- vLLM serving Qwen3-8B on H200 GPUs
- LoRA training setup (peft) verified — 7.6M trainable params (0.09%)

## What Didn't Work

### rllm verl backend ❌
- `feat/add_rl` branch's TTT module depends on rllm + tinker backend
- tinker requires paid API key (Thinking Machines Lab)
- verl backend has import error: `VerlEngine` not found in rllm 0.3.0rc0
- **Root cause**: rllm and verl version incompatibility

### OpenCode + Qwen3-1.7B ❌
- OpenCode requests 32000 output tokens by default
- Qwen3-1.7B max context = 40960 → overflow after ~9K input tokens
- **Fix**: switched to Qwen3-8B (131K context)

### CORAL gateway (litellm proxy) ❌
- Gateway never started on port 4000 despite `gateway.enabled: true` in config
- Likely a litellm startup issue or path resolution problem
- **Workaround**: pointed OpenCode directly to vLLM on port 8000

### GRPO training ❌
- Without gateway, no LLM request/response traces were captured
- Rewrote to use git diffs from CORAL attempts as training signal
- But Qwen3-8B as autonomous agent is too weak — gets stuck in tool-calling loops
- Specifically: can't correctly invoke `coral eval` via OpenCode's bash tool schema
- Agent loops on `coral eval --message "..."` → bash tool rejects because `description` param missing

### Qwen3-8B as coding agent ❌
- Can generate code (wrote genetic algorithm for circle packing)
- But fails at multi-step autonomous agent workflow:
  - Incorrect tool call formatting
  - Gets stuck in retry loops
  - Can't recover from tool errors
- CORAL paper uses Claude Opus 4.6 for a reason — small open models lack agent capability

## Key Insight

**The fundamental tension in CORAL TTT:**

| Requirement | What it needs | Conflict |
|---|---|---|
| CORAL agent | Strong coding agent (Claude Opus level) | Weights not trainable |
| RL training | Open-source model with trainable weights | Too weak as autonomous agent |

This is the core research challenge. Possible solutions:

1. **Distillation-first**: Use Claude to generate high-quality CORAL trajectories, then distill into an open model, then do TTT on the distilled model
2. **Larger open model**: Qwen3-72B or Llama-4-Maverick might be strong enough as agents while still being trainable (needs 4-8 H200s)
3. **Simplified agent loop**: Don't use full CORAL agent autonomy — use a ThetaEvolve-style fixed pipeline (prompt → generate code → eval → GRPO) with the open model, but leverage CORAL's knowledge structure (notes, skills, attempts)
4. **Hybrid**: Use Claude for agent decisions (explore/exploit, what to try next), use open model for code generation only, train the open model with RL

## Files on p5en

```
/fsx/xuanj/CORAL-xuan/          # Your fork with TTT module
/fsx/xuanj/CORAL-xuan/ttt/      # TTT trainer code
/fsx/xuanj/CORAL-ttt/            # Original feat/add_rl branch
/fsx/xuanj/coral-ttt-venv/       # Python venv with all deps
```

## Next Steps

1. **Option 3 is most promising**: Write a simplified ThetaEvolve-style loop that uses CORAL's evaluator but not its full agent autonomy. The open model just generates code diffs, not full agent trajectories.
2. Try Qwen3-32B or 72B if GPU budget allows — might cross the agent capability threshold.
3. Get CORAL gateway working for proper trace collection.
