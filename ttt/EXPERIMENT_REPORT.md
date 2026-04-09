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

## Round 2: Qwen3-32B (2026-04-09 afternoon)

### Setup
- vLLM serving Qwen3-32B with TP=4 on 8×H200 (node 9)
- OpenCode agent on node 10
- `--max-model-len 65536` with `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`

### Results
- **Agent capability**: Much stronger than 8B. Qwen3-32B can:
  - Reason about code structure in `<think>` blocks
  - Correctly use read/edit/bash tools
  - Identify and analyze bugs (found indentation error)
  - Attempt to use `coral eval` for submission
- **7 eval attempts** produced, all crashed (score=0)

### New Blockers
1. **OpenCode edit tool + git worktree mismatch**: Agent edits files via OpenCode's edit tool, but git doesn't detect changes. The edit tool may use a virtual filesystem that doesn't write to the actual git worktree. All `coral eval` attempts fail with "nothing to commit".
2. **Context overflow (again)**: After ~7 tool calls, conversation reaches 33K+ input tokens. OpenCode requests 32K output → total exceeds 65K limit. Agent gets stuck in compaction/retry loop.
3. **OpenCode hardcodes max_tokens=32000**: This is the root cause of all context overflow issues. Any model with context < 65K will eventually overflow as conversation grows.

### Key Finding
**Qwen3-32B has sufficient agent capability** — the failures are all infrastructure issues (OpenCode tool integration), not model intelligence. With proper tool integration, 32B should work for CORAL TTT.

## Fundamental Tension (Updated)

The problem is not just "strong models aren't trainable" — it's a **three-way tension**:

| Requirement | Constraint |
|---|---|
| Strong agent capability | Need ≥32B model |
| Trainable weights | Need open-source model served locally |
| Compatible agent runtime | OpenCode has bugs with git worktree + hardcoded max_tokens |

## Next Steps

1. **Most promising: ThetaEvolve-style fixed pipeline** — Don't use OpenCode as autonomous agent. Instead:
   - Use vLLM to serve Qwen3-32B
   - Write a simple prompt → generate code → eval loop (like ThetaEvolve)
   - Use CORAL's grader for evaluation
   - Do GRPO on the generated code
   - This bypasses all OpenCode integration issues

2. **Kernel engineering** — Better suited for fixed pipeline approach since task structure is clear (generate Triton kernel → compile → benchmark runtime). But needs GPU for both model serving AND kernel evaluation.

3. **Fix OpenCode integration** — File upstream issues for:
   - Edit tool not writing to actual filesystem in git worktrees
   - Hardcoded max_tokens=32000 (should be configurable or adaptive)
