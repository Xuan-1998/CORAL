# CORAL TTT with slime

Replaces the hacky rllm+verl stack with slime's proper RL framework.

## Architecture

```
slime training (Megatron/GRPO)
    ↕ weight sync
slime rollout (SGLang serves model)
    ↕ API calls
CORAL agent (kiro-cli / OpenCode)
    ↕ code generation + eval
CORAL grader (remote GPU eval)
    ↕
Score → slime Sample.reward
```

## Key file: `slime_rollout.py`

Custom rollout function for slime that:
1. Starts/resumes CORAL agent
2. Waits for N eval attempts
3. Reads attempt scores + code from CORAL's attempt JSONs
4. Converts to slime `Sample` objects with `reward = score`
5. Returns `RolloutFnTrainOutput` for slime's GRPO training

## Why slime > our hand-rolled GRPO

| Feature | Our GRPO | slime |
|---|---|---|
| KL penalty | ❌ | ✅ built-in |
| Replay buffer | ❌ | ✅ data buffer |
| Proper batching | ❌ (1 sample at a time) | ✅ Megatron |
| Weight sync | restart vLLM | ✅ SGLang hot reload |
| Gradient accumulation | ❌ | ✅ |
| Mixed precision training | manual | ✅ Megatron |

This should fix the catastrophic forgetting we saw with R2/R3.

## Usage

```bash
# Set env vars
export CORAL_TASK_YAML=examples/kernel_engineering/trimul/task_ttt.yaml
export CORAL_GPU_NODE=p5en-odcr-queue-dy-p5en48xlarge-28

# Run with slime
bash ttt/run_slime.sh --model Qwen/Qwen3-32B
```

## Dependencies

- slime (`pip install slime` or clone from github.com/THUDM/slime)
- CORAL (this repo)
- kiro-cli or OpenCode (agent runtime)
