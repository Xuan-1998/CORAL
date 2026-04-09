# ttt — Test-Time Training for CORAL

Minimal TTT implementation: vLLM + CORAL + GRPO. No rllm dependency.

## Architecture

```
GPU 0: vLLM serves Qwen3-8B (OpenAI-compatible API, port 8000)
         ↑
CORAL gateway (litellm proxy, port 4000) — logs all request/response to JSONL
         ↑
OpenCode agent — autonomous coding agent solving the task
         ↓
CORAL grader — evaluates agent's code, returns score
         ↓
GPU 1: LoRA model — GRPO update using (traces, rewards)
         ↓
Merge LoRA → restart vLLM with updated weights → repeat
```

## Usage

```bash
# On p5en cluster
export PATH=/opt/slurm/bin:$PATH

# Allocate 2 nodes (need at least 2 GPUs on one node)
salloc --nodelist=p5en-odcr-queue-dy-p5en48xlarge-7,p5en-odcr-queue-dy-p5en48xlarge-8 \
  --gpus-per-node=8 --time=4:00:00 --no-shell

# Run TTT
bash ttt/run.sh <JOB_ID>

# Or directly
python -m ttt.train \
  --task examples/circle_packing/task_ttt.yaml \
  --model Qwen/Qwen3-8B \
  --steps 20 \
  --evals-per-step 3
```

## How it works

Each training step:
1. Start/resume CORAL agent
2. Agent generates code, submits for evaluation (N evals per step)
3. Stop agent, collect gateway traces (LLM request/response pairs)
4. Compute reward = score_improvement over parent attempt
5. GRPO update: weight loss by normalized advantage
6. Every K steps: merge LoRA, save checkpoint, restart vLLM

## Key files

- `train.py` — Main trainer (vLLM management, CORAL orchestration, GRPO)
- `run.sh` — Launch script for p5en cluster
- `../examples/circle_packing/task_ttt.yaml` — Task config for TTT
- `../examples/circle_packing/litellm_vllm.yaml` — Gateway → vLLM routing
