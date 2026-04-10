#!/bin/bash
# Launch CORAL TTT with slime
# slime handles: model serving (SGLang), GRPO training (Megatron), weight sync
# We provide: custom rollout function that runs CORAL agent
#
# Usage:
#   bash ttt/run_slime.sh --model Qwen/Qwen3-32B --task examples/kernel_engineering/trimul/task_ttt.yaml
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-32B}
TASK=${TASK:-examples/kernel_engineering/trimul/task_ttt.yaml}
GPU_NODE=${GPU_NODE:-""}
EVALS_PER_BATCH=${EVALS_PER_BATCH:-4}

export CORAL_TASK_YAML=$TASK
export CORAL_GPU_NODE=$GPU_NODE
export CORAL_EVALS_PER_BATCH=$EVALS_PER_BATCH

echo "=== CORAL TTT with slime ==="
echo "Model: $MODEL"
echo "Task: $TASK"
echo "GPU Node: $GPU_NODE"

python train.py \
    --model-path $MODEL \
    --custom-rollout-fn ttt.slime_rollout.generate_rollout_coral \
    --algorithm grpo \
    --grpo-group-size $EVALS_PER_BATCH \
    --kl-penalty-coeff 0.01 \
    --lr 2e-6 \
    --rollout-batch-size $EVALS_PER_BATCH \
    --train-batch-size $EVALS_PER_BATCH \
    --lora-rank 16 \
    --lora-alpha 32 \
    --save-interval 5 \
    --sglang-mem-fraction-static 0.8 \
    --tensor-model-parallel-size 4 \
    "$@"
