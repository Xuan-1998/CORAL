#!/bin/bash
export PYTHONPATH=/fsx/xuanj/megatron-lm-full:/fsx/xuanj/CORAL-xuan:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
set -ex

export PYTHONUNBUFFERED=1
export RAY_ADDRESS="auto"

SCRIPT_DIR=/fsx/xuanj/slime-framework/scripts
source ${SCRIPT_DIR}/models/qwen3-32B.sh

# Use distilled checkpoint if available, otherwise base model
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-32B}
ROLLOUT_FN=${ROLLOUT_FN:-ttt.slime_rollout.generate_rollout_coral}

# CORAL config
export CORAL_TASK_YAML=/fsx/xuanj/CORAL-xuan/examples/kernel_engineering/trimul/task_ttt.yaml
export CORAL_GPU_NODE=p5en-odcr-queue-dy-p5en48xlarge-22
export CORAL_EVALS_PER_BATCH=4
export PYTHONPATH=/fsx/xuanj/CORAL-xuan:$PYTHONPATH

cd /fsx/xuanj/slime-framework

CKPT_ARGS=(
    --hf-checkpoint $MODEL_PATH
    --save /fsx/xuanj/slime_checkpoints/
    --save-interval 5
)

ROLLOUT_ARGS=(
    --rollout-function-path $ROLLOUT_FN
    --prompt-data /fsx/xuanj/kernel_prompts.jsonl
    --input-key prompt
    --label-key label
    --rollout-batch-size 4
    --max-new-tokens 4096
    --temperature 0.8
)

TRAIN_ARGS=(
    --algorithm grpo
    --grpo-group-size 4
    --kl-penalty-coeff 0.0
    --lr 2e-6
    --train-iters 50 --num-epoch 10 --num-rollout 100
    --micro-batch-size 1
    --global-batch-size 4
    --seq-length 4096
    --max-position-embeddings 32768
    --lora-rank 16
    --lora-alpha 32
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 1
    --num-training-nodes 1
    --num-rollout-nodes 1 --rollout-num-gpus 8 --actor-num-nodes 1 --actor-num-gpus-per-node 8
)

python train.py \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $MODEL_PATH \
    --bf16 \
    --no-gradient-accumulation-fusion --no-rope-fusion \
    2>&1 | tee /fsx/xuanj/slime_ttt.log
