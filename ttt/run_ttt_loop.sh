#!/bin/bash
# Full TTT loop: pipeline → GRPO train → pipeline with updated model
# Measures correctness rate before and after training
set -euo pipefail
JOBID=${1:?Usage: $0 <SLURM_JOB_ID>}
NODE=p5en-odcr-queue-dy-p5en48xlarge-9
export PATH=/opt/slurm/bin:$HOME/.local/bin:$PATH

echo "=== Full TTT Loop: Collect → Train → Evaluate ==="

# Phase 1: Collect samples (use existing pipeline results if available)
SAMPLES_DIR=/fsx/xuanj/ttt_pipeline_results
TASK=examples/kernel_engineering/trimul/task.yaml
MODEL=Qwen/Qwen3-32B
CKPT_DIR=/fsx/xuanj/ttt_grpo_checkpoint

echo "[$(date)] Phase 1: Checking existing samples..."
SAMPLE_COUNT=$(ls $SAMPLES_DIR/step_*_best.py 2>/dev/null | wc -l)
echo "Found $SAMPLE_COUNT samples"

if [ "$SAMPLE_COUNT" -lt 5 ]; then
    echo "Not enough samples. Run pipeline first:"
    echo "  bash ttt/run_pipeline.sh $JOBID"
    exit 1
fi

# Phase 2: GRPO training on node 10 (separate from vLLM on node 9)
echo "[$(date)] Phase 2: GRPO training on node 10..."
TRAIN_NODE=p5en-odcr-queue-dy-p5en48xlarge-10
srun --jobid=$JOBID --nodelist=$TRAIN_NODE --ntasks=1 --gpus=2 bash -c "
source /fsx/xuanj/coral-ttt-venv/bin/activate
cd /fsx/xuanj/CORAL-xuan
CUDA_VISIBLE_DEVICES=0 python -m ttt.grpo_train \
    --model $MODEL \
    --samples-dir $SAMPLES_DIR \
    --task $TASK \
    --output-dir $CKPT_DIR \
    --train-gpu 0 \
    --epochs 3 \
    --lr 1e-5
"
echo "[$(date)] Training done."

# Phase 3: Restart vLLM with trained model
echo "[$(date)] Phase 3: Restarting vLLM with trained model..."
# Kill old vLLM
ssh $NODE "pkill -9 -f vllm 2>/dev/null" 2>/dev/null || true
sleep 5

srun --jobid=$JOBID --nodelist=$NODE --ntasks=1 --gpus=8 bash -c "
source /fsx/xuanj/coral-ttt-venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  python -m vllm.entrypoints.openai.api_server \
  --model $CKPT_DIR --port 8000 --trust-remote-code \
  --tensor-parallel-size 4 --max-model-len 32768 --dtype bfloat16 \
  --host 0.0.0.0
" > /fsx/xuanj/ttt-vllm-trained.log 2>&1 &

SERVE_IP=$(scontrol show node $NODE | grep NodeAddr | awk -F= '{print $2}' | awk '{print $1}')
for i in $(seq 1 120); do
  curl -s http://$SERVE_IP:8000/health > /dev/null 2>&1 && break
  sleep 2
done
echo "[$(date)] Trained vLLM ready"

# Phase 4: Re-run pipeline with trained model to measure improvement
echo "[$(date)] Phase 4: Evaluating trained model..."
EVAL_DIR=/fsx/xuanj/ttt_trained_results
rm -rf $EVAL_DIR 2>/dev/null

srun --jobid=$JOBID --nodelist=$NODE --ntasks=1 --gpus=4 --overlap bash -c "
source /fsx/xuanj/coral-ttt-venv/bin/activate
cd /fsx/xuanj/CORAL-xuan
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m ttt.pipeline \
    --task $TASK \
    --model $CKPT_DIR \
    --vllm-url http://localhost:8000/v1 \
    --steps 10 --samples-per-step 4 \
    --workdir $EVAL_DIR
" 2>&1 | tee /fsx/xuanj/ttt-trained-eval.log

echo "[$(date)] === COMPARISON ==="
echo "Before training (base model):"
grep "★" /fsx/xuanj/ttt-pipeline-main.log 2>/dev/null | tail -3
echo "After training (GRPO model):"
grep "★" /fsx/xuanj/ttt-trained-eval.log 2>/dev/null | tail -3
echo "[$(date)] Done!"
