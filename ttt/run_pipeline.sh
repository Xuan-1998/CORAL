#!/bin/bash
# Run ThetaEvolve-style pipeline on p5en
# Node 9: vLLM (TP=4) + kernel eval (GPU)
# Usage: bash ttt/run_pipeline.sh <JOB_ID>
set -euo pipefail
JOBID=${1:?Usage: $0 <SLURM_JOB_ID>}
NODE=${NODE:-p5en-odcr-queue-dy-p5en48xlarge-9}
export PATH=/opt/slurm/bin:$HOME/.local/bin:$PATH

echo "=== ThetaEvolve Pipeline: TriMul Kernel on $NODE ==="

srun --jobid=$JOBID --nodelist=$NODE --ntasks=1 --gpus=8 bash -c '
export PATH=$HOME/.local/bin:$PATH
source /fsx/xuanj/coral-ttt-venv/bin/activate
cd /fsx/xuanj/CORAL-xuan

MODEL=Qwen/Qwen3-32B

# Start vLLM on GPUs 0-3
echo "[$(date)] Starting vLLM (TP=4)..."
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  python -m vllm.entrypoints.openai.api_server \
  --model $MODEL --port 8000 --trust-remote-code \
  --tensor-parallel-size 4 --max-model-len 32768 --dtype bfloat16 \
  --host 0.0.0.0 > /fsx/xuanj/ttt-vllm.log 2>&1 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null" EXIT

for i in $(seq 1 120); do
  curl -s http://localhost:8000/health > /dev/null 2>&1 && break
  sleep 2
done
curl -s http://localhost:8000/health > /dev/null 2>&1 || { echo "vLLM failed"; tail -20 /fsx/xuanj/ttt-vllm.log; exit 1; }
echo "[$(date)] vLLM ready"

# Run pipeline (kernel eval uses remaining GPUs)
echo "[$(date)] Starting pipeline..."
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m ttt.pipeline \
  --task examples/kernel_engineering/trimul/task.yaml \
  --model $MODEL \
  --vllm-url http://localhost:8000/v1 \
  --steps 20 \
  --samples-per-step 4 \
  --temperature 0.8 \
  --workdir /fsx/xuanj/ttt_pipeline_results \
  2>&1 | tee /fsx/xuanj/ttt-pipeline.log

echo "[$(date)] Done!"
'
