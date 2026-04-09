#!/bin/bash
# CORAL TTT with Qwen3-72B: Node 1 = vLLM (TP=8), Node 2 = LoRA training
set -euo pipefail
JOBID=${1:?Usage: $0 <SLURM_JOB_ID>}
shift || true
SERVE_NODE=p5en-odcr-queue-dy-p5en48xlarge-9
TRAIN_NODE=p5en-odcr-queue-dy-p5en48xlarge-10
export PATH=/opt/slurm/bin:$HOME/.local/bin:$HOME/.npm-global/bin:$HOME/.opencode/bin:$PATH

echo "=== CORAL TTT: Qwen3-72B ==="
echo "Serve: $SERVE_NODE | Train: $TRAIN_NODE | Job: $JOBID"

# Step 1: Start vLLM on serve node (all 8 GPUs, TP=8)
echo "[$(date)] Starting vLLM on $SERVE_NODE..."
srun --jobid=$JOBID --nodelist=$SERVE_NODE --ntasks=1 --gpus=8 bash -c '
source /fsx/xuanj/coral-ttt-venv/bin/activate
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-72B --port 8000 --trust-remote-code \
  --tensor-parallel-size 8 --max-model-len 65536 --dtype bfloat16 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0
' > /fsx/xuanj/ttt-vllm.log 2>&1 &
VLLM_PID=$!

# Get serve node IP for cross-node access
SERVE_IP=$(scontrol show node $SERVE_NODE | grep NodeAddr | awk -F= '{print $2}' | awk '{print $1}')
echo "Serve node IP: $SERVE_IP"

# Wait for vLLM
for i in $(seq 1 180); do
  curl -s http://$SERVE_IP:8000/health > /dev/null 2>&1 && break
  sleep 2
done
if ! curl -s http://$SERVE_IP:8000/health > /dev/null 2>&1; then
  echo "vLLM failed:"; tail -30 /fsx/xuanj/ttt-vllm.log; kill $VLLM_PID 2>/dev/null; exit 1
fi
echo "[$(date)] vLLM ready at $SERVE_IP:8000"

# Update opencode.json to point to serve node
sed -i "s|http://localhost:8000|http://$SERVE_IP:8000|g" /fsx/xuanj/CORAL-xuan/examples/circle_packing/seed/opencode.json

# Step 2: Run TTT training on train node
echo "[$(date)] Starting TTT on $TRAIN_NODE..."
srun --jobid=$JOBID --nodelist=$TRAIN_NODE --ntasks=1 --gpus=2 bash -c "
export PATH=\$HOME/.local/bin:\$HOME/.npm-global/bin:\$HOME/.opencode/bin:\$PATH
source /fsx/xuanj/coral-ttt-venv/bin/activate
cd /fsx/xuanj/CORAL-xuan

python -m ttt.train \
  --task examples/circle_packing/task_ttt.yaml \
  --model Qwen/Qwen3-72B \
  --vllm-gpu 0 \
  --train-gpu 1 \
  --steps 20 \
  --evals-per-step 2 \
  --lr 1e-5 \
  --lora-rank 16 \
  --reload-every 5 \
  --checkpoint-dir /fsx/xuanj/ttt_checkpoints \
  $@
" 2>&1 | tee /fsx/xuanj/ttt-train.log

echo "[$(date)] Done. Cleaning up..."
kill $VLLM_PID 2>/dev/null
