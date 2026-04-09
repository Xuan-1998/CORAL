#!/bin/bash
# Run CORAL TTT on p5en cluster
# Usage: bash ttt/run.sh <SLURM_JOB_ID> [--model Qwen/Qwen3-8B] [--steps 20]
set -euo pipefail

JOBID=${1:?Usage: $0 <SLURM_JOB_ID> [extra args for train.py]}
shift
NODE=${NODE:-p5en-odcr-queue-dy-p5en48xlarge-7}
export PATH=/opt/slurm/bin:$HOME/.local/bin:$HOME/.npm-global/bin:$HOME/.opencode/bin:$PATH

echo "=== CORAL TTT on $NODE (job $JOBID) ==="
echo "Extra args: $@"

srun --jobid=$JOBID --nodelist=$NODE --ntasks=1 --gpus=4 bash -c "
export PATH=\$HOME/.local/bin:\$HOME/.npm-global/bin:\$HOME/.opencode/bin:\$PATH
source /fsx/xuanj/coral-ttt-venv/bin/activate
cd $(pwd)

python -m ttt.train \
  --task examples/circle_packing/task.yaml \
  --model Qwen/Qwen3-8B \
  --vllm-gpu 0 \
  --train-gpu 1 \
  --steps 20 \
  --evals-per-step 3 \
  --lr 1e-5 \
  --lora-rank 16 \
  --reload-every 5 \
  --checkpoint-dir /fsx/xuanj/ttt_checkpoints \
  $@
"
