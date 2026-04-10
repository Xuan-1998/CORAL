#!/bin/bash
# Run slime TTT in Docker container
# Requires: 2 nodes with GPUs, Docker/enroot available
# Usage: bash ttt/run_docker_slime.sh <SLURM_JOB_ID>
set -euo pipefail

JOBID=${1:?Usage: $0 <SLURM_JOB_ID>}
export PATH=/opt/slurm/bin:$HOME/.local/bin:$PATH

# Build slime Docker image (first time only)
IMAGE=slimerl/slime:coral-ttt
if ! docker images | grep -q "slimerl/slime.*coral-ttt"; then
    echo "Building slime Docker image..."
    cd /fsx/xuanj/slime-framework
    docker build -t $IMAGE -f docker/Dockerfile .
fi

# Get node IPs
NODES=$(scontrol show job $JOBID | grep NodeList | awk -F= '{print $2}')
N1=$(echo $NODES | tr ',' '\n' | head -1)
N2=$(echo $NODES | tr ',' '\n' | tail -1)
N1_IP=$(scontrol show node $N1 | grep NodeAddr | awk -F= '{print $2}' | awk '{print $1}')

echo "Nodes: $N1 ($N1_IP), $N2"

# Run slime in Docker with GPU access
srun --jobid=$JOBID --nodes=2 --ntasks-per-node=1 --gpus-per-node=8 \
    --container-image=$IMAGE \
    --container-mounts=/fsx/xuanj:/fsx/xuanj \
    bash -c "
export PYTHONPATH=/fsx/xuanj/CORAL-xuan:\$PYTHONPATH
export CORAL_TASK_YAML=/fsx/xuanj/CORAL-xuan/examples/kernel_engineering/trimul/task_ttt.yaml
export CORAL_GPU_NODE=$N1
export CUDA_DEVICE_MAX_CONNECTIONS=1

cd /fsx/xuanj/slime-framework
source /fsx/xuanj/CORAL-xuan/ttt/run_slime_ttt.sh
"
