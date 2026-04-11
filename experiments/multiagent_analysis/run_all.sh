#!/usr/bin/env bash
# Run all ablation experiments for multiagent gain source analysis.
# Designed for b300 cluster (CPU-only tasks).
#
# Usage: ./run_all.sh [--trials 4] [--hours 3]

set -euo pipefail
cd "$(dirname "$0")/../.."  # CORAL root

TRIALS=${1:-4}
HOURS=${2:-3}
CONFIGS_DIR="experiments/multiagent_analysis/configs"

echo "=== Multiagent Gain Source Analysis ==="
echo "Trials: $TRIALS, Hours per run: $HOURS"
echo ""

run_experiment() {
    local config="$1"
    local name="$2"
    local trial="$3"

    echo "[$(date)] Starting: $name (trial $trial)"
    timeout "${HOURS}h" uv run coral start -c "$config" \
        workspace.run_dir="./results/ablation/${name}/trial_${trial}" \
        2>&1 | tee "results/ablation/${name}/trial_${trial}.log" || true
    echo "[$(date)] Finished: $name (trial $trial)"
}

mkdir -p results/ablation

# ── Exp A: Sampling efficiency ──
echo "=== Exp A: 1-agent baseline (×4 independent) ==="
for t in $(seq 1 $TRIALS); do
    run_experiment "$CONFIGS_DIR/exp_a_1agent.yaml" "1agent_baseline" "$t"
done

echo "=== Exp A: 4-agent co-evolution ==="
for t in $(seq 1 $TRIALS); do
    run_experiment "$CONFIGS_DIR/exp_a_4agent_coevol.yaml" "4agent_coevol" "$t"
done

# ── Exp C: Knowledge ablation ──
echo "=== Exp C: 4-agent attempts-only ==="
for t in $(seq 1 $TRIALS); do
    run_experiment "$CONFIGS_DIR/exp_c_attempts_only.yaml" "4agent_attempts_only" "$t"
done

echo "=== Exp C: 4-agent no sharing ==="
for t in $(seq 1 $TRIALS); do
    run_experiment "$CONFIGS_DIR/exp_c_no_sharing.yaml" "4agent_no_sharing" "$t"
done

# ── Analysis ──
echo ""
echo "=== Running analysis ==="
uv run python experiments/multiagent_analysis/analyze_gains.py \
    --results-dir ./results/ablation

echo ""
echo "=== All experiments complete ==="
