#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# Install dependencies if needed
uv sync --extra ttt

# Run training — forward all arguments to the trainer
# Example:
#   ./ttt/train.sh +task_yaml=examples/circle_packing/task.yaml
#   ./ttt/train.sh +task_yaml=examples/circle_packing/task.yaml +repeat=5000
exec uv run python -m ttt.trainer "$@"
