#!/usr/bin/env bash
# Runs a training step inside WSL and logs to outputs/<name>.log. Usage: bash finetune/run_train.sh <name> [train.py args]
set -uo pipefail
cd "$(dirname "$0")/.."
NAME="$1"; shift
mkdir -p outputs
source "$HOME/ccp-ft-venv/bin/activate"
export PYTHONUNBUFFERED=1
python finetune/train.py "$@" > "outputs/$NAME.log" 2>&1
echo "EXIT_CODE=$?" >> "outputs/$NAME.log"
