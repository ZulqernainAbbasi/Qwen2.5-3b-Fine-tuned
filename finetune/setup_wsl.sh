#!/usr/bin/env bash
# One-time setup of the fine-tuning environment inside WSL2 (Ubuntu).
# Usage (from Windows PowerShell):  wsl -d Ubuntu -- bash finetune/setup_wsl.sh
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
VENV="$HOME/ccp-ft-venv"
[ -d "$VENV" ] || uv venv -p 3.12 "$VENV"
source "$VENV/bin/activate"
# RTX 50-series (Blackwell, sm_120) needs a CUDA 12.8+ PyTorch build
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install unsloth unsloth_zoo trl datasets pandas
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
echo SETUP_DONE
