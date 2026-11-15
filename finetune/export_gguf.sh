#!/usr/bin/env bash
# Convert outputs/merged (16-bit safetensors) to outputs/car-assistant-q8_0.gguf with llama.cpp's converter.
# Usage: wsl -d Ubuntu -- bash finetune/export_gguf.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/ccp-ft-venv/bin/activate"
LCPP="$HOME/llama.cpp"
[ -d "$LCPP" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LCPP"
export PATH="$HOME/.local/bin:$PATH"
uv pip install -q gguf sentencepiece protobuf
python "$LCPP/convert_hf_to_gguf.py" outputs/merged --outtype q8_0 --outfile outputs/car-assistant-q8_0.gguf
ls -la outputs/car-assistant-q8_0.gguf
echo GGUF_DONE
