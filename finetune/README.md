# Fine-tuning (QLoRA on RTX 5060, WSL2)

1. One-time setup (PowerShell, project root):
   `wsl -d Ubuntu -- bash finetune/setup_wsl.sh`
2. Smoke test (10 steps):
   `wsl -d Ubuntu -- bash -lc "source ~/ccp-ft-venv/bin/activate && cd '/mnt/c/Users/Zulqernain Abbasi/Documents/CCP-Car-Assistant' && python finetune/train.py --max_steps 10"`
3. Full run: same command without `--max_steps 10` (about 3 epochs over ~1,170 examples).
4. Merge: same command with `--export_only`, which writes `outputs/merged`. Then convert to GGUF: `wsl -d Ubuntu -- bash finetune/export_gguf.sh` (writes `outputs/car-assistant-q8_0.gguf`).
5. Register in Ollama (Windows):
   `ollama pull qwen2.5:3b-instruct; .\.venv\Scripts\python -m scripts.make_modelfile; ollama create car-assistant -f Modelfile`

Outputs: `outputs/train_log.csv` (loss curve), `outputs/train_summary.json` (peak VRAM, steps).
