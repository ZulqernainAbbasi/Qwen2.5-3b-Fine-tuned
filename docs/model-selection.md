# Model Selection (Requirement 1)

We compared three small open-weight instruction models that can be fine-tuned with QLoRA on an 8 GB laptop GPU (RTX 5060) and served locally with Ollama.

| | Qwen2.5-3B-Instruct (chosen) | Llama-3.2-3B-Instruct | Phi-3.5-mini-instruct |
|---|---|---|---|
| Parameters | 3.09B | 3.2B | 3.8B |
| Context | 32K | 128K | 128K |
| Licence | Qwen Research License (research/academic use; commercial use restricted) | Llama 3.2 Community License | MIT |
| QLoRA 4-bit VRAM (seq 2048, batch 2) | fits in 8 GB | fits in 8 GB | tight in 8 GB |
| Native tool calling in Ollama | yes | yes | limited |
| Roman Urdu / multilingual | good (multilingual pre-training) | fair | fair |

**Why Qwen2.5-3B:** it is the smallest model in this group with reliable tool calling (needed by the LangGraph agent), strong multilingual ability for English and Roman Urdu users, and comfortable QLoRA training on 8 GB VRAM. Its research licence suits this academic project.

**If this became a commercial product:** switch to Qwen2.5-1.5B-Instruct or Qwen2.5-7B-Instruct (both Apache 2.0) or Phi-3.5-mini (MIT). The training scripts only need a different `BASE_MODEL` in `finetune/train.py`.

Source for the Qwen licence split: Qwen2.5 release blog (qwenlm.github.io/blog/qwen2.5-llm).
