"""QLoRA fine-tuning of Qwen2.5-3B-Instruct with Unsloth (run inside WSL2).

Smoke test:  python finetune/train.py --max_steps 10
Full run:    python finetune/train.py
Export:      python finetune/train.py --export_only   (merges LoRA -> outputs/merged, 16-bit safetensors)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs"
BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct"
MAX_SEQ = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--export_only", action="store_true")
    args = ap.parse_args()

    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only

    if args.export_only:
        model, tok = FastLanguageModel.from_pretrained(str(OUT / "lora"), max_seq_length=MAX_SEQ, load_in_4bit=True)
        model.save_pretrained_merged(str(OUT / "merged"), tok, save_method="merged_16bit")
        print("merged model saved to", OUT / "merged")
        return

    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tok = FastLanguageModel.from_pretrained(BASE_MODEL, max_seq_length=MAX_SEQ, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=42)
    tok = get_chat_template(tok, chat_template="qwen-2.5")

    data_dir = ROOT / "data" / "finetune"
    ds = load_dataset("json", data_files={"train": str(data_dir / "train.jsonl"), "eval": str(data_dir / "eval.jsonl")})
    ds = ds.map(lambda b: {"text": [tok.apply_chat_template(m, tokenize=False) for m in b["messages"]]},
                batched=True, remove_columns=["messages", "type"])

    cfg = SFTConfig(
        output_dir=str(OUT / "ckpt"), dataset_text_field="text", max_length=MAX_SEQ,
        per_device_train_batch_size=2, gradient_accumulation_steps=4, learning_rate=2e-4,
        num_train_epochs=args.epochs, max_steps=args.max_steps, warmup_steps=10, lr_scheduler_type="linear",
        logging_steps=5, eval_strategy="steps" if args.max_steps > 0 else "epoch",
        eval_steps=args.max_steps if args.max_steps > 0 else None,
        save_strategy="no", optim="adamw_8bit", weight_decay=0.01, seed=42, report_to="none",
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported())
    trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=ds["train"], eval_dataset=ds["eval"], args=cfg)
    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                      response_part="<|im_start|>assistant\n")
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3

    OUT.mkdir(exist_ok=True)
    with open(OUT / "train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss", "eval_loss"])
        w.writeheader()
        for h in trainer.state.log_history:
            if "loss" in h or "eval_loss" in h:
                w.writerow({"step": h.get("step"), "loss": h.get("loss"), "eval_loss": h.get("eval_loss")})
    (OUT / "train_summary.json").write_text(json.dumps({"peak_vram_gb": round(peak, 2),
                                                        "steps": trainer.state.global_step}, indent=2))
    print(f"peak VRAM {peak:.2f} GB, steps {trainer.state.global_step}")
    if args.max_steps < 0:
        model.save_pretrained(str(OUT / "lora"))
        tok.save_pretrained(str(OUT / "lora"))
        print("LoRA adapter saved to", OUT / "lora")


if __name__ == "__main__":
    main()
