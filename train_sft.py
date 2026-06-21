"""QLoRA SFT of Qwen2.5-1.5B-Base on the Khichdi Hinglish dataset.

Self-contained. MANUAL chat formatting + loss masking (explicit, version-safe).
transformers 5.x Trainer + PEFT LoRA + bitsandbytes 4-bit (QLoRA). Runs on the pod.

v2: MAXLEN 2048, force-append <|im_end|> on truncation (stop-token fix), LoRA rank 32.

  python train_sft.py --inspect   # show one formatted example + its loss mask, then exit
  python train_sft.py --smoke     # tiny run (200 ex, 20 steps) to validate the pipeline
  python train_sft.py             # full run (logs to W&B)
"""
from __future__ import annotations
import argparse
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE = "Qwen/Qwen2.5-1.5B"
DATASET = "sarthaksenapati/khichdi-sft"
MAXLEN = 2048

# Our own clean ChatML template (no injected system prompt) -> full control, train==inference.
CHATML = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def get_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.chat_template = CHATML
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def make_tokenize_fn(tok):
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    def fn(ex):
        user = [{"role": "user", "content": ex["instruction"]}]
        full = user + [{"role": "assistant", "content": ex["response"]}]
        prompt_text = tok.apply_chat_template(user, add_generation_prompt=True, tokenize=False)
        full_text = tok.apply_chat_template(full, add_generation_prompt=False, tokenize=False)
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
        if len(full_ids) > MAXLEN:
            full_ids = full_ids[:MAXLEN - 1] + [im_end]   # GUARANTEE the row ends with the stop token
        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(full_ids))):
            labels[i] = -100             # mask prompt; train only on the assistant turn (+ <|im_end|>)
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}
    return fn


def inspect(tok):
    ex = load_dataset(DATASET, "train", split="train")[0]
    enc = make_tokenize_fn(tok)(ex)
    ids, labels = enc["input_ids"], enc["labels"]
    trained = [i for i, l in zip(ids, labels) if l != -100]
    print("=== FULL TEXT ===\n" + tok.decode(ids))
    print("\n=== TOKENS TRAINED ON (label != -100) ===\n" + tok.decode(trained))
    print(f"\ntotal={len(ids)}  masked_prompt={labels.count(-100)}  trained_assistant={len(trained)}")
    print("trains on the stop token (last label != -100):", labels[-1] != -100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="/workspace/sft-qwen1.5b-hinglish-v2")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    args = ap.parse_args()
    set_seed(42)

    tok = get_tokenizer()
    if args.inspect:
        inspect(tok); return

    ds = load_dataset(DATASET, "train", split="train")
    if args.smoke:
        ds = ds.select(range(200))
    split = ds.train_test_split(test_size=0.05, seed=42)
    tok_fn = make_tokenize_fn(tok)
    cols = ds.column_names
    train = split["train"].map(tok_fn, remove_columns=cols)
    val = split["test"].map(tok_fn, remove_columns=cols)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
                                                 device_map={"": 0}, dtype=torch.bfloat16)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=(1 if args.smoke else args.epochs),
        max_steps=(20 if args.smoke else -1),
        lr_scheduler_type="cosine",
        warmup_steps=30,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps", eval_steps=100,
        save_strategy="steps", save_steps=200, save_total_limit=2,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=("none" if args.smoke else "wandb"),
        run_name="sft-qwen1.5b-hinglish-v2",
        seed=42,
    )
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, return_tensors="pt")
    trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=val,
                      data_collator=collator)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
