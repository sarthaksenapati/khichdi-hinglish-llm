"""Day 9: generate base & SFT completions on held-out eval prompts -> paired JSONL.

For each held-out eval prompt, generate from BASE (adapter disabled) and SFT
(adapter enabled) with IDENTICAL decoding, and write {prompt, a: base, b: sft}
for the win-rate judge. Runs on the pod (GPU). Optionally uploads to HF.

Batched generation: prompts are run in groups (left-padded) so the GPU is fed
many sequences per forward pass instead of one at a time -> far higher util.

  python eval_generate.py --push-repo sarthaksenapati/khichdi-eval
  python eval_generate.py --batch 32 --limit 50
"""
from __future__ import annotations
import argparse
import json
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B"
ADAPTER = "sarthaksenapati/qwen1.5b-hinglish-sft-v2"   # adapter pulled from HF
DATASET = "sarthaksenapati/khichdi-sft"
CHATML = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/eval/base_vs_sft.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--push-repo", default=None, help="HF dataset repo to upload result to")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(ADAPTER)
    tok.chat_template = CHATML
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # decoder-only generation needs LEFT padding
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
                                                device_map={"": 0}, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, ADAPTER).eval()

    prompts = [r["instruction"] for r in load_dataset(DATASET, "eval_prompts", split="train")]
    if args.limit:
        prompts = prompts[:args.limit]

    def gen_batch(ps):  # IDENTICAL decoding for base and SFT — only the adapter differs
        outs = []
        for i in range(0, len(ps), args.batch):
            chunk = ps[i:i + args.batch]
            texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                             add_generation_prompt=True, tokenize=False) for p in chunk]
            enc = tok(texts, return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     repetition_penalty=1.2, no_repeat_ngram_size=3,
                                     eos_token_id=im_end, pad_token_id=tok.pad_token_id)
            gen_only = out[:, enc.input_ids.shape[1]:]          # left-pad => same input width for all rows
            outs.extend(tok.decode(g, skip_special_tokens=True).strip() for g in gen_only)
            print(f"{min(i + args.batch, len(ps))}/{len(ps)}")
        return outs

    print("generating BASE ...")
    with model.disable_adapter():
        base_outs = gen_batch(prompts)
    print("generating SFT ...")
    sft_outs = gen_batch(prompts)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p, a, b in zip(prompts, base_outs, sft_outs):
            f.write(json.dumps({"prompt": p, "a": a, "b": b}, ensure_ascii=False) + "\n")
    print("wrote", args.out)

    if args.push_repo:
        from huggingface_hub import HfApi, create_repo
        create_repo(args.push_repo, repo_type="dataset", exist_ok=True)
        HfApi().upload_file(path_or_fileobj=args.out, path_in_repo="base_vs_sft.jsonl",
                            repo_id=args.push_repo, repo_type="dataset")
        print("uploaded to", args.push_repo)


if __name__ == "__main__":
    main()
