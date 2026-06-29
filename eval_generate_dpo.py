"""Day 15: generate SFT vs DPO completions on the held-out eval prompts.

The DPO LoRA sits on top of the MERGED SFT model (M = base + SFT adapter). So:
  adapter DISABLED -> M               = SFT model
  adapter ENABLED  -> M + DPO LoRA    = DPO model
Toggling only the DPO adapter (identical decoding) is the fairest A/B.

Writes {prompt, a: sft, b: dpo} for the win-rate judge, and also saves a fully
merged DPO model locally for the MMLU run. Runs on the pod (GPU).

  python eval_generate_dpo.py --push-repo sarthaksenapati/khichdi-eval
"""
from __future__ import annotations
import argparse
import json
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B"
SFT_ADAPTER = "sarthaksenapati/qwen1.5b-hinglish-sft-v2"
DPO_ADAPTER = "sarthaksenapati/qwen1.5b-hinglish-dpo"
DATASET = "sarthaksenapati/khichdi-sft"
CHATML = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=DPO_ADAPTER, help="aligned adapter on top of merged SFT")
    ap.add_argument("--tag", default="dpo", help="name for outputs: dpo / ipo / kto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--merged-out", default=None, help="where to save the merged model for MMLU")
    ap.add_argument("--push-repo", default=None)
    args = ap.parse_args()
    args.out = args.out or f"data/eval/{args.tag}_vs_sft.jsonl"
    args.merged_out = args.merged_out or f"/workspace/{args.tag}-merged"

    tok = AutoTokenizer.from_pretrained(SFT_ADAPTER)
    tok.chat_template = CHATML
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    # base + SFT -> merge to M; then DPO LoRA on top of M
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": 0})
    merged_sft = PeftModel.from_pretrained(base, SFT_ADAPTER).merge_and_unload()
    model = PeftModel.from_pretrained(merged_sft, args.adapter).eval()

    prompts = [r["instruction"] for r in load_dataset(DATASET, "eval_prompts", split="train")]
    if args.limit:
        prompts = prompts[:args.limit]

    def gen_batch(ps):  # identical decoding; only the DPO adapter differs
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
            gen_only = out[:, enc.input_ids.shape[1]:]
            outs.extend(tok.decode(g, skip_special_tokens=True).strip() for g in gen_only)
            print(f"{min(i + args.batch, len(ps))}/{len(ps)}")
        return outs

    print("generating SFT (adapter disabled) ...")
    with model.disable_adapter():
        sft_outs = gen_batch(prompts)
    print(f"generating {args.tag.upper()} (adapter enabled) ...")
    aligned_outs = gen_batch(prompts)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p, a, b in zip(prompts, sft_outs, aligned_outs):
            f.write(json.dumps({"prompt": p, "a": a, "b": b}, ensure_ascii=False) + "\n")
    print("wrote", args.out)

    if args.push_repo:
        from huggingface_hub import HfApi, create_repo
        create_repo(args.push_repo, repo_type="dataset", exist_ok=True)
        HfApi().upload_file(path_or_fileobj=args.out, path_in_repo=os.path.basename(args.out),
                            repo_id=args.push_repo, repo_type="dataset")
        print("uploaded to", args.push_repo)

    # save fully merged model for the MMLU run (lm_eval pretrained=<this path>)
    print(f"merging {args.tag.upper()} adapter and saving for MMLU ...")
    merged = model.merge_and_unload()
    merged.save_pretrained(args.merged_out)
    tok.save_pretrained(args.merged_out)
    print(f"saved merged {args.tag.upper()} model to", args.merged_out)


if __name__ == "__main__":
    main()
