"""Week 3 / Day 12 (pod): sample N completions per pref-prompt from the SFT model.

On-policy preference data: we draw several DIFFERENT samples from our own SFT
model (temperature sampling), so a judge can later rank them into chosen/rejected.
Greedy would give identical samples — useless for preferences — so we sample.

Writes {prompt, samples: [s1..sN]} JSONL. Runs on the pod (GPU). Optional HF push.

  python -m src.data.gen_pref_samples --n 4 --push-repo sarthaksenapati/khichdi-pref
  python -m src.data.gen_pref_samples --limit 50        # quick subset
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
ADAPTER = "sarthaksenapati/qwen1.5b-hinglish-sft-v2"   # the SFT policy we sample from
DATASET = "sarthaksenapati/khichdi-sft"
CHATML = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pref/pref_samples.jsonl")
    ap.add_argument("--n", type=int, default=4, help="samples per prompt")
    ap.add_argument("--batch", type=int, default=16, help="prompts per forward pass")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--push-repo", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(ADAPTER)
    tok.chat_template = CHATML
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
                                                device_map={"": 0}, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, ADAPTER).eval()

    prompts = [r["instruction"] for r in load_dataset(DATASET, "pref_prompts", split="train")]
    if args.limit:
        prompts = prompts[:args.limit]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(0, len(prompts), args.batch):
            chunk = prompts[i:i + args.batch]
            texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                             add_generation_prompt=True, tokenize=False) for p in chunk]
            enc = tok(texts, return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                                     temperature=args.temperature, top_p=args.top_p,
                                     num_return_sequences=args.n,        # N samples per prompt
                                     repetition_penalty=1.2, no_repeat_ngram_size=3,
                                     eos_token_id=im_end, pad_token_id=tok.pad_token_id)
            gen = out[:, enc.input_ids.shape[1]:]                         # (batch*N, gen_len)
            decoded = [tok.decode(g, skip_special_tokens=True).strip() for g in gen]
            for j, p in enumerate(chunk):
                samples = decoded[j * args.n:(j + 1) * args.n]            # this prompt's N samples
                f.write(json.dumps({"prompt": p, "samples": samples}, ensure_ascii=False) + "\n")
                written += 1
            print(f"{min(i + args.batch, len(prompts))}/{len(prompts)}")
    print("wrote", args.out, "rows:", written)

    if args.push_repo:
        from huggingface_hub import HfApi, create_repo
        create_repo(args.push_repo, repo_type="dataset", exist_ok=True)
        HfApi().upload_file(path_or_fileobj=args.out, path_in_repo="pref_samples.jsonl",
                            repo_id=args.push_repo, repo_type="dataset")
        print("uploaded to", args.push_repo)


if __name__ == "__main__":
    main()
