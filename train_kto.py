"""Day 17: KTO training, using our from-scratch loss (src/dpo/loss.py).

Same merge-and-toggle setup as DPO (adapter ENABLED = policy, DISABLED = frozen
SFT reference). The difference is the DATA and the LOSS:
  - data: unpaired {prompt, completion, label} (desirable / undesirable)
  - loss: per-example, judged against a reference point z0 (a detached, >=0 KL
    estimate). Here z0 is the batch-mean log-ratio (a documented simplification of
    KTO's mismatched-pair KL estimate).

  python train_kto.py --epochs 3 --bs 4 --grad-accum 8 --beta 0.1 --lr 1e-5
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.dpo.loss import sequence_logps, kto_loss

BASE = "Qwen/Qwen2.5-1.5B"
SFT_ADAPTER = "sarthaksenapati/qwen1.5b-hinglish-sft-v2"
DATA_REPO = "sarthaksenapati/khichdi-pref"
CHATML = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def encode(tok, im_end, prompt, completion, maxlen):
    ptext = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                    add_generation_prompt=True, tokenize=False)
    pid = tok(ptext, add_special_tokens=False)["input_ids"]
    rid = tok(completion, add_special_tokens=False)["input_ids"] + [im_end]
    ids = (pid + rid)[:maxlen]
    mask = ([0] * len(pid) + [1] * len(rid))[:maxlen]
    return ids, mask


class KTODS(Dataset):
    def __init__(self, rows, tok, im_end, maxlen):
        self.rows, self.tok, self.im_end, self.maxlen = rows, tok, im_end, maxlen

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ids, mask = encode(self.tok, self.im_end, r["prompt"], r["completion"], self.maxlen)
        return ids, mask, bool(r["label"])


def collate(batch):
    return [(i, m) for i, m, _ in batch], torch.tensor([lab for _, _, lab in batch])


def pad_batch(seqs, pad_id, device):
    T = max(len(ids) for ids, _ in seqs)
    ids_b, attn_b, mask_b = [], [], []
    for ids, mask in seqs:
        n = T - len(ids)
        ids_b.append(ids + [pad_id] * n)
        attn_b.append([1] * len(ids) + [0] * n)
        mask_b.append(mask + [0] * n)
    t = lambda x: torch.tensor(x, device=device)
    return t(ids_b), t(attn_b), t(mask_b).float()


def logps(model, ids, attn, loss_mask):
    out = model(input_ids=ids, attention_mask=attn).logits
    return sequence_logps(out, ids, loss_mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/kto-qwen1.5b-hinglish")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--maxlen", type=int, default=1024)
    args = ap.parse_args()
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained(SFT_ADAPTER)
    tok.chat_template = CHATML
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": 0})
    merged = PeftModel.from_pretrained(base, SFT_ADAPTER).merge_and_unload()
    lora = LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(merged, lora)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    path = hf_hub_download(DATA_REPO, "kto_data.jsonl", repo_type="dataset")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    print(f"loaded {len(rows)} KTO examples "
          f"({sum(r['label'] for r in rows)} desirable / {sum(not r['label'] for r in rows)} undesirable)")
    dl = DataLoader(KTODS(rows, tok, im_end, args.maxlen), batch_size=args.bs,
                    shuffle=True, collate_fn=collate)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    step = 0
    for epoch in range(args.epochs):
        for bi, (seqs, labels) in enumerate(dl):
            ids, attn, lm = pad_batch(seqs, tok.pad_token_id, dev)
            labels = labels.to(dev)

            p_lp = logps(model, ids, attn, lm)                 # policy   (grad)
            with torch.no_grad(), model.disable_adapter():
                r_lp = logps(model, ids, attn, lm)             # reference (frozen SFT)

            des = labels.bool()
            des_loss, undes_loss, des_rew, undes_rew, kl = kto_loss(
                p_lp[des], r_lp[des], p_lp[~des], r_lp[~des], beta=args.beta)
            loss = torch.cat([des_loss, undes_loss]).mean()
            (loss / args.grad_accum).backward()

            if (bi + 1) % args.grad_accum == 0:
                opt.step(); opt.zero_grad(); step += 1
                dr = des_rew.mean().item() if des_rew.numel() else float("nan")
                ur = undes_rew.mean().item() if undes_rew.numel() else float("nan")
                print(f"epoch {epoch} step {step} | loss {loss.item():.4f} "
                      f"| z0 {kl.item():.3f} | reward des {dr:.3f} undes {ur:.3f}")

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("saved KTO adapter to", args.out)


if __name__ == "__main__":
    main()
