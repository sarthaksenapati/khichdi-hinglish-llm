"""Day 14: DPO training, using our from-scratch loss (src/dpo/loss.py).

ONE model serves as both policy and reference: we merge the SFT adapter into the
base (-> model M = our SFT), then add a FRESH LoRA adapter for DPO.
  adapter ENABLED  -> policy   (M + DPO-LoRA, trainable)
  adapter DISABLED -> reference (M alone, frozen)  via model.disable_adapter()

Data: preference pairs {prompt, chosen, rejected} from HF (khichdi-pref).
Out:  a DPO LoRA adapter saved to --out.

  python train_dpo.py --epochs 1 --bs 2 --grad-accum 8 --beta 0.1 --lr 5e-6
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
from src.dpo.loss import sequence_logps, dpo_loss

BASE = "Qwen/Qwen2.5-1.5B"
SFT_ADAPTER = "sarthaksenapati/qwen1.5b-hinglish-sft-v2"
PAIRS_REPO = "sarthaksenapati/khichdi-pref"
CHATML = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def encode(tok, im_end, prompt, response, maxlen):
    """prompt+response token ids and a loss_mask that is 1 only on the response (+stop)."""
    ptext = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                    add_generation_prompt=True, tokenize=False)
    pid = tok(ptext, add_special_tokens=False)["input_ids"]
    rid = tok(response, add_special_tokens=False)["input_ids"] + [im_end]
    ids = (pid + rid)[:maxlen]
    mask = ([0] * len(pid) + [1] * len(rid))[:maxlen]
    return ids, mask


class PairDS(Dataset):
    def __init__(self, rows, tok, im_end, maxlen):
        self.rows, self.tok, self.im_end, self.maxlen = rows, tok, im_end, maxlen

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return (encode(self.tok, self.im_end, r["prompt"], r["chosen"], self.maxlen),
                encode(self.tok, self.im_end, r["prompt"], r["rejected"], self.maxlen))


def pad_batch(seqs, pad_id, device):
    """seqs: list of (ids, mask) -> right-padded ids, attn, loss_mask tensors."""
    T = max(len(ids) for ids, _ in seqs)
    ids_b, attn_b, mask_b = [], [], []
    for ids, mask in seqs:
        n = T - len(ids)
        ids_b.append(ids + [pad_id] * n)
        attn_b.append([1] * len(ids) + [0] * n)
        mask_b.append(mask + [0] * n)
    t = lambda x: torch.tensor(x, device=device)
    return t(ids_b), t(attn_b), t(mask_b).float()


def collate(batch):
    return [c for c, _ in batch], [j for _, j in batch]


def logps(model, ids, attn, loss_mask):
    out = model(input_ids=ids, attention_mask=attn).logits
    return sequence_logps(out, ids, loss_mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/dpo-qwen1.5b-hinglish")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
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

    # base + SFT adapter -> merge into M (our SFT model, bf16)
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": 0})
    merged = PeftModel.from_pretrained(base, SFT_ADAPTER).merge_and_unload()
    # fresh DPO LoRA on top of M
    lora = LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(merged, lora)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    path = hf_hub_download(PAIRS_REPO, "pref_pairs.jsonl", repo_type="dataset")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    print(f"loaded {len(rows)} preference pairs")
    dl = DataLoader(PairDS(rows, tok, im_end, args.maxlen), batch_size=args.bs,
                    shuffle=True, collate_fn=collate)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    step = 0
    for epoch in range(args.epochs):
        for bi, (chosen, rejected) in enumerate(dl):
            ci, ca, cm = pad_batch(chosen, tok.pad_token_id, dev)
            ji, ja, jm = pad_batch(rejected, tok.pad_token_id, dev)

            pc = logps(model, ci, ca, cm)                 # policy chosen  (grad)
            pr = logps(model, ji, ja, jm)                 # policy rejected (grad)
            with torch.no_grad(), model.disable_adapter():
                rc = logps(model, ci, ca, cm)             # reference chosen  (frozen SFT)
                rj = logps(model, ji, ja, jm)             # reference rejected

            loss, crew, jrew = dpo_loss(pc, pr, rc, rj, beta=args.beta)
            loss = loss.mean()
            (loss / args.grad_accum).backward()

            if (bi + 1) % args.grad_accum == 0:
                opt.step(); opt.zero_grad(); step += 1
                acc = (crew > jrew).float().mean().item()
                margin = (crew - jrew).mean().item()
                print(f"epoch {epoch} step {step} | loss {loss.item():.4f} "
                      f"| reward_acc {acc:.2f} | reward_margin {margin:.3f}")

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("saved DPO adapter to", args.out)


if __name__ == "__main__":
    main()
