---
base_model: Qwen/Qwen2.5-1.5B
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
language:
- hi
- en
datasets:
- sarthaksenapati/khichdi-pref
- sarthaksenapati/khichdi-sft
tags:
- base_model:adapter:Qwen/Qwen2.5-1.5B
- lora
- transformers
- kto
- preference-optimization
- hinglish
- code-switching
---

# Model Card for qwen1.5b-hinglish-kto

A **KTO (Kahneman–Tversky Optimization)**-aligned LoRA adapter for the Hindi–English (romanized **Hinglish**) assistant. In a three-way comparison (DPO / IPO / KTO) trained on the same data, **KTO was the strongest / tied-best method — using only cheap binary feedback** rather than preference pairs.

## Model Details

### Model Description

KTO is grounded in prospect theory: instead of pairwise preferences, it learns from **unpaired binary labels** (each output tagged *desirable* or *undesirable*) and pushes each output's implicit reward above/below a reference point `z0`. The loss was implemented from scratch (`src/dpo/loss.py`, `kto_loss`) and verified against its reference formula and analytic invariants (loss = `1 − σ(0)` = 0.5 per example at init).

The KTO LoRA was trained on top of the **merged SFT model**, which is therefore its reference policy.

- **Developed by:** Sarthak Senapati
- **Model type:** LoRA adapter for a causal decoder-only LM, KTO-aligned
- **Language(s):** Romanized Hindi–English (Hinglish, primary), English, some Devanagari
- **License:** Apache-2.0 (adapter); base Qwen2.5-1.5B is Apache-2.0
- **Finetuned from:** `sarthaksenapati/qwen1.5b-hinglish-sft-v2` (from `Qwen/Qwen2.5-1.5B`)

### Model Sources

- **Repository:** https://github.com/sarthaksenapati/khichdi-hinglish-llm
- **SFT model (reference):** https://huggingface.co/sarthaksenapati/qwen1.5b-hinglish-sft-v2
- **Sibling models:** [DPO](https://huggingface.co/sarthaksenapati/qwen1.5b-hinglish-dpo) · [IPO](https://huggingface.co/sarthaksenapati/qwen1.5b-hinglish-ipo)
- **Preference dataset:** https://huggingface.co/datasets/sarthaksenapati/khichdi-pref

## Uses

### Direct Use

Low-stakes conversational assistance in Hinglish. Among the three aligned variants, this (or DPO) is the one to use.

### Out-of-Scope Use

Not safety-aligned. No high-stakes use (medical/legal/financial), no reliance on factual correctness.

## Bias, Risks, and Limitations

Same family of limitations as the SFT model: the dominant remaining weakness is **stop-control** — the model rambles / does not reliably stop, which shows up as both run-on and truncated answers; plus weak Devanagari, ~15% fabricated entities, occasional unverified claims, and synthetic-data style bias. KTO answers were also marginally longer than SFT's (157 vs 150 words), a minor possible length confound on its win-rate. See `reports/error_analysis.md` in the repo.

## How to Get Started with the Model

This adapter was trained on top of the **merged SFT model** — apply the SFT adapter and merge it first, then load this KTO adapter:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B"
SFT  = "sarthaksenapati/qwen1.5b-hinglish-sft-v2"
KTO  = "sarthaksenapati/qwen1.5b-hinglish-kto"

CHATML = ("{% for m in messages %}"
          "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>' + '\n' }}"
          "{% endfor %}"
          "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")

tok = AutoTokenizer.from_pretrained(KTO); tok.chat_template = CHATML
im_end = tok.convert_tokens_to_ids("<|im_end|>")

base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")
base = PeftModel.from_pretrained(base, SFT).merge_and_unload()   # merge SFT (the reference)
model = PeftModel.from_pretrained(base, KTO).eval()              # apply KTO adapter

msgs = [{"role": "user", "content": "Weekend pe Goa trip plan karne mein help karo."}]
text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**ids, max_new_tokens=256, do_sample=False,
                     repetition_penalty=1.2, no_repeat_ngram_size=3, eos_token_id=im_end)
print(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
```

## Training Details

- **Data:** `sarthaksenapati/khichdi-pref` → `kto_data.jsonl` — the 1,497 preference pairs reshaped into **2,994 unpaired binary examples** (each `chosen` → desirable, each `rejected` → undesirable).
- **Method:** KTO (prospect-theoretic binary alignment), from-scratch loss, LoRA. Reference point `z0` estimated per batch (a documented simplification of canonical KTO's mismatched-pair KL estimate).
- **Setup:** merge SFT into base (= reference); fresh LoRA for KTO; `disable_adapter()` recovers the reference.
- **Hyperparameters:** bf16; LoRA r=16, alpha=32; β=0.1; lr=1e-5 (AdamW); 3 epochs; batch 4 × grad-accum 8; gradient checkpointing. Trainable 18.46M / 1.56B (1.18%). First-step loss = 0.5 = `1 − σ(0)` (correctness probe).

## Evaluation

- **Testing data:** 300 held-out prompts from `khichdi-sft` (`eval_prompts`), unseen in training.
- **Win-rate vs SFT:** **76.3%** of decisive pairs (95% CI **68.5–82.6**) — the highest of the three methods, and clear of chance. (DPO 68.2%, IPO 53.2%; KTO vs DPO is a statistical tie as their CIs overlap.)
- **MMLU (0-shot slice, ±0.0137):** **0.6184** — flat vs base (0.6263) and SFT (0.6184); no capability loss.

### Summary

KTO matched or beat the pairwise methods (DPO/IPO) while using only **binary** feedback, which is cheaper to collect than preference pairs — the headline finding of the comparison. It preserved general capability. The main remaining weakness is stop-control, common to all three methods, which the length-matched preference data could not address; the targeted fix is stopping-specific preference data.

## Environmental Impact

- **Hardware:** 1× NVIDIA RTX 4090 (rented, RunPod); **< 1 GPU-hour**; carbon negligible.

## Technical Specifications

Qwen2.5-1.5B decoder-only transformer + LoRA; objective = KTO prospect-theoretic loss (per-example desirable/undesirable vs a reference point z0) with the SFT model as frozen reference. Software: PyTorch, transformers, PEFT, bf16; DeepSeek (via OpenRouter) for evaluation.

## Citation

```
@misc{khichdi2026,
  title  = {Khichdi: post-training Qwen2.5-1.5B into a Hinglish assistant (SFT -> DPO/IPO/KTO)},
  author = {Senapati, Sarthak},
  year   = {2026},
  url    = {https://github.com/sarthaksenapati/khichdi-hinglish-llm}
}
```

## Model Card Authors

Sarthak Senapati

### Framework versions

- PEFT 0.19.1
