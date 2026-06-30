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
- ipo
- preference-optimization
- hinglish
- code-switching
---

# Model Card for qwen1.5b-hinglish-ipo

An **IPO (Identity Preference Optimization)**-aligned LoRA adapter for the Hindi–English (romanized **Hinglish**) assistant. This is one of three preference-optimization variants (DPO / IPO / KTO) trained on the same data to compare methods; it is published mainly as part of that comparison.

## Model Details

### Model Description

IPO replaces DPO's log-sigmoid loss with a **squared loss** that regresses the policy–reference margin toward a finite target `1/(2β)`, rather than pushing it to infinity. This makes it more conservative and resistant to overfitting deterministic preferences. The loss was implemented from scratch (`src/dpo/loss.py`, `ipo_loss`) and verified against its reference formula and analytic invariants (loss = `(1/(2β))²` = 25.0 at init for β=0.1).

The IPO LoRA was trained on top of the **merged SFT model**, which is therefore its reference policy.

- **Developed by:** Sarthak Senapati
- **Model type:** LoRA adapter for a causal decoder-only LM, IPO-aligned
- **Language(s):** Romanized Hindi–English (Hinglish, primary), English, some Devanagari
- **License:** Apache-2.0 (adapter); base Qwen2.5-1.5B is Apache-2.0
- **Finetuned from:** `sarthaksenapati/qwen1.5b-hinglish-sft-v2` (from `Qwen/Qwen2.5-1.5B`)

### Model Sources

- **Repository:** https://github.com/sarthaksenapati/khichdi-hinglish-llm
- **SFT model (reference):** https://huggingface.co/sarthaksenapati/qwen1.5b-hinglish-sft-v2
- **Sibling models:** [DPO](https://huggingface.co/sarthaksenapati/qwen1.5b-hinglish-dpo) · [KTO](https://huggingface.co/sarthaksenapati/qwen1.5b-hinglish-kto)
- **Preference dataset:** https://huggingface.co/datasets/sarthaksenapati/khichdi-pref

## Uses

### Direct Use

Low-stakes conversational assistance in Hinglish. In practice, prefer the **KTO or DPO** sibling — IPO did not significantly outperform the SFT baseline in this setup (see Results).

### Out-of-Scope Use

Not safety-aligned. No high-stakes use (medical/legal/financial), no reliance on factual correctness.

## Bias, Risks, and Limitations

Same family of limitations as the SFT model: weak stop-control (rambling / no clean stop), weak Devanagari, occasional fabricated entities and confident unverified claims, and synthetic-data style bias. See `reports/error_analysis.md` in the repo. **Additionally:** with the shared hyperparameters used here, IPO under-performed — its squared loss is more sensitive to β than DPO's, and the DPO-tuned β=0.1 likely under-drove it.

## How to Get Started with the Model

This adapter was trained on top of the **merged SFT model** — apply the SFT adapter and merge it first, then load this IPO adapter:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B"
SFT  = "sarthaksenapati/qwen1.5b-hinglish-sft-v2"
IPO  = "sarthaksenapati/qwen1.5b-hinglish-ipo"

CHATML = ("{% for m in messages %}"
          "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>' + '\n' }}"
          "{% endfor %}"
          "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")

tok = AutoTokenizer.from_pretrained(IPO); tok.chat_template = CHATML
im_end = tok.convert_tokens_to_ids("<|im_end|>")

base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")
base = PeftModel.from_pretrained(base, SFT).merge_and_unload()   # merge SFT (the reference)
model = PeftModel.from_pretrained(base, IPO).eval()              # apply IPO adapter

msgs = [{"role": "user", "content": "Ek achhi morning routine ke liye tips do."}]
text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**ids, max_new_tokens=256, do_sample=False,
                     repetition_penalty=1.2, no_repeat_ngram_size=3, eos_token_id=im_end)
print(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
```

## Training Details

- **Data:** `sarthaksenapati/khichdi-pref` — 1,497 on-policy preference pairs (same as DPO).
- **Method:** IPO (squared-loss preference optimization), from-scratch loss, LoRA.
- **Setup:** merge SFT into base (= reference); fresh LoRA for IPO; `disable_adapter()` recovers the reference.
- **Hyperparameters:** bf16; LoRA r=16, alpha=32; β=0.1; lr=1e-5 (AdamW); 3 epochs; batch 2 × grad-accum 8; gradient checkpointing. Trainable 18.46M / 1.56B (1.18%). First-step loss = 25.0 = `(1/(2β))²` (correctness probe).

## Evaluation

- **Testing data:** 300 held-out prompts from `khichdi-sft` (`eval_prompts`), unseen in training.
- **Win-rate vs SFT:** **53.2%** of decisive pairs (95% CI **43.9–62.2**) — the CI **includes 50%**, so IPO is **not** significantly better than SFT in this configuration. Length neutral (SFT 150 ≈ IPO 149 words).
- **MMLU (0-shot slice, ±0.0137):** **0.6254** — flat vs base (0.6263); no capability loss.

### Summary

IPO preserved capability and answer length but, with hyperparameters shared across all three methods, did **not** separate from the SFT baseline. The likely cause is that IPO's squared loss needs its own β tuning. For practical use, prefer the DPO or KTO sibling. Published for transparency and as part of the DPO/IPO/KTO comparison (see the repo and the write-up).

## Environmental Impact

- **Hardware:** 1× NVIDIA RTX 4090 (rented, RunPod); **< 1 GPU-hour**; carbon negligible.

## Technical Specifications

Qwen2.5-1.5B decoder-only transformer + LoRA; objective = IPO squared loss with the SFT model as frozen reference. Software: PyTorch, transformers, PEFT, bf16; DeepSeek (via OpenRouter) for evaluation.

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
