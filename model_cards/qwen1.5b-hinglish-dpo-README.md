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
- dpo
- preference-optimization
- hinglish
- code-switching
---

# Model Card for qwen1.5b-hinglish-dpo

A DPO-aligned LoRA adapter that turns **Qwen2.5-1.5B-Base** into a Hindi–English (romanized **Hinglish**) chat assistant. This is the **preference-optimization (DPO) stage** that sits on top of an SFT model; it makes the assistant's answers more preferred while preserving general capability.

## Model Details

### Model Description

This adapter is the final stage of a two-step post-training pipeline (SFT → DPO) built end-to-end on rented GPUs for a few dollars. The base model was first supervised-fine-tuned into a Hinglish assistant (`qwen1.5b-hinglish-sft-v2`); this DPO stage then optimizes that SFT model on **on-policy preference pairs** using a **from-scratch Direct Preference Optimization loss** (verified against TRL's formula and analytic invariants).

The DPO LoRA was trained on top of the **merged SFT model**, so the SFT model is its reference policy. Toggling this adapter on/off recovers the DPO policy vs the SFT reference.

- **Developed by:** Sarthak Senapati
- **Model type:** LoRA adapter for a causal decoder-only LM, DPO-aligned
- **Language(s) (NLP):** Romanized Hindi–English code-switching (Hinglish), primary; some English and Devanagari
- **License:** Apache-2.0 (adapter); base model Qwen2.5-1.5B is Apache-2.0
- **Finetuned from model:** `sarthaksenapati/qwen1.5b-hinglish-sft-v2` (itself fine-tuned from `Qwen/Qwen2.5-1.5B`)

### Model Sources

- **Repository:** https://github.com/sarthaksenapati/khichdi-hinglish-llm
- **SFT model (reference):** https://huggingface.co/sarthaksenapati/qwen1.5b-hinglish-sft-v2
- **Preference dataset:** https://huggingface.co/datasets/sarthaksenapati/khichdi-pref
- **Writeup:** https://sarthak-senapati.hashnode.dev/teaching-a-base-model-to-speak-hinglish-part-1-sft

## Uses

### Direct Use

Low-stakes conversational assistance in Hinglish: casual Q&A, suggestions, explanations, and everyday instruction-following in romanized Hindi–English.

### Downstream Use

A starting point for further preference optimization (IPO/KTO), or as a small, cheap Hinglish chat model for experimentation and as a portfolio/teaching artifact for the SFT → DPO pipeline.

### Out-of-Scope Use

Not safety-aligned. Do not use for high-stakes advice (medical, legal, financial), factual lookup where correctness matters, or any safety-critical or adversarial setting. The model can produce confident but wrong or dubious claims.

## Bias, Risks, and Limitations

- **Not safety-aligned** — no RLHF safety tuning; can produce harmful, biased, or false content.
- **Synthetic-data artifacts** — training data was LLM-generated, so the model can repeat plausible-sounding but incorrect or pseudoscientific claims.
- **Weak Devanagari** — the corpus is romanized-primary (Devanagari costs ~2.8× more tokens/word in this tokenizer), so native-script output is weaker.
- **Small model** — at 1.5B parameters, reasoning and knowledge are limited (MMLU ≈ 0.62 on a slice).
- **Residual verbosity / stopping** — DPO reduced but did not fully solve the SFT model's tendency to ramble; clean stopping would need a length-aware preference signal.

### Recommendations

Use only for low-stakes Hinglish interaction. Do not rely on factual correctness. Keep a human in the loop for anything consequential.

## How to Get Started with the Model

This DPO adapter was trained on top of the **merged SFT model**, so you must apply the SFT adapter first, merge it, then load this DPO adapter:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B"
SFT  = "sarthaksenapati/qwen1.5b-hinglish-sft-v2"
DPO  = "sarthaksenapati/qwen1.5b-hinglish-dpo"

CHATML = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

tok = AutoTokenizer.from_pretrained(DPO)
tok.chat_template = CHATML
im_end = tok.convert_tokens_to_ids("<|im_end|>")

base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")
base = PeftModel.from_pretrained(base, SFT).merge_and_unload()   # 1) merge SFT (the reference)
model = PeftModel.from_pretrained(base, DPO).eval()              # 2) apply DPO adapter

msgs = [{"role": "user", "content": "Mujhe weekend ke liye ek easy dinner recipe batao."}]
text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**ids, max_new_tokens=256, do_sample=False,
                     repetition_penalty=1.2, no_repeat_ngram_size=3, eos_token_id=im_end)
print(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
```

## Training Details

### Training Data

`sarthaksenapati/khichdi-pref` — **1,497 on-policy preference pairs** `{prompt, chosen, rejected}`. Built by sampling 4 completions per held-out prompt from the SFT model (temperature 0.7, top-p 0.9, with foreign-script token suppression), then ranking them with an independent judge (DeepSeek, shuffled to cancel position bias) into best (chosen) and worst (rejected). Pairs with no clear quality gap were dropped.

### Training Procedure

Direct Preference Optimization with a **from-scratch sigmoid DPO loss** (`L = −log σ(β · [(logπθ(y_w) − logπref(y_w)) − (logπθ(y_l) − logπref(y_l))])`), verified numerically against TRL's formula and analytic invariants (loss = log 2 at init, correct gradient signs). One model serves as both policy and reference: the SFT adapter is merged into the base (= reference), and a fresh LoRA adapter is added for DPO; `disable_adapter()` recovers the reference.

#### Preprocessing

Prompt+response tokenized with a ChatML template; loss is masked to the response tokens (plus the stop token) only.

#### Training Hyperparameters

- **Training regime:** bf16 non-mixed precision
- **Method:** DPO (sigmoid loss), LoRA
- **LoRA:** r = 16, alpha = 32, dropout = 0.0, targets = q/k/v/o + gate/up/down proj
- **β (KL strength):** 0.1
- **Learning rate:** 1e-5, AdamW
- **Epochs:** 3 (279 optimizer steps)
- **Batch:** 2 × grad-accum 8 (effective 16)
- **Gradient checkpointing:** on
- **Trainable params:** 18.46M / 1.56B (1.18%)

#### Speeds, Sizes, Times

Adapter ≈ 74 MB. ~279 steps in roughly 25–40 minutes on a single RTX 4090. First-step loss = 0.6931 (log 2), confirming the policy equals the reference at initialization.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

300 held-out prompts from `sarthaksenapati/khichdi-sft` (`eval_prompts` split), never seen in SFT or preference generation.

#### Metrics

- **Win-rate (DPO vs SFT):** blind pairwise preference judged by an independent model (DeepSeek), each pair scored in both orderings (a win counts only if it survives the position swap), with a Wilson 95% confidence interval. Reference is the SFT model, to isolate the marginal value of the DPO stage.
- **MMLU (1,140-question slice, 0-shot):** capability-regression check.
- **Length canary:** mean answer length, to detect length-driven reward over-optimization.

### Results

- **DPO vs SFT win-rate:** **68.2%** of decisive pairs (95% CI **59.0–76.1%**, clear of chance), 190 ties of 300.
- **Length canary:** SFT 150 words ≈ DPO 152 words — no length inflation.
- **MMLU:** base 0.6263 → SFT 0.6184 → **DPO 0.6193** (flat within noise; no alignment tax).

#### Summary

DPO produces a clean, statistically significant preference gain over its SFT reference while preserving general capability and answer length — i.e. better-preferred answers with no measurable over-optimization. (An earlier 2-epoch run reached only 60.9% with a CI touching chance; diagnosing under-training and retraining at 3 epochs / lr 1e-5 produced this result, verified safe by the flat MMLU and length canary.)

## Environmental Impact

- **Hardware Type:** 1× NVIDIA RTX 4090 (rented)
- **Hours used:** < 1 GPU-hour for the DPO stage
- **Cloud Provider:** RunPod
- **Carbon Emitted:** negligible (sub-hour single-GPU run)

## Technical Specifications

### Model Architecture and Objective

Qwen2.5-1.5B decoder-only transformer + LoRA adapter; objective = sigmoid DPO (Bradley–Terry preference loss) with the SFT model as the frozen reference.

### Compute Infrastructure

- **Hardware:** single RTX 4090 (24 GB)
- **Software:** PyTorch, transformers, PEFT, bf16; OpenRouter (DeepSeek) for judging

## Citation

If you use this model, please cite the repository:

```
@misc{khichdi2026,
  title  = {Khichdi: post-training Qwen2.5-1.5B into a Hinglish assistant (SFT -> DPO)},
  author = {Senapati, Sarthak},
  year   = {2026},
  url    = {https://github.com/sarthaksenapati/khichdi-hinglish-llm}
}
```

## Model Card Authors

Sarthak Senapati

## Model Card Contact

Via the GitHub repository: https://github.com/sarthaksenapati/khichdi-hinglish-llm

### Framework versions

- PEFT 0.19.1
