"""Base vs SFT generations on Hinglish probe prompts.

Loads base Qwen2.5-1.5B (4-bit) + the trained LoRA adapter, and for each prompt
generates from BASE (adapter disabled) and SFT (adapter enabled), side by side.
"""
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B"
ADAPTER = "/workspace/sft-qwen1.5b-hinglish-v2"   # or "sarthaksenapati/qwen1.5b-hinglish-sft-v2"
CHATML = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
PROMPTS = [
    "Mujhe ek chhoti si biryani recipe batao, sirf 3 steps mein.",
    "Yaar mera laptop bahut slow hai, kuch quick fixes suggest karo.",
    "Compound interest kaise kaam karta hai, ek simple example ke saath samjhao.",
    "मुझे घर पर करने वाले तीन व्यायाम बताइए।",
    "Mere dost ki shaadi ke liye ek line ka badhai message likho.",
    "What is the capital of France?",
]


def main():
    tok = AutoTokenizer.from_pretrained(ADAPTER)
    tok.chat_template = CHATML
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
                                                device_map={"": 0}, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    def gen(prompt):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=200, do_sample=False,
                                 repetition_penalty=1.2, no_repeat_ngram_size=3,
                                 eos_token_id=im_end, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()

    for p in PROMPTS:
        print("=" * 70)
        print("PROMPT:", p)
        with model.disable_adapter():
            print("\n[BASE]:", gen(p))
        print("\n[SFT ]:", gen(p))
        print()


if __name__ == "__main__":
    main()
