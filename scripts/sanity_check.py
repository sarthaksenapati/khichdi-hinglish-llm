"""Day 1: Environment + VRAM sanity check.

Loads Qwen2.5-1.5B-Base in a chosen precision, runs one forward pass and
a short greedy generation, and reports VRAM. Goal: verify your VRAM
arithmetic empirically (fp32 ~6 GB, bf16 ~3 GB, NF4 ~1.1 GB for 1.5B)
and SEE raw base-model behavior (it should ramble, not assist -- that's
the 'before' picture for your whole project).

Usage:
    python scripts/sanity_check.py --precision nf4
    python scripts/sanity_check.py --precision bf16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.utils import get_logger, set_seed

logger = get_logger("sanity_check")

PROBE_PROMPT = "User: Mujhe ek achhi biryani recipe batao.\nAssistant:"


def load_model(model_name: str, precision: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict = {}
    if precision == "fp32":
        kwargs["torch_dtype"] = torch.float32
    elif precision == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif precision == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",            # quantile codebook for N(0,1) weights
            bnb_4bit_compute_dtype=torch.bfloat16, # matmuls dequantize to bf16
            bnb_4bit_use_double_quant=True,        # quantize the scale constants too
        )
    else:
        raise ValueError(f"unknown precision: {precision}")

    device_map = "auto" if torch.cuda.is_available() else None
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map, **kwargs
    )
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--precision", choices=["fp32", "bf16", "nf4"], default="nf4")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s", cuda)
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    model, tokenizer = load_model(args.model, args.precision)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %.2fB", n_params / 1e9)
    # Expected resident size: fp32=4B/param, bf16=2B, nf4~0.55B (incl. scales)
    bytes_per = {"fp32": 4.0, "bf16": 2.0, "nf4": 0.55}[args.precision]
    logger.info("Back-of-envelope weights size: %.2f GB", n_params * bytes_per / 1e9)

    inputs = tokenizer(PROBE_PROMPT, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    logger.info("Forward pass OK. Logits shape: %s (= [batch, seq, vocab])",
                tuple(logits.shape))

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                                  skip_special_tokens=True)
    logger.info("BASE MODEL completion (expect rambling, NOT assistance):")
    print("-" * 60)
    print(PROBE_PROMPT + completion)
    print("-" * 60)

    if cuda:
        peak = torch.cuda.max_memory_allocated() / 1e9
        logger.info("Peak VRAM allocated: %.2f GB", peak)
        logger.info("Compare against the back-of-envelope number above; the gap "
                    "is activations + CUDA context + fragmentation.")


if __name__ == "__main__":
    main()
