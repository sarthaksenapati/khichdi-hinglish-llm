"""Day 2: Base-model behavior probe.

Runs a FIXED set of prompts (configs/probe.yaml) through a model and dumps
each completion to a markdown table for you to annotate by hand. The point:
SEE that a base model is a document-completer, not an assistant -- it
continues text, drifts, repeats, and often never emits a stop token.

This same harness is reused on the pod for SFT and DPO checkpoints, so the
base/SFT/DPO comparison uses identical prompts. Hence --model and --label.

Usage (CPU, slow but fine for a few prompts):
    python scripts/base_behavior.py --limit 4
    python scripts/base_behavior.py                 # all prompts
    python scripts/base_behavior.py --model <ckpt> --label sft   # later
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from src.utils import get_logger, set_seed

logger = get_logger("base_behavior")


def load(model_name: str, precision: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict = {}
    if precision == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif precision == "fp32":
        kwargs["torch_dtype"] = torch.float32
    elif precision == "nf4":  # GPU only
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    device_map = "auto" if torch.cuda.is_available() else None
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, **kwargs)
    if device_map is None:
        model = model.to("cpu")
    model.eval()
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/probe.yaml"))
    ap.add_argument("--model", default=None, help="override config model_name")
    ap.add_argument("--label", default="base", help="tag for the output file")
    ap.add_argument("--precision", default=None, choices=["fp32", "bf16", "nf4"])
    ap.add_argument("--limit", type=int, default=None, help="run only first N prompts")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_name = args.model or cfg["model_name"]
    precision = args.precision or cfg.get("precision", "fp32")
    template = cfg["template"]
    prompts = cfg["prompts"]
    if args.limit:
        prompts = prompts[: args.limit]
    out = args.out or Path(f"reports/base_behavior_{args.label}.md")

    logger.info("Loading %s in %s (CUDA=%s)", model_name, precision, torch.cuda.is_available())
    model, tok = load(model_name, precision)

    lines = [f"# Behavior probe -- `{model_name}` ({args.label})", "",
             "Annotate the **Failure mode** column yourself: continuation / "
             "no-stop / repetition / persona-drift / refused-correctly / ok.", ""]
    for p in prompts:
        text = template.format(prompt=p["text"])
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, max_new_tokens=cfg.get("max_new_tokens", 80),
                do_sample=False,  # greedy: show the model's DEFAULT (most-likely) behavior
                pad_token_id=tok.eos_token_id,
            )
        completion = tok.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        stopped = out_ids[0][-1].item() == tok.eos_token_id
        logger.info("[%s] done (emitted EOS=%s)", p["id"], stopped)
        lines += [
            f"## {p['id']} -- {p['category']}", "",
            f"**Prompt:** {p['text']}", "",
            f"**Completion:**", "```", completion.strip(), "```",
            f"*Emitted stop token:* {stopped}", "",
            f"**Failure mode:** _______", "",
        ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
