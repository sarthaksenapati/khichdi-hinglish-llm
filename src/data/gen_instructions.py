"""Stage 1: generate diverse Hinglish INSTRUCTIONS (user messages).

Sampling taxonomy cells (script x domain x task x register x switch) and
asking per-cell is what gives the dataset its SHAPE — we decide the
distribution here, Self-Instruct style, instead of letting the model default.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from src.data.openrouter import chat
from src.utils import get_logger, set_seed

logger = get_logger("gen_instructions")

STYLE = {
    "romanized": "romanized Hinglish (Hindi grammar frame + English content words, Latin script)",
    "english": "plain English",
    "devanagari": "Hindi in Devanagari script",
}


def build_prompt(script, domain, task, register, switch, n):
    return [
        {"role": "system", "content":
            "You generate realistic user messages that real Indian users type to an "
            "assistant. Output ONLY a JSON array of strings, no commentary."},
        {"role": "user", "content":
            f"Write {n} distinct {register}, {switch}-code-switched user messages in "
            f"{STYLE[script]}, about {domain}, each a {task} request. "
            "Each message must be SELF-CONTAINED: include all needed context inline. Do NOT "
            "refer to 'this recipe', 'the above', an attachment, or any text the assistant "
            "cannot see. For romanized, use Latin script only; Hindi grammar with English "
            "nouns/tech terms is ideal. "
            f"Make the {n} messages MAXIMALLY DIVERSE — different subtopics, situations and "
            "phrasings; do NOT cluster many messages around the same object or template "
            "(e.g. not many about 'what can I make with <appliance>'). "
            "The user must ask for general help the assistant can give WITHOUT accessing "
            "private accounts, real-time prices, or the user's personal data. "
            f"Vary length. Return a JSON array of {n} strings."},
    ]


def parse_array(text: str) -> list[str]:
    s, e = text.find("["), text.rfind("]")          # be forgiving: grab first [...] block
    if s == -1 or e == -1:
        return []
    try:
        return [str(x).strip() for x in json.loads(text[s:e + 1]) if str(x).strip()]
    except json.JSONDecodeError:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/generate.yaml")
    ap.add_argument("--cells", type=int, default=8, help="how many random cells to sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(cfg["out_instructions"]); out.parent.mkdir(parents=True, exist_ok=True)

    scripts, weights = list(cfg["scripts"]), list(cfg["scripts"].values())
    written = 0
    with out.open("a", encoding="utf-8") as f:
        for _ in range(args.cells):
            script = random.choices(scripts, weights=weights)[0]   # honor the 70/20/10 mix
            domain = random.choice(cfg["domains"])
            task = random.choice(cfg["task_types"])
            register = random.choice(cfg["registers"])
            switch = random.choice(cfg["switch_density"])
            msgs = build_prompt(script, domain, task, register, switch, cfg["per_batch"])
            try:
                text = chat(msgs, model=cfg["gen_model"],
                            temperature=cfg["instruction_temperature"], max_tokens=2048)
            except Exception as e:
                logger.warning("cell failed: %s", e); continue
            for instr in parse_array(text):
                rec = {"instruction": instr, "script": script, "domain": domain,
                       "task": task, "register": register, "switch": switch}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            logger.info("cell %s/%s/%s -> total %d", script, domain, task, written)
    logger.info("wrote %d instructions to %s", written, out)


if __name__ == "__main__":
    main()