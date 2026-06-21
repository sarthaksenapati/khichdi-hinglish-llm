"""Day 4: English top-up generator (script-LOCKED, no code-switching).

The main pipeline's english bucket leaked into romanized because every prompt
pushed code-switching. This generates PLAIN ENGLISH instruction-response pairs to
fill the 12% English bucket. Two stages in one script; responses parallelized.
Writes to a SEPARATE file (data/raw/english_topup.jsonl) so the main pool is safe.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from src.data.openrouter import chat
from src.utils import get_logger, set_seed

logger = get_logger("english_topup")

DOMAINS = ["cooking", "finance", "tech", "travel", "health", "career", "daily_life", "education"]
TASKS = ["qna", "howto", "explanation", "advice", "creative"]
REGISTERS = ["casual", "neutral", "polite"]

INSTR_SYS = ("You generate realistic user messages in PLAIN, fluent ENGLISH — no Hindi, no "
             "code-switching, no Hinglish. Output ONLY a JSON array of strings, no commentary.")
RESP_SYS = ("Reply in clear, concise English. Answer the user's actual message, be helpful, "
            "then stop. No Hindi or code-switching.")


def instr_prompt(domain, task, register, n):
    return [
        {"role": "system", "content": INSTR_SYS},
        {"role": "user", "content":
            f"Write {n} distinct {register} user messages in plain English, about {domain}, "
            f"each a {task} request. Each must be SELF-CONTAINED (any needed context inline). "
            f"Make them MAXIMALLY DIVERSE — different subtopics, no repeated template. The user "
            f"must ask for general help answerable without private accounts or real-time data. "
            f"Return a JSON array of {n} strings."},
    ]


def parse_array(text: str) -> list[str]:
    s, e = text.find("["), text.rfind("]")
    if s == -1 or e == -1:
        return []
    try:
        return [str(x).strip() for x in json.loads(text[s:e + 1]) if str(x).strip()]
    except json.JSONDecodeError:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/generate.yaml")
    ap.add_argument("--cells", type=int, default=70, help="domain/task/register combos to sample")
    ap.add_argument("--per-batch", type=int, default=22)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="data/raw/english_topup.jsonl")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model = cfg["gen_model"]

    # Stage 1: plain-English instructions
    recs = []
    for _ in range(args.cells):
        d, t, r = random.choice(DOMAINS), random.choice(TASKS), random.choice(REGISTERS)
        try:
            text = chat(instr_prompt(d, t, r, args.per_batch), model=model,
                        temperature=cfg["instruction_temperature"], max_tokens=2048)
        except Exception as e:
            logger.warning("instr cell failed: %s", e); continue
        for ins in parse_array(text):
            recs.append({"instruction": ins, "script": "english", "domain": d,
                         "task": t, "register": r, "switch": "none"})
        logger.info("instructions so far: %d", len(recs))

    # Stage 2: English responses, parallel
    def work(rec):
        rec = dict(rec)
        msgs = [{"role": "system", "content": RESP_SYS},
                {"role": "user", "content": rec["instruction"]}]
        rec["response"] = chat(msgs, model=model,
                               temperature=cfg["response_temperature"], max_tokens=1024).strip()
        return rec

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    lock, done = threading.Lock(), 0
    with out.open("w", encoding="utf-8") as f, \
            concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, rec): rec for rec in recs}
        for fut in concurrent.futures.as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:
                logger.warning("skip: %s", e); continue
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            done += 1
            if done % 50 == 0:
                logger.info("answered %d/%d", done, len(recs))
    logger.info("done -> %s (%d rows)", out, done)


if __name__ == "__main__":
    main()