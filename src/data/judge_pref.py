"""Week 3 / Day 12 (local): rank the N samples per prompt into chosen/rejected.

Reads {prompt, samples:[...]}, asks an INDEPENDENT judge (DeepSeek) to name the
best and worst sample (shown shuffled to fight position bias), and emits a DPO
pair {prompt, chosen, rejected}. Pairs with no clear best/worst are dropped — a
preference pair is only useful when there is a real quality gap.

  python -m src.data.judge_pref --in data/pref/pref_samples.jsonl \
      --out data/pref/pref_pairs.jsonl --workers 12
"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.openrouter import chat
from src.utils import get_logger

logger = get_logger("judge_pref")
JUDGE = "deepseek/deepseek-chat"   # independent of generator (Llama) and policy (Qwen)


def judge_prompt(user_msg, cands):
    block = "\n\n".join(f"[{i + 1}]\n{c}" for i, c in enumerate(cands))
    return [
        {"role": "system", "content":
            "You rank assistant replies to the same user message. Judge correctness, "
            "helpfulness, and natural Hinglish/English. Penalize rambling, not stopping, "
            "wrong-language drift, and repetition. Reply with EXACTLY two lines:\n"
            "BEST: <number>\nWORST: <number>"},
        {"role": "user", "content":
            f"User message:\n{user_msg}\n\nCandidate replies:\n{block}\n\n"
            f"Give the single best and single worst by number."},
    ]


def parse_best_worst(text, n):
    best = worst = None
    for line in (text or "").splitlines():
        m = re.search(r"BEST\s*[:\-]?\s*(\d+)", line, re.I)
        if m:
            best = int(m.group(1))
        m = re.search(r"WORST\s*[:\-]?\s*(\d+)", line, re.I)
        if m:
            worst = int(m.group(1))
    if best and worst and 1 <= best <= n and 1 <= worst <= n and best != worst:
        return best, worst
    return None


def judge_row(row):
    samples = [s for s in row.get("samples", []) if s and s.strip()]
    if len(samples) < 2:
        return None
    order = list(range(len(samples)))
    random.shuffle(order)                       # shuffle -> kill position bias
    shown = [samples[i] for i in order]
    res = parse_best_worst(chat(judge_prompt(row["prompt"], shown), model=JUDGE,
                                temperature=0, max_tokens=12), len(shown))
    if not res:
        return None
    best_i, worst_i = res
    chosen = shown[best_i - 1]
    rejected = shown[worst_i - 1]
    if chosen.strip() == rejected.strip():
        return None
    return {"prompt": row["prompt"], "chosen": chosen, "rejected": rejected}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="data/pref/pref_pairs.jsonl")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    logger.info("%d prompts, judge=%s", len(rows), JUDGE)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pairs, dropped = [], 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(judge_row, rows):
            if r:
                pairs.append(r)
            else:
                dropped += 1

    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.info("kept %d pairs, dropped %d (no clear gap) -> %s", len(pairs), dropped, args.out)


if __name__ == "__main__":
    main()
