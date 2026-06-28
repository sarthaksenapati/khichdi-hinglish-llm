"""Day 8: LLM-as-judge win-rate (position-swapped, Wilson CI, length-aware).

Compares two models' completions on the same prompts with an INDEPENDENT judge.
Runs both orderings to cancel position bias; a win counts only if the model is
preferred in BOTH orders (else tie). Reports win/tie/loss, a 95% CI, and mean
answer lengths (length-bias canary).

Input JSONL rows: {"prompt": ..., "a": <model-A reply>, "b": <model-B reply>}
  python -m src.eval.win_rate --in data/eval/base_vs_sft.jsonl --a-name base --b-name sft
"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.openrouter import chat
from src.utils import get_logger

logger = get_logger("win_rate")
JUDGE = "deepseek/deepseek-chat"   # independent of Llama (data) and Qwen (policy)


def judge_prompt(prompt, first, second):
    return [
        {"role": "system", "content":
            "You compare two assistant replies to the same user message and pick the better one. "
            "Judge correctness, helpfulness, and natural Hinglish/English — penalize rambling, "
            "not stopping, and wrong-language drift. Answer with ONLY one token: A, B, or TIE."},
        {"role": "user", "content":
            f"User message:\n{prompt}\n\n[A]:\n{first}\n\n[B]:\n{second}\n\nBetter reply? A, B, or TIE."},
    ]


def parse_choice(text):
    t = (text or "").strip().upper()
    for tok in ("TIE", "A", "B"):
        if t.startswith(tok):
            return tok
    return None


def judge_pair(row):
    # order 1: a shown as A; order 2: swapped, b shown as A  -> cancels position bias
    c1 = parse_choice(chat(judge_prompt(row["prompt"], row["a"], row["b"]), model=JUDGE,
                           temperature=0, max_tokens=4))
    c2 = parse_choice(chat(judge_prompt(row["prompt"], row["b"], row["a"]), model=JUDGE,
                           temperature=0, max_tokens=4))
    v1 = {"A": "a", "B": "b", "TIE": "tie", None: "tie"}[c1]
    v2 = {"A": "b", "B": "a", "TIE": "tie", None: "tie"}[c2]   # swapped
    return v1 if v1 == v2 else "tie"        # disagreement across orders = position-biased = tie


def wilson(wins, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((center - margin) / denom, (center + margin) / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    logger.info("%d pairs, judge=%s", len(rows), JUDGE)

    verdicts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_pair, r): r for r in rows}
        for fut in concurrent.futures.as_completed(futs):
            try:
                verdicts.append(fut.result())
            except Exception as e:
                logger.warning("skip: %s", e)

    a_wins, b_wins, ties = verdicts.count("a"), verdicts.count("b"), verdicts.count("tie")
    decisive = a_wins + b_wins
    b_rate = b_wins / decisive if decisive else 0.0
    lo, hi = wilson(b_wins, decisive)
    a_len = sum(len(r["a"].split()) for r in rows) / max(len(rows), 1)
    b_len = sum(len(r["b"].split()) for r in rows) / max(len(rows), 1)

    logger.info("n=%d | %s wins=%d  %s wins=%d  ties=%d", len(verdicts), args.a_name, a_wins,
                args.b_name, b_wins, ties)
    logger.info("%s win-rate vs %s (excl. ties): %.1f%%  [95%% CI %.1f–%.1f%%]",
                args.b_name, args.a_name, 100 * b_rate, 100 * lo, 100 * hi)
    logger.info("mean length (words): %s=%.0f  %s=%.0f  <- length-bias canary",
                args.a_name, a_len, args.b_name, b_len)


if __name__ == "__main__":
    main()