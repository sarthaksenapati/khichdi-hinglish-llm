"""Day 4 (stage 3): select the final SFT set + reserve held-out prompts.

Per script bucket, keep the top-N by (score_avg, naturalness) to hit the 80/12/8
target. Then reserve disjoint held-out PREFERENCE and EVAL prompt sets from the
leftover decent-quality rows, so they never overlap SFT (data_spec §7).
Writes sft_final.jsonl, heldout_pref_prompts.jsonl, heldout_eval_prompts.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import get_logger, set_seed

logger = get_logger("filter")

TARGETS = {"romanized": 8000, "english": 1200, "devanagari": 800}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/clean/sft_scored.jsonl")
    ap.add_argument("--out", default="data/clean/sft_final.jsonl")
    ap.add_argument("--pref-out", default="data/clean/heldout_pref_prompts.jsonl")
    ap.add_argument("--eval-out", default="data/clean/heldout_eval_prompts.jsonl")
    ap.add_argument("--n-pref", type=int, default=1500)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--min-score", type=float, default=7.0, help="quality floor for held-out prompts")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    by = defaultdict(list)
    for r in rows:
        by[r.get("script_detected", "?")].append(r)

    selected, leftover = [], []
    for script, target in TARGETS.items():
        bucket = by.get(script, [])
        # sort by score, then naturalness, then a seeded random tiebreak (unbiased among ties)
        bucket.sort(key=lambda r: (r["score_avg"], r.get("score_naturalness", 0), rng.random()),
                    reverse=True)
        keep, rest = bucket[:target], bucket[target:]
        selected += keep
        leftover += rest
        if len(keep) < target:
            logger.warning("%s: only %d available, wanted %d", script, len(keep), target)
        logger.info("%s: kept %d (min score_avg in cut = %.2f)",
                    script, len(keep), keep[-1]["score_avg"] if keep else 0)

    for script, bucket in by.items():            # non-target scripts -> leftover
        if script not in TARGETS:
            leftover += bucket

    # held-out prompts: decent-quality leftovers, disjoint from SFT and each other
    pool = [r for r in leftover if r["score_avg"] >= args.min_score]
    rng.shuffle(pool)
    pref = pool[:args.n_pref]
    eval_ = pool[args.n_pref:args.n_pref + args.n_eval]

    rng.shuffle(selected)                         # shuffle so buckets are interleaved
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for path, data in [(args.out, selected), (args.pref_out, pref), (args.eval_out, eval_)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("SFT final: %d -> %s", len(selected), args.out)
    logger.info("held-out preference prompts: %d -> %s", len(pref), args.pref_out)
    logger.info("held-out eval prompts: %d -> %s", len(eval_), args.eval_out)
    logger.info("final SFT mix: %s", dict(Counter(r["script_detected"] for r in selected)))


if __name__ == "__main__":
    main()