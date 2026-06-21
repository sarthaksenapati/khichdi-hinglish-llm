"""Day 5: MinHash + LSH near-deduplication of the scored pool.

Exact dups were already removed (instruction-keying during generation). This
catches NEAR-duplicate instructions (e.g. the "<appliance> ka use" clusters):
shingle each instruction -> MinHash signature -> LSH to find near-dups at
Jaccard >= threshold -> keep the HIGHEST-SCORED row per cluster, drop the rest.
Output: data/clean/sft_pool_dedup.jsonl  (then re-run filter.py on it).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasketch import MinHash, MinHashLSH

from src.utils import get_logger, set_seed

logger = get_logger("dedup")

WORD = re.compile(r"\w+", re.UNICODE)


def shingles(text: str, k: int = 5) -> set:
    toks = WORD.findall(text.lower())
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def minhash(text: str, num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in shingles(text):
        m.update(sh.encode("utf-8"))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/clean/sft_scored.jsonl")
    ap.add_argument("--out", default="data/clean/sft_pool_dedup.jsonl")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--num-perm", type=int, default=64)
    args = ap.parse_args()
    set_seed(42)

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    logger.info("pool: %d rows", len(rows))

    mhs = [minhash(r["instruction"], args.num_perm) for r in rows]

    # process best-scored first, so the row we KEEP from each near-dup cluster is the best one
    order = sorted(range(len(rows)), key=lambda i: rows[i].get("score_avg", 0), reverse=True)
    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)
    kept, dropped = [], 0
    for i in order:
        if lsh.query(mhs[i]):          # a near-duplicate is already kept
            dropped += 1
            continue
        lsh.insert(str(i), mhs[i])
        kept.append(i)

    kept_rows = [rows[i] for i in kept]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("near-dedup at Jaccard>=%.2f: dropped %d, kept %d / %d -> %s",
                args.threshold, dropped, len(kept_rows), len(rows), args.out)


if __name__ == "__main__":
    main()