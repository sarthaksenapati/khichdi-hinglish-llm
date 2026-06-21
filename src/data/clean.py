"""Day 4 (stage 1): heuristic cleaning + script relabeling + mix measurement.

Reads data/raw/sft_raw.jsonl, applies cheap local filters, detects the ACTUAL
script of each response (generation labels are unreliable), writes survivors to
data/clean/sft_clean.jsonl, and prints drop reasons + the TRUE script mix.
No API calls — pure local, fast.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import get_logger

logger = get_logger("clean")

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
WORD = re.compile(r"[A-Za-z\u0900-\u097F]+")
# CJK, Hangul, Hiragana/Katakana, Arabic, Thai \u2014 none belong in Hinglish/English data.
FOREIGN = re.compile(r"[\u4E00-\u9FFF\uAC00-\uD7AF\u3040-\u30FF\u0600-\u06FF\u0E00-\u0E7F]")

# High-frequency romanized-Hindi function words. Their presence marks code-switch;
# English text contains ~none of them, so the ratio cleanly separates the two.
HINDI_MARKERS = {
    "hai", "hain", "ho", "hota", "hoti", "kya", "kyun", "kaise", "kab", "kahan",
    "mujhe", "mera", "meri", "mere", "main", "hum", "aap", "tum", "karo", "karna",
    "kar", "karta", "karte", "karein", "ke", "ka", "ki", "ko", "mein", "nahi",
    "nahin", "aur", "ek", "ye", "yeh", "woh", "wo", "toh", "to", "bhi", "se", "par",
    "liye", "raha", "rahe", "rahi", "kuch", "accha", "achha", "bahut", "yaar",
    "sakte", "sakta", "sakti", "chahiye", "hoga", "hogi", "abhi", "phir", "sab",
    "apne", "apna", "jab", "agar", "hi", "na", "matlab", "thoda", "zyada", "kaam",
}


def detect_script(text: str) -> str:
    """Devanagari by Unicode; romanized-vs-English by Hindi function-word ratio."""
    nonspace = [c for c in text if not c.isspace()]
    if not nonspace:
        return "empty"
    deva = len(DEVANAGARI.findall(text))
    if deva / len(nonspace) > 0.15:
        return "devanagari"
    toks = [t.lower() for t in WORD.findall(text)]
    if not toks:
        return "english"
    marker_ratio = sum(1 for t in toks if t in HINDI_MARKERS) / len(toks)
    return "romanized" if marker_ratio >= 0.05 else "english"


def max_ngram_repeat(text: str, n: int = 3) -> float:
    """Fraction the most-repeated n-gram occupies — catches looping/degeneration."""
    toks = text.lower().split()
    if len(toks) < n * 2:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    counts = Counter(grams)
    return max(counts.values()) / len(grams)


def drop_reason(row: dict, min_words: int, max_words: int, rep_thresh: float):
    instr = row.get("instruction") or ""
    resp = (row.get("response") or "").strip()
    if not resp:
        return "empty"
    if "\ufffd" in resp or "\ufffd" in instr:  # replacement char = encoding garbage
        return "garbage"
    if FOREIGN.search(resp) or FOREIGN.search(instr):  # stray CJK/Arabic/etc. = glitch
        return "foreign_script"
    nwords = len(resp.split())
    if nwords < min_words:
        return "too_short"
    if nwords > max_words:
        return "too_long"
    if max_ngram_repeat(resp) > rep_thresh:
        return "repetition"
    if nwords > 150 and resp[-1] not in ".!?।…\"')":  # long + no terminal punct = cut off
        return "truncated"
    return None                                # keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", default=["data/raw/sft_raw.jsonl"])
    ap.add_argument("--out", default="data/clean/sft_clean.jsonl")
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--max-words", type=int, default=350)
    ap.add_argument("--rep-thresh", type=float, default=0.30)
    args = ap.parse_args()

    rows = []
    for path in args.inp:
        rows += [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    logger.info("loaded %d raw rows from %d file(s)", len(rows), len(args.inp))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    drops, detected, mismatches = Counter(), Counter(), Counter()
    kept = 0
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            reason = drop_reason(r, args.min_words, args.max_words, args.rep_thresh)
            if reason:
                drops[reason] += 1
                continue
            sd = detect_script(r["response"])
            detected[sd] += 1
            if r.get("script") != sd:
                mismatches[(r.get("script", "?"), sd)] += 1
            r["script_detected"] = sd
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept += 1

    logger.info("kept %d / %d (%.1f%%)", kept, len(rows), 100 * kept / len(rows))
    logger.info("drops by reason: %s", dict(drops))
    logger.info("TRUE script mix (detected on the response):")
    for s, n in detected.most_common():
        logger.info("   %-11s %6d  (%.1f%%)", s, n, 100 * n / max(kept, 1))
    logger.info("label→detected mismatches (top 8): %s",
                dict(mismatches.most_common(8)))


if __name__ == "__main__":
    main()