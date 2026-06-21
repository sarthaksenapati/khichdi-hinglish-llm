"""Day 4 (stage 2): parallel LLM quality scoring.

Scores each (instruction, response) on correctness / helpfulness / naturalness
(1-5) via OpenRouter. Parallel + resumable. Writes data/clean/sft_scored.jsonl.
The scorer is a SEPARATE concern from the Week-3 preference judge.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.openrouter import chat
from src.utils import get_logger

logger = get_logger("score")

STYLE_NAME = {
    "romanized": "romanized Hinglish (Hindi grammar + English content words)",
    "english": "plain English",
    "devanagari": "Hindi in Devanagari script",
}


def score_prompt(instr: str, resp: str, script: str):
    style = STYLE_NAME.get(script, "the user's language")
    return [
        {"role": "system", "content":
            "You are a HARSH quality critic for assistant training data. Most replies are only "
            "adequate — be stingy and use the FULL 1-10 range, spreading scores widely. Reserve "
            "9-10 ONLY for replies that are specific, complete, accurate, and perfectly natural "
            "with zero flaws. Adequate-but-generic replies get 5-6. Any real flaw caps it at 4. "
            "First name the single biggest flaw in <=8 words, then score. Return ONLY compact "
            'JSON: {"flaw":"...","correctness":n,"helpfulness":n,"naturalness":n} with each n in 1-10.'},
        {"role": "user", "content":
            f"Expected reply style: {style}.\n"
            "correctness = factually right, no errors. helpfulness = specific and complete, not "
            "generic boilerplate. naturalness = fluent, native-sounding for that style; penalize "
            "translationese, awkwardness, or wrong-language drift.\n\n"
            f"[USER]: {instr}\n[ASSISTANT]: {resp}"},
    ]


def parse_scores(text: str):
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        d = json.loads(text[s:e + 1])
        return {k: int(d[k]) for k in ("correctness", "helpfulness", "naturalness")}
    except Exception:
        return None


def row_id(r: dict) -> str:
    return hashlib.md5((r["instruction"] + "||" + r.get("response", "")).encode("utf-8")).hexdigest()


def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        for l in path.open(encoding="utf-8"):
            try:
                done.add(json.loads(l)["_id"])
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/clean/sft_clean.jsonl")
    ap.add_argument("--out", default="data/clean/sft_scored.jsonl")
    ap.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shuffle", action="store_true", help="shuffle before --limit (diagnostic)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    for r in rows:
        r["_id"] = row_id(r)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    todo = [r for r in rows if r["_id"] not in done]
    if args.shuffle:
        random.seed(42)
        random.shuffle(todo)
    if args.limit:
        todo = todo[:args.limit]
    logger.info("%d rows, %d done, %d to score (%d workers, model=%s)",
                len(rows), len(done), len(todo), args.workers, args.model)

    def work(r):
        msgs = score_prompt(r["instruction"], r.get("response", ""), r.get("script_detected", "?"))
        text = chat(msgs, model=args.model, temperature=0.0, max_tokens=80)
        sc = parse_scores(text)
        if sc is None:
            raise ValueError("unparseable: " + text[:60])
        r = dict(r)
        r.update({f"score_{k}": v for k, v in sc.items()})
        r["score_avg"] = round(sum(sc.values()) / 3, 2)
        return r

    lock, n = threading.Lock(), 0
    with out.open("a", encoding="utf-8") as f, \
            concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, r): r for r in todo}
        for fut in concurrent.futures.as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:
                logger.warning("skip: %s", e); continue
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            n += 1
            if n % 200 == 0:
                logger.info("scored %d/%d", n, len(todo))
    logger.info("done -> %s (scored %d)", out, n)


if __name__ == "__main__":
    main()