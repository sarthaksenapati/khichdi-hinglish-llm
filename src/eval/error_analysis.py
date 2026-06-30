"""Day 19: qualitative error analysis of a model's held-out completions.

Labels each completion against a fixed failure taxonomy using the judge LLM,
aggregates the counts into a failure profile, and dumps a readable sample of
flagged examples so you can validate the labels BY HAND (trust-but-verify).

Input JSONL rows: {"prompt": ..., "a": ..., "b": ...}  (we analyse --field, default b)
  python -m src.eval.error_analysis --in data/eval/kto_vs_sft.jsonl --field b --name kto
"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.openrouter import chat
from src.utils import get_logger

logger = get_logger("error_analysis")
JUDGE = "deepseek/deepseek-chat"

TAXONOMY = ["no_clean_stop", "repetition", "hallucination", "dubious_claim",
            "language_issue", "not_following", "incomplete"]


def label_prompt(prompt, reply):
    return [
        {"role": "system", "content":
            "You are an error annotator for a Hindi-English (Hinglish) assistant. Given a user "
            "message and the assistant's reply, list which issues are present. Use ONLY these tags:\n"
            "no_clean_stop (rambles past the answer / never stops), repetition (repeats phrases or "
            "ideas), hallucination (fabricated facts/entities), dubious_claim (confident "
            "unverified/pseudoscientific claim), language_issue (wrong-language drift or broken "
            "Hinglish/Devanagari), not_following (ignores/misreads the instruction), incomplete "
            "(cuts off without answering). If the reply is fine, return an empty list. "
            'Reply with ONLY JSON: {"tags": [...], "note": "<=12 words"}'},
        {"role": "user", "content": f"User message:\n{prompt}\n\nAssistant reply:\n{reply}"},
    ]


def parse(text):
    try:
        m = re.search(r"\{.*\}", text or "", re.S)
        obj = json.loads(m.group(0))
        tags = [t for t in obj.get("tags", []) if t in TAXONOMY]
        return tags, str(obj.get("note", ""))[:120]
    except Exception:
        return None, None


def label_row(row, field):
    reply = row.get(field, "")
    tags, note = parse(chat(label_prompt(row["prompt"], reply), model=JUDGE,
                            temperature=0, max_tokens=80))
    if tags is None:
        return None
    return {"prompt": row["prompt"], "completion": reply, "tags": tags, "note": note}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--field", default="b", help="which completion to analyse (a or b)")
    ap.add_argument("--name", default="model")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sample", type=int, default=15, help="flagged examples to dump for review")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    logger.info("labelling %d %s completions, judge=%s", len(rows), args.name, JUDGE)

    labelled = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(lambda x: label_row(x, args.field), rows):
            if r:
                labelled.append(r)

    out = f"data/eval/{args.name}_errors.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in labelled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(labelled)
    counts = Counter(t for r in labelled for t in r["tags"])
    clean = sum(1 for r in labelled if not r["tags"])
    logger.info("=== %s failure profile (n=%d) ===", args.name, n)
    logger.info("clean (no issues): %d (%.0f%%)", clean, 100 * clean / max(n, 1))
    for tag in TAXONOMY:
        c = counts[tag]
        logger.info("  %-15s %3d  (%.0f%%)", tag, c, 100 * c / max(n, 1))

    # dump a readable sample of FLAGGED examples for hand-validation
    md = f"data/eval/{args.name}_error_sample.md"
    flagged = [r for r in labelled if r["tags"]][:args.sample]
    with open(md, "w", encoding="utf-8") as f:
        f.write(f"# {args.name} — error-analysis sample ({len(flagged)} flagged of {n})\n\n")
        for r in flagged:
            f.write(f"**tags:** {', '.join(r['tags'])}  \n**note:** {r['note']}\n\n")
            f.write(f"- **prompt:** {r['prompt'][:300]}\n")
            f.write(f"- **reply:** {r['completion'][:600]}\n\n---\n\n")
    logger.info("wrote %s and %s", out, md)


if __name__ == "__main__":
    main()
