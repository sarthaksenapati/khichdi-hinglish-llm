"""Stage 2: generate a natural Hinglish RESPONSE per instruction.

Resumable: skips instructions already in the output, so Ctrl-C + rerun is safe
over a long rate-limited job. Few-shot exemplars anchor naturalness (anti-translationese).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from src.data.openrouter import chat
from src.utils import get_logger, set_seed

logger = get_logger("gen_responses")

SYS = {
    "romanized": "Reply in natural romanized Hinglish (Hindi grammar + English content words), "
                 "Latin script ONLY — never Devanagari, Tamil, Kannada or other scripts. Answer "
                 "the user's ACTUAL message; never copy the example conversations. Be helpful "
                 "and concise, then STOP. Do not switch to pure English.",
    "english": "Reply in clear, concise English. Answer the user's actual message; never copy "
               "the example conversations. Be helpful, then stop.",
    "devanagari": "Reply in natural Hindi (Devanagari). Answer the user's actual message; never "
                  "copy the example conversations. Be helpful and concise, then stop.",
}

# Two DIFFERENT exemplars (tech + cooking) so no single one dominates -> less parroting.
FEWSHOT = [   # real, natural Hinglish — anchors style, fights translationese
    {"role": "user", "content": "Yaar mera laptop slow ho gaya hai, kya karu?"},
    {"role": "assistant", "content":
        "Pehle background apps band karo aur startup programs disable karo. Phir disk "
        "cleanup chalao. Agar phir bhi slow hai toh ek SSD upgrade sabse bada difference dega."},
    {"role": "user", "content": "Ghar par 10 minute mein kya healthy snack bana sakta hoon?"},
    {"role": "assistant", "content":
        "Sabse easy hai fruit chaat — koi bhi fruits kaato, thoda chaat masala aur nimbu daalo, "
        "ho gaya. Ya phir roasted makhana with a little salt, woh bhi 5 minute mein ready."},
]


def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["instruction"])
            except Exception:
                pass
    return done


def answer_one(r: dict, cfg: dict) -> dict:
    """Generate one response. Runs inside a worker thread; returns the full row."""
    script = r.get("script", "romanized")
    msgs = [{"role": "system", "content": SYS[script]}, *FEWSHOT,
            {"role": "user", "content": r["instruction"]}]
    resp = chat(msgs, model=cfg["gen_model"],
                temperature=cfg["response_temperature"], max_tokens=1024)
    return {**r, "response": resp.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/generate.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=10, help="concurrent API calls")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    src, dst = Path(cfg["out_instructions"]), Path(cfg["out_sft"])
    dst.parent.mkdir(parents=True, exist_ok=True)

    done = load_done(dst)
    rows = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
    todo = [r for r in rows if r["instruction"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    logger.info("%d total, %d done, %d to do (%d workers)",
                len(rows), len(done), len(todo), args.workers)

    # I/O-bound work (waiting on the API), so threads give a ~Nx speedup.
    # One writer guarded by a lock keeps the JSONL append safe + resumable.
    write_lock = threading.Lock()
    answered = 0
    with dst.open("a", encoding="utf-8") as f, \
            concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(answer_one, r, cfg): r for r in todo}
        for fut in concurrent.futures.as_completed(futures):
            r = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:
                logger.warning("skip (%s): %s", r["instruction"][:40], e); continue
            with write_lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()                       # flush so a crash keeps finished rows
            answered += 1
            if answered % 50 == 0:
                logger.info("answered %d/%d", answered, len(todo))
    logger.info("done -> %s (answered %d)", dst, answered)


if __name__ == "__main__":
    main()