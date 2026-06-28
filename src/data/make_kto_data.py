"""Day 17: reformat preference PAIRS into KTO's unpaired BINARY data.

KTO doesn't use (chosen, rejected) pairs — it uses single completions each tagged
desirable/undesirable. We derive that from the pairs: every `chosen` -> desirable,
every `rejected` -> undesirable. Same information, KTO's shape. ~2x the rows.

  python -m src.data.make_kto_data --in data/pref/pref_pairs.jsonl \
      --out data/pref/kto_data.jsonl --push-repo sarthaksenapati/khichdi-pref
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/pref/pref_pairs.jsonl")
    ap.add_argument("--out", default="data/pref/kto_data.jsonl")
    ap.add_argument("--push-repo", default=None)
    args = ap.parse_args()

    pairs = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    n_des = n_undes = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps({"prompt": p["prompt"], "completion": p["chosen"],
                                "label": True}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"prompt": p["prompt"], "completion": p["rejected"],
                                "label": False}, ensure_ascii=False) + "\n")
            n_des += 1
            n_undes += 1
    print(f"wrote {args.out}: {n_des} desirable + {n_undes} undesirable = {n_des + n_undes} rows")

    if args.push_repo:
        from huggingface_hub import HfApi, create_repo
        create_repo(args.push_repo, repo_type="dataset", exist_ok=True)
        HfApi().upload_file(path_or_fileobj=args.out, path_in_repo="kto_data.jsonl",
                            repo_id=args.push_repo, repo_type="dataset")
        print("uploaded to", args.push_repo)


if __name__ == "__main__":
    main()
