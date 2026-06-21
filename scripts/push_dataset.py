from datasets import load_dataset

repo = "sarthaksenapati/khichdi-sft"
for name, path in [
    ("train", "data/clean/sft_final.jsonl"),
    ("pref_prompts", "data/clean/heldout_pref_prompts.jsonl"),
    ("eval_prompts", "data/clean/heldout_eval_prompts.jsonl"),
]:
    ds = load_dataset("json", data_files=path, split="train")
    ds.push_to_hub(repo, config_name=name, private=True)
    print(f"pushed {name}: {len(ds)} rows")