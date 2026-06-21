# Lab Notes

Daily engineering log for the Khichdi project — what I did, what I learned, and
the key number from each day.

---

## Day 1 — Tokenizer forensics

- **Did:** Scaffolded the repo (config-driven, seeded). Wrote `tokenizer_report.py`
  (reads the corpora dir, writes a markdown report) and a VRAM sanity check; ran both on
  Qwen2.5-1.5B-Base. Read the Qwen2.5 technical report and HF "Summary of the tokenizers".

- **Learned:** Qwen's BPE fertility on my corpora — EN **1.17** tok/word, romanized Hinglish
  **1.62**, Devanagari **4.59**. Surprise: *fewer characters but more tokens* — Devanagari costs
  ~2.8× more tokens than romanized despite being shorter, because Qwen learned English subword
  merges but almost no Devanagari ones and falls back to raw bytes (even हूँ = 5–6 tokens).
  From the Qwen report: BBPE (byte-level BPE), GQA, SwiGLU, RoPE, RMSNorm, QKV bias, MoE.
  From HF: subword / byte-pair / unigram / wordpiece / word-level / character-level tokenization.

- **Next:** Decide the romanized-vs-Devanagari mix for the SFT set (Day 2 data spec).

- **Numbers to remember:** bf16 vs nf4 peak VRAM — *deferred to pod (no local NVIDIA GPU)*.
  Base model on the Hinglish probe: rambled, defaulted to English, emitted a stop token only 1/8.

## Day 2 — Base model behavior + data spec

- **Did:** Wrote the dataset contract (`data_spec.md`) — taxonomy, code-switch mix, sources,
  filters, splits — all decided *before* collecting. Built a reusable probe that runs the same
  fixed prompts across every checkpoint. Read the InstructGPT paper, the RLHF pipeline, and
  LIMA ("Less Is More for Alignment").

- **Learned:** The four base-model failure modes — continuation, no-stop, repetition, persona
  drift. RLHF aligns models to user intent: SFT → RM (reward model) → PPO. LIMA: a few thousand
  high-quality examples beat hundreds of thousands of mediocre ones, because SFT *selects a
  behavior mode* rather than teaching new knowledge. The "data moat" idea. My probe confirmed
  the theory: the base model stopped only 1/8, defaulted to English even on Hinglish prompts,
  and produced repetition + byte-garbage on Devanagari.

- **Next:** Build the synthetic generation pipeline (first hit on the live API).

- **Number to remember:** Script mix decided — 70% romanized / 20% English / 10% Devanagari.

## Day 3 — Data sourcing + synthetic generation pipeline

- **Did:** Built a resumable two-stage synthetic pipeline on OpenRouter (Llama-3.3-70B). Read
  the Self-Instruct paper and the Alpaca recipe. Ran a HuggingFace dataset survey. Iterated
  through several small test batches to validate quality before scaling to the full run.
  Used threadpoolexecutor to have 10 wokers and 10x the speed of data generation(answers to those instructions).

- **Learned:** The bootstrap loop (tiny seed → model generates tasks → learns from them).
  Separate instruction generation from response generation — sweep taxonomy cells to control the
  *shape* of the dataset, then answer in a second pass. Native code-switched Hinglish instruction
  data is almost nonexistent publicly — most is machine-translated (XQuAD, synthetic);
  AI4Bharat's indic-instruct-data-v0.1 is the only serious source (385K, CC-BY-4.0) but it's
  mostly Hindi, not true Hinglish. Found and fixed real data bugs across iterations: few-shot
  parroting, non-self-contained instructions, template collapse, script drift, within-cell
  repetition, and unanswerable "access my account" requests. Also caught a metadata bug — the
  english/devanagari buckets leak into romanized, so the labels don't match the data (will
  relabel by language-ID on Day 4).

- **Next:** Relabel the dataset by language-ID and measure the true script mix. Refined source
  strategy: synthetic stays primary (now justified by the survey), `findnitai` human subset
  becomes the naturalness anchor + a few real rows, AI4Bharat fills the Devanagari bucket and
  supplies extra prompts.

- **Numbers to remember:** 900 cells → 21,832 instructions + responses (romanized / English /
  Devanagari). Generation cost ~$2.5–3 total.


## Day 4 — Cleaning, Script Relabeling, and Filtering

- **Did:** Built the cleaning + filtering pipeline — data cleaning, script detection via regex
  (Unicode ranges, words, foreign scripts), and heuristic filters for repetition, wrong script,
  truncation, and corrupted text. Used a **function-word ratio** for romanized-vs-English because
  no standard language-ID handles romanized Indic text. Generated English + Devanagari top-ups
  (script-locked prompts) to fix the leaked mix. Ran LLM-as-judge quality scoring on three axes
  (correctness, helpfulness, naturalness), then built the final SFT set. Prevented data leakage
  by keeping training / preference / evaluation sets disjoint.

- **Learned:** The filtering ladder (cheap heuristics before expensive LLM scoring). Dataset
  balancing to a target mix. Why a scorer uses temperature 0 (deterministic, consistent grades).
  Two big judge lessons: (1) **LLM-as-judge is lenient by default** — a 1-5 rubric scored 99%
  at 5.0; a harsh 1-10 rubric with forced flaw-naming produced real tiers (7 / 7.67 / 8). (2) the
  judge has blind spots — correctness was pinned at 8, so my **manual audit** (not the metric)
  caught health-claim sycophancy. Trust your eyes, not just the pipeline.

- **Next:** MinHash near-dedup + decontamination, then to the pod.

- **Numbers to remember:** Combined pool 24,406 → **24,319** cleaned (foreign-script filter caught
  52). True pre-top-up mix **86.6% rom / 7.6% EN / 5.8% Deva** → revised target **80/12/8**.
  Top-ups: +1,529 EN, +1,094 Deva. Scored **24,242** (77 unparseable skips, ~0.3%). Final SFT
  **10,000 = 8,000 / 1,200 / 800**; quality floors rom **7.67**, EN 7.33, Deva 7.00. Held-out:
  **1,500 preference + 300 eval** prompts (disjoint). Day-4 API cost ~$3.

## Day 5 — MinHash, near-dedup, decontamination

- **Did:** Ran `dedup.py` to near-deduplicate the scored pool (MinHash + LSH), keeping the
  highest-scored row per near-dup cluster, then re-ran the filter on the deduped pool to produce
  the final dedup-clean 10k + disjoint held-out sets.

- **Learned:** **Shingling** — turn each instruction into a set of overlapping k-word sequences
  (shingles). **Jaccard similarity** — J = |A∩B|/|A∪B|; J=1 identical, J=0 disjoint. **MinHash** —
  hash every shingle and record the minimum hash over the set; the key property is
  P[minhash(A)==minhash(B)] = Jaccard(A,B), so a 64-number signature preserves similarity.
  **LSH** (locality-sensitive hashing) — split each signature into bands of r rows, hash each
  band, and only compare candidate pairs that collide in a band; it's a filtering system that
  turns O(n²) into ~O(n). Also: dedup *before* splitting train/eval, to prevent leakage.

- **Next:** Day 6 on the pod — push dataset to HF Hub, env setup, chat templating + loss masking,
  first SFT run.

- **Number to remember:** Dropped only **180 near-dups (0.74%)** at Jaccard ≥ 0.8 — a low number,
  meaning the Day-3 diversity prompting worked. Re-filtered to the same **10,000 (8,000/1,200/800)**,
  floors 7.67 / 7.33 / 7.00. Local/free (CPU).

## Day 6 — Chat templating, loss masking, first SFT run

- **Did:** Rented an RTX 4090 (50 GB container + 50 GB network volume) on RunPod @ $0.70/hr.
  Created a Weights & Biases account to monitor the run. Pushed the dataset to HF Hub (3 splits).
  Wrote and ran `train_sft.py` to train my first model (QLoRA), and saw my first-ever training
  outcome with `generate.py`, which compared the base Qwen against my LoRA-adapter-enabled model.
  For generation: `max_new_tokens=150` to cap output, `repetition_penalty=1.2` (makes repetition
  less attractive), and `no_repeat_ngram_size=3` (decoder refuses to repeat any 3-gram). Pushed
  the adapter to HF Hub, then terminated the pod.

- **Learned:** **Chat templating** — instructions/responses are wrapped into a chat format with
  `<|im_start|>` / `<|im_end|>` and system/user/assistant roles; the template encodes who is
  speaking and where each message starts/ends. In SFT the **training format must equal the
  inference format**. **Loss masking** — set prompt-token labels to `-100` so they're ignored in
  the loss (ignoring *loss* ≠ ignoring *attention* — the model still reads the prompt). The
  assistant turn (incl. `<|im_end|>`) keeps real labels so the model learns to **stop**. I used a
  plain `transformers` Trainer with **manual** masking — NOT TRL's `SFTTrainer` — so the masking
  is explicit and version-safe; verified the mask with `--inspect` before training. QLoRA =
  frozen 4-bit base + bf16 LoRA adapters (~1% of params). Payoff: SFT flips the base model from an
  **English document-completer** into a **Hinglish assistant**.

- **Next:** Day 7 — fix the **no-stop bug** (force-append `<|im_end|>` after truncation + raise
  `MAXLEN` 1024→2048), hyperparameter iteration (3 epochs / rank 32), compare vs this baseline.
  Then close Week 1 with LinkedIn post + X thread + blog draft. Devanagari quality +
  health-sycophancy → Week-3 DPO targets.

- **Number to remember:** Trainable params **18.46M / 1.56B = 1.18%**. SFT **eval_loss 1.67 → 1.23**
  over **2 epochs / 1,188 steps**, ~29 min, ~$0.34 GPU. Adapter **73.9 MB** on Hub. Diagnosis:
  no-stop traced to **MAXLEN=1024 truncating `<|im_end|>`** on long Devanagari examples
  (4.59 tok/word → a 350-word answer ≈ 1,600 tokens, cut before the stop token).

## Day 7 — Stop-token fix + iteration (SFT v2)

- **Did:** Retrained SFT **v2** with the fix: raised `MAXLEN` 1024→2048, force-append `<|im_end|>`
  on any truncated row, and bumped LoRA rank 16→32. Re-ran base-vs-v2 generation. Verified
  `<|im_end|>` tokenizes to a single id (151645) to rule out a tokenization bug. Pushed the v2
  adapter to HF Hub. Hit a CUDA OOM (MAXLEN 2048 + rank 32 at batch 8) → fixed by halving the
  batch to 4 and doubling grad-accum to 4 (same effective batch of 16, less activation memory).

- **Learned:** A fix can be *partially* right and still not solve the problem. The change improved
  eval (1.23→1.19) and fixed the Devanagari truncation, but the **rambling persisted** — so
  truncation was only a minor factor. Tokenization checked out, so it's not a bug. Real cause:
  the stop token is ~1 of every ~150 assistant tokens, so a model can have low eval_loss while
  barely learning to emit it (eval_loss doesn't "see" the stopping failure), and the verbose,
  multi-part training responses taught it to keep going. Conclusion: **this is what DPO fixes, not
  more SFT** — preferring concise, clean-stopping answers over rambling ones. Chasing it further
  on the SFT side has diminishing returns.

- **Next:** Week 2 — build the evaluation harness (win-rate with CIs, MMLU slice for capability
  regression, a small Hinglish MT-Bench-style set). Then Week 3 — preference data + DPO (loss
  from scratch, verified vs TRL) to fix the rambling, plus an IPO/KTO comparison.

- **Number to remember:** v2 **eval_loss 1.187** (vs v1's 1.233). Rank 32 → **36.9M trainable
  (2.34%)**, ~43 min, adapter **148 MB**. Stopping unchanged → deferred to DPO.
