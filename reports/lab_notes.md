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

## Day 8 — Evaluation design + win-rate harness

- **Did:** Built the win-rate harness (`src/eval/win_rate.py`) with DeepSeek as an independent
  judge. Removed position bias in `judge_pair` by scoring each pair in both orders and only
  counting a win if the verdict holds both ways (otherwise tie). Implemented the Wilson confidence
  interval from scratch (wins, n, z confidence multiplier, p = point estimate, denominator,
  center, margin). Validated on a 2-row test.

- **Learned:** Judge biases — position bias (caused by answer order), length bias (longer looks
  better), self-preference (a judge favors its own outputs). Wilson interval = the proper range
  around a win-rate. The eval toolkit: MT-Bench (absolute 1–10 scoring across categories — coding,
  reasoning, math, etc.), MMLU (capability-regression check — a large multiple-choice benchmark
  that flags if I broke general ability), held-out preference accuracy (does the model rank chosen
  > rejected on pairs it never trained on), KL divergence (how far the trained policy has drifted
  from the SFT reference).

- **Next:** Day 9 on the pod — generate base + SFT completions on the 300 held-out eval prompts
  (identical decoding), and run an MMLU slice on base vs SFT. Then Day 10 — run the judge and
  report win-rate + CI + length.

- **Number to remember:** Judge = DeepSeek (independent of the Llama generator and the Qwen
  policy). A win counts only when the verdict survives a position swap; 95% CI via Wilson.
  Harness validated locally, $0.

## Day 9 — Evaluating a fine-tuned model: generation harness + capability regression

- **Did:** Generated base + SFT completions on the 300 held-out eval prompts with identical
  decoding (only the adapter toggled), batched on the pod, and uploaded the pairs to HF
  (`khichdi-eval`). Ran an MMLU slice (1,140 q) on base vs SFT via lm-eval-harness. This day
  produced the eval *inputs* — no judging yet.

- **Learned:** A fair comparison holds everything constant except the weights — same prompts,
  decoding, max tokens, and chat template. Two eval families: generation evaluation (a model
  produces text and something judges the quality) and loglikelihood evaluation (MMLU — computes
  the probability of a given continuation conditioned on a prompt; fits MCQ and classification
  style evaluation). Alignment tax — the performance cost that can appear when models are tuned to
  follow instructions on human preferences, sometimes reducing performance on other academic
  tasks. Also: GPU inference scales by batching, not threads (compute-bound, not I/O-bound).

- **Next:** Day 10 — run the win-rate judge on the uploaded completions; report win-rate + CI +
  length canary.

- **Number to remember:** MMLU base 0.626 → SFT 0.618 (−0.8 pt, within one stderr; STEM
  identical). No measurable alignment tax on a 1,140-question slice.


## Day 10 — Win-rate: turning pairwise judgement into a defensible metric

- **Did:** Downloaded the base/SFT completions from HF and ran the win-rate judge
  (`src/eval/win_rate.py`) over the 300 held-out pairs — DeepSeek judge, position-swapped, Wilson
  CI. SFT won 234, base won 5, 61 ties → SFT win-rate **97.9%** over base (95% CI 95.2–99.1%) on
  239 decisive pairs. Length canary clean (base 157 w, SFT 151 w). Built the report-grade figure
  (`assets/winrate_mmlu.png`) pairing the win-rate with the MMLU delta.

- **Learned:** Win-rate is computed from pairwise preferences; including ties in the denominator
  blurs the rate, so the denominator is decisive pairs only. A confidence interval (CI) given the
  data is a range of plausible values for the true win-rate p, with (say) 95% confidence. Wald vs
  Wilson — Wilson recenters the midpoint and widens/shrinks the interval in a controlled way,
  staying inside [0,1]; it is widely recommended for accuracy, click-through rate, and preference
  win-rates, especially when the sample size isn't huge and proportions are skewed. The length
  canary guards against the judge's length bias — a win paired with comparable/shorter answers is
  a clean win.

- **Next:** Week 3 — build preference data and implement DPO (loss from scratch, verified vs TRL),
  targeting the documented weaknesses (rambling, no clean stop).

- **Number to remember:** SFT preferred over base 97.9% of decisive pairs (95% CI 95.2–99.1%),
  judged blind by an independent model, no length bias — paired with flat MMLU = quality up,
  capability intact.

## Day 11 — Preference learning & DPO: the theory and designing the preference data

- **Did:** Studied preference learning and the DPO loss. Decided the preference-data construction
  strategy — on-policy best-of-N + judge — and built the two-script pipeline:
  `src/data/gen_pref_samples.py` (sample N completions per prompt from the SFT model on the pod)
  and `src/data/judge_pref.py` (rank them into chosen/rejected with an independent judge).

- **Learned:** The DPO formula. DPO (Direct Preference Optimization) makes the chosen answer more
  likely than the rejected one, *relative to where the SFT model already was* — the reference is
  what stops it from collapsing. DPO works best on-policy: the pairs should be drawn from a
  distribution close to the model you're training. Best approach here — sample several completions
  from my own SFT model and let a judge rank them, so the model learns to prefer its own better
  samples over its own worse ones. From the DPO paper: it's a direct alignment method that removes
  the explicit reward model and the RL pipeline of RLHF. Key claim — parameterize the reward so it
  yields a closed-form optimal policy, then optimize that policy with a simple classification-style
  loss. Bradley–Terry model — estimates the relative skill of entities from the outcomes of paired
  comparisons; the DPO loss is built on it. The reference model acts as the anchor: without it the
  policy could maximize preference fit while destroying fluency and general behavior.

- **Next:** Day 12 — generate the preference samples on the pod and judge them into chosen/rejected
  pairs. Then Day 13 — implement the DPO loss from scratch and verify it against TRL.

- **Number to remember:** DPO drops the reward model + RL of RLHF via a closed-form optimal policy;
  it raises the chosen answer's log-prob *relative to the frozen SFT reference*, anchored by β.
  On-policy data (rank my own samples) targets my own failure modes.

## Day 12 — Generating & judging the preference set

- **Did:** Spun a volume-less pod and sampled N=4 completions per prompt from the SFT model for
  all 1,500 pref-prompts (`gen_pref_samples.py`), pushed to HF (`khichdi-pref`). Judged locally
  with `judge_pref.py` (DeepSeek, shuffled to fight position bias) → kept **1,497 / 1,500** pairs
  (dropped 3 with no clear gap). Hand-checked pairs: chosen beats rejected on quality, with length
  parity (chosen 147 w vs rejected 145 w) — the judge picked on quality, not length.

- **Learned:** Temperature rescales the logits before softmax — T<1 sharpens the distribution
  (more deterministic), T>1 flattens it (more diverse); T→0 is greedy. Top-p (nucleus) sampling
  keeps the smallest set of tokens whose cumulative probability ≥ p (e.g. 0.95), renormalizes, and
  samples only from that nucleus. For preference data you need diversity, so sample (not greedy) —
  greedy would give 4 identical samples with no signal. `num_return_sequences` draws all N samples
  in one batched forward pass. Constrained decoding via `suppress_tokens` masks chosen token ids to
  −∞ at every step. Holtzman et al., "The Curious Case of Neural Text Degeneration" — why greedy/
  beam search produce bland, repetitive, degenerate text (greedy picks the argmax each step; beam
  keeps the top-k sequences by cumulative probability). Honest limitation: on-policy chosen answers
  still ramble, so DPO will mainly fix quality, not stopping.

- **Iteration log (how the output improved):**
  1. **Smoke run 1** (T=0.8, top-p=0.95, rep-penalty=1.2): samples were diverse but riddled with
     Thai/CJK characters injected mid-sentence, plus rambling.
  2. **Lowered decoding** (T=0.7, top-p=0.9, rep-penalty=1.1) — hypothesis: a high repetition
     penalty under sampling pushes probability mass onto unused (foreign) tokens. Helped a little,
     but the script drift persisted — it's a real property of the model, not just decoding.
  3. **Added `suppress_tokens`** banning ~36,799 foreign-script token ids (CJK/Hangul/Kana/Arabic/
     Thai). Thai/CJK gone — but a few residual private-use & CJK-compatibility glyphs (e.g. U+FA1B)
     still leaked.
  4. **Extended the suppression range** to U+E000–U+FAFF (~36,994 tokens). Cleaner still, but a
     couple of Hebrew/Arabic presentation-form chars and a leaked `<|fim_middle|>` special token
     remained.
  5. **Stopped chasing unicode blocks; added an allowlist sanitizer at the judge step** — keep
     ASCII + Devanagari + common punctuation, strip everything else and any `<|...|>` markup. This
     is robust by construction (can't miss a block) and preserves Devanagari for the 12% slice.
     Final pairs are clean.

- **Next:** Day 13 — implement the DPO loss from scratch and verify it numerically against TRL.

- **Number to remember:** `suppress_tokens` (constrained decoding) at generation + an allowlist
  sanitizer at judging = clean on-policy preference data. Kept 1,497/1,500; chosen/rejected length
  parity (147/145 w) proves the labels are quality-driven, not length-driven.

## Day 13 — DPO loss from scratch + numerical verification

- **Did:** Wrote `src/dpo/loss.py` — `sequence_logps` (causal-shift, gather, response-mask, sum)
  and `dpo_loss` (the Bradley–Terry objective). The data flow:
                   Policy Model                      Reference Model
                      │                                  │
         prompt + chosen response           prompt + chosen response
                      │                                  │
                  logits                             logits
                      │                                  │
            sequence_logps()                  sequence_logps()
                      │                                  │
            policy_chosen_logps              ref_chosen_logps
                      │                                  │

         prompt + rejected response         prompt + rejected response
                      │                                  │
                  logits                             logits
                      │                                  │
            sequence_logps()                  sequence_logps()
                      │                                  │
          policy_rejected_logps            ref_rejected_logps
                      │                                  │
                      └──────────────┬───────────────────┘
                                     │
                policy_ratio = chosen − rejected
                reference_ratio = chosen − rejected
                                     │
                 margin = policy_ratio − reference_ratio
                                     │
                    loss = −logσ(β × margin)
                                     │
                              Backpropagation
- **Verified:** Wrote `scripts/verify_dpo_loss.py` with four checks — `sequence_logps` vs a
  brute-force per-token loop; the analytic invariant policy==reference ⇒ loss = log 2 and rewards
  = 0; gradient direction (raising chosen lowers loss, raising rejected raises it); and a numerical
  match to TRL's sigmoid-DPO formula (1e-6). All passed. Note: TRL 1.7.0 removed the standalone
  `dpo_loss` method, so I matched its exact formula + the analytic invariants rather than calling
  the object — invariants test meaning, not just agreement with a library.

- **Learned:** Instead of probabilities we use log-probabilities. An LM produces logits — the
  output at position t predicts token t+1, which is why we shift (causal shifting). How log-softmax
  works, and how a product of probabilities becomes a sum of log-probabilities (so a sequence
  log-prob is a masked sum of per-token log-probs). The roles of the policy and reference models,
  and how the loss and its gradient are shaped by them in DPO — the gradient is largest on the
  pairs the policy currently gets most wrong.

- **Next:** Day 14 — load the SFT model as policy (trainable) + frozen reference, compute these
  log-probs from real forward passes over the 1,497 pairs, and run DPO on the pod.

- **Number to remember:** First training loss on Day 14 should be ≈ 0.693 (log 2), because the
  policy starts equal to the reference (margin 0 ⇒ σ(0)=0.5 ⇒ −log 0.5). A free correctness probe
  for the run.

## Day 14 — DPO training run: policy + frozen reference, from-scratch loss

- **Did:** Implemented the DPO training pipeline (`train_dpo.py`) using my from-scratch loss. Flow:
                   Base Model
                     │
                     ▼
          Load SFT LoRA Adapter
                     │
                     ▼
        Merge LoRA into Base Model
                     │
                     ▼
      M = Frozen SFT Model (Reference)
                     │
         Add NEW DPO LoRA Adapter
                     │
                     ▼
      M + DPO LoRA = Policy Model
                     │
                     ▼
         Preference Dataset
      (prompt, chosen, rejected)
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
   chosen sequence        rejected sequence
         │                       │
         ▼                       ▼
Policy log probs         Policy log probs
Reference log probs      Reference log probs
         │                       │
         └───────────┬───────────┘
                     ▼
               DPO Loss
                     ▼
         Update ONLY DPO LoRA

- **Ran:** 2 epochs, 186 optimizer steps, on the 1,497 preference pairs. Trainable params 18.4M /
  1.56B (1.18%). First-step loss = **0.6931 = log 2 exactly** — the correctness probe fired (policy
  == reference at init, margin 0). Epoch 0 was mostly flat/noisy; epoch 1 improved — loss dropping
  into the 0.5s, reward_acc more often 1.0, reward_margin growing to 0.2–0.5. Saved the DPO LoRA
  adapter and pushed it to HF (`qwen1.5b-hinglish-dpo`).

- **Learned:** We need two policies, so merge the SFT adapter into the base weights to get the SFT
  model (M), then attach a fresh LoRA adapter for DPO on top: adapter ENABLED → policy, adapter
  DISABLED → reference. Used bf16 instead of 4-bit — at 1.5B params there's no real gain from
  quantization and merging is cleaner. Per-step reward_acc/margin are noisy because they're logged
  on a 2-sample micro-batch, so judge the trend across steps, not single lines.

- **Next:** Day 15 — generate DPO completions on the 300 held-out prompts, run the win-rate harness
  DPO vs SFT, and MMLU on the DPO model.

- **Number to remember:** First DPO step loss = 0.6931 (log 2) confirmed the from-scratch loss live.
  The DPO gain was modest and emerged in epoch 1 — the real verdict is the held-out win-rate.

## Day 15 — The DPO verdict: SFT-vs-DPO win-rate + capability check

- **Did:** Generated SFT vs DPO completions on the 300 held-out prompts (`eval_generate_dpo.py` —
  toggle the DPO adapter on/off over the merged SFT model, identical greedy decoding, so only DPO
  differs), pushed `dpo_vs_sft.jsonl` to `khichdi-eval`. Ran the win-rate harness (DPO vs SFT,
  DeepSeek judge, position-swapped, Wilson CI) and an MMLU slice on the merged DPO model. Built the
  pipeline figure (`assets/pipeline_results.png`).

- **Result (v2 — the chosen model):** DPO preferred over SFT **68.2%** of decisive pairs
  (95% CI **59.0–76.1%**, clear of chance), 190 ties, length canary clean (SFT 150 w ≈ DPO 152 w),
  MMLU **0.6193** (flat vs SFT's 0.6184 / base's 0.6263). Quality up, no length gaming, no
  capability tax. Full pipeline: SFT vs base 97.9% (CI 95.2–99.1%) → DPO vs SFT 68.2%; MMLU
  0.626 → 0.618 → 0.619 across base/SFT/DPO.

- **Iteration log (v1 → v2):**
  1. **v1** (2 epochs, lr 5e-6): DPO over SFT **60.9%** but 95% CI **50.4–70.5%** — lower bound
     barely above chance, 213 ties, MMLU 0.6263. A real but weakly-significant gain. (Screenshots:
     `DPO 1–5`, `mmlu 1`/`mmlu 2`.)
  2. **Diagnosed under-training** from the Day-14 curve — epoch 1 was still improving and 71% of
     eval pairs tied, so the model had barely moved on most prompts.
  3. **v2** (3 epochs, lr 1e-5): training fit much stronger (final loss ~0.16, reward_margin ~1.8,
     reward_acc 1.0). Eval: **68.2%, CI 59.0–76.1%** (clear of chance), ties down to 190.
     (Screenshots: `dpo a`/`dpo b`, `mmlu a`/`mmlu b`.)
  4. **Verified no over-optimization:** despite the aggressive training, MMLU stayed flat (0.6193)
     and DPO answer length did not inflate (152 ≈ 150). Both over-optimization tripwires held, so
     v2 is a clean improvement, not a traded-off one. v2 is the model.

- **Learned:** Evaluate a stage against its own *reference*, not base — DPO vs SFT isolates the
  preference stage's marginal value (DPO vs base would smuggle in the SFT gains). Reward
  over-optimization is the failure mode of pushing DPO too hard (it games length / judge quirks and
  erodes capability) — guard against it with a length canary + a capability check, not training
  metrics. The β/KL anchor is why capability survived even the aggressive run. More epochs helped up
  to a sweet spot; the real ceiling is the preference data, not the schedule.

- **Next:** Week 3 wrap — IPO/KTO comparison on the same pairs; update README + model card with the
  pipeline figure and DPO results.

- **Number to remember:** DPO (v2) beats SFT 68.2% (95% CI 59.0–76.1%, clear of chance) with MMLU
  and answer-length both flat — a clean preference gain with no over-optimization. The win came from
  fixing under-training (2→3 epochs, lr 5e-6→1e-5), verified safe by the guardrails.