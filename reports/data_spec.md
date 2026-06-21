# Data Spec — Khichdi SFT Set (the contract)

> Pre-registers every dataset choice so each one has a reason, not a vibe.
> Decided on Day 2, BEFORE collecting a single row. Numbers below are
> committed; the parentheticals are the defense for each one.

## 0. Evidence base (Days 1–2)
- Qwen2.5 fertility: English **1.17**, romanized Hinglish **1.62**, Devanagari **4.59** tok/word.
  Romanized is ~2.8× cheaper than Devanagari; even हूँ ("hoon") costs 5–6 byte-tokens in Devanagari.
- Base-model probe (8 prompts, greedy): emitted a stop token **1/8** times; answered in
  **English to every romanized prompt**; **degenerated on Devanagari** (`बैठक: बैठक �`);
  ignored format constraints ("3 steps", "ek line"); **complied with an unsafe request**.
- Implication: SFT must teach (a) STOP, (b) respond in Hinglish, (c) honor format constraints.
  The model will not speak Hinglish on its own — the SFT data is the only thing that makes it.

## 1. Goal (one sentence)
A friendly everyday-life assistant for Indian users that answers in natural romanized
Hinglish — covering cooking, tech help, travel, personal finance basics, career, health/
wellness, and writing help — follows instructions, and stops cleanly.

## 2. Script / code-switch policy  ← headline decision
- **Primary script: romanized.** It tokenizes 2.8× cheaper than Devanagari AND the base
  model already handles it best — the efficient path and the product-correct path coincide.
- Target style: **matrix-language frame** — Hindi grammar skeleton, English content words
  ("laptop slow ho raha hai, kuch quick fixes batao"). This is how the target users type.
- **Mix (sums to 100%):**
  - romanized code-switched — **70%**
  - pure English — **20%**
  - Devanagari — **10%**
- Reasoning: 70% romanized = the core product on the cheap-token path. 20% English =
  retention insurance (the base is strong in English; Hinglish-only training erodes it, and
  real users sometimes send fully-English turns). 10% Devanagari = robustness so Devanagari
  *input* doesn't break the model — but Devanagari is NOT a first-class output target
  (it degenerated in the probe; reaching quality there is out of budget). See §8 limitation.

## 3. Taxonomy (curate to fill cells, not just hit a count)
| Axis | Buckets & target |
|---|---|
| Task type | Q&A/factual **25%**, how-to/advice **25%**, explanation **20%**, creative/writing **15%**, rewrite/summarize/translate **15%** |
| Domain | roughly even across cooking, personal-finance, tech-help, travel, health/wellness, career, daily-life/relationships, education (~12% each) |
| Length | short **30%** / medium **50%** / long **20%** (short bucket deliberately teaches STOP) |
| Register | casual (yaar/bro) **40%** / neutral **40%** / polite **20%** |
| Switch density | light **25%** / medium **50%** / heavy **25%** |

## 4. Size & sources
- **Final SFT set: 10,000** examples.  Raw target **~22,000** → cut to 10k (~2.2:1).
  (LIMA: quality > quantity. 10k is large enough to shift behavior, small enough for one
  person to quality-audit in 3 weeks, and cheap to generate (~$3–5 on OpenRouter).)

**Survey finding that drives this section (HF, Day 3):** native code-switched Hinglish
instruction data is essentially *nonexistent* publicly — the available sets are overwhelmingly
machine-translated from English (translationese). This is the explicit justification for a
synthetic-primary pipeline with naturalness controls (§5.3, §8), rather than fine-tuning on an
off-the-shelf set.

- Sources, used by their actual strengths (NOT bulk-imported — translated rows reintroduce the
  translationese we filter out):
  1. **Synthetic generation** via OpenRouter (Llama-3.3-70B) — **~85% of romanized**. The
     backbone; full control over code-switch style, taxonomy, and answerability.
  2. **`findnitai/english-to-hinglish` (human-annotated subset only)** — real Hinglish, used as
     (a) few-shot naturalness anchors in the generator and (b) a small batch of genuine real-human
     training rows. License: unspecified → verify before redistributing on the Hub.
  3. **`ai4bharat/indic-instruct-data-v0.1`** (CC-BY-4.0, attribute) — fills the **Devanagari
     bucket** (real Devanagari instruction rows, sidestepping the generator's script-conditioning
     leak) and supplies extra human-originated **prompts** (responses discarded, re-answered in our
     style).
  4. **Style-only mining** from L3Cube-HingCorpus / HinGE (real Hinglish text, not instruction
     format) — harvest authentic phrasings to seed few-shot anchors. Not used as rows.
  - Excluded: translated QA sets (e.g. xquad-based collections) — translationese by construction.

## 5. Filtering ladder (cheap → expensive)
1. Heuristics: response length **5–512 tokens**; drop >30% trigram repetition; drop encoding
   garbage (the `�` char); drop truncated (no sentence-final punctuation).
2. Language-ID / code-switch ratio: in the 70% bucket require a genuine mix (neither ~100%
   English nor ~100% Hindi); pure-English allowed only inside the 20% bucket; stray
   pure-Devanagari allowed only inside the 10% bucket.
3. LLM quality score (1–5) on three axes: correctness, helpfulness, **naturalness of
   code-switching**. Keep ≥4. Naturalness is the differentiator — a generic scorer misses it.
4. Manual audit: read **100 random rows** at each major stage (native-speaker naturalness check).

## 6. Dedup & decontamination
- Exact: hash normalized text.
- Near-dup: MinHash + LSH, **Jaccard ≥ 0.80** → drop. (Synthetic data clones templates.)
- Decontaminate (8-gram overlap) against: MMLU slice, Hinglish-MT-Bench questions, the
  held-out preference prompts, and the eval prompts. **Log every removal** (the log is an artifact).

## 7. Held-out splits — reserved FIRST, frozen now
From a single prompt pool, carve out before building anything:
- **Eval prompts: 300** — never trained on anywhere (used Day 14 + Day 19).
- **Preference prompts: 1,500** — never in SFT (used Week 3 for DPO pairs).
- **SFT: 9,000 train / 500 val** (out of the 10k), split by prompt so no prompt leaks across.
Freezing now = the eval cannot be accused of being fitted to results.

## 8. Open risks & mitigations
- **Synthetic translationese** (reads like literal MT, not real Hinglish): naturalness axis in
  §5.3 + manual audit + few-shot the generator with real code-switched exemplars.
- **Judge length bias** leaking into data/eval: position-swap + report response-length stats (Week 2–3).
- **Romanization has no canonical spelling** (chahiye/chahiey): accept variation, don't
  over-normalize; keep generation-prompt spelling internally consistent.
- **Mode collapse back to English**: the §5.2 language-ID filter enforces real Hinglish in the 70%.
- **Documented limitation:** Devanagari output is not a quality target in v1 (evidence: §0 probe + fertility).
