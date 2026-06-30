# Error analysis: where the aligned models still fail

Win-rate and MMLU say *which* model is preferred and *whether* capability held — neither says
*how* the model still fails. This is a qualitative pass over the held-out completions to
characterize the remaining failure modes of the two best models (DPO and KTO).

## Method

Each of the 300 held-out completions per model was labelled by an independent LLM annotator
(DeepSeek) against a fixed failure taxonomy, then aggregated into a per-category rate. A random
sample of flagged outputs was then **read by hand** to validate the labels — the same trust-but-
verify discipline used for the LLM judge. Script: `src/eval/error_analysis.py`.

Taxonomy: `no_clean_stop`, `repetition`, `hallucination`, `dubious_claim`, `language_issue`,
`not_following`, `incomplete`.

## Failure profile (n=300 each)

| Failure         | KTO | DPO |
|-----------------|-----|-----|
| repetition      | 76% | 76% |
| incomplete      | 60% | 58% |
| language_issue  | 31% | 37% |
| no_clean_stop   | 32% | 35% |
| hallucination   | 15% | 15% |
| not_following   |  7% | 10% |
| dubious_claim   |  3% |  4% |
| clean (no issues) | 1% | 0% |

**Read these as relative, not absolute.** The annotator is liberal — `repetition` in particular is
over-applied, and a ~1% "clean" rate overstates how broken the outputs are. The trustworthy signal
is the *ranking* and the *shape* of the profile, not the exact percentages.

## Findings (hand-validated)

**1. The two methods fail identically.** KTO and DPO have near-indistinguishable profiles. The
residual weaknesses are therefore a property of the **SFT base and the preference data, not the
choice of alignment loss** — consistent with the three methods being statistically close on
win-rate (Day 18).

**2. The dominant failure is stop-control.** `no_clean_stop` and `incomplete` are two symptoms of
one root cause: the model rarely emits a clean stop token, so it rambles — often into a second,
self-directed turn ("Yeh bahut acchi advice mil raha hai!") — until `max_new_tokens` truncates it
mid-sentence. That single behaviour accounts for most of `no_clean_stop`, `incomplete`, and a large
share of `repetition`. This is exactly the limitation predicted at the data-design stage: the
preference pairs were length-matched (chosen ≈ rejected in length), so DPO/KTO had **no "stop
cleanly" signal to learn from**. The error analysis confirms empirically that the data, not the
training, is the bottleneck.

**3. Eval-time decoding artifacts.** A minority of outputs contain token junk (`+lsi`, `spNet`,
`⏤⏤⏤`). The evaluation used plain greedy decoding without the foreign-token suppression and
allowlist sanitizer built for the preference data, so some artifacts leaked back in. This inflates
`language_issue`; it is cosmetic and fixable at inference by applying the sanitizer.

**4. Factual reliability is moderate, not solved.** ~15% of outputs contain a fabricated entity
(e.g. invented hotel/brand names) and ~3-4% repeat a confident unverified claim. Low enough to be a
secondary concern behind stopping, but a real limitation for any factual use — reinforcing the
low-stakes-only scope.

## What would actually fix the top issue

More DPO/IPO/KTO on the current data will **not** fix stopping — there is no signal for it in
length-matched pairs. The targeted fix is **stopping-specific preference data**: pairs where the
chosen response is a clean, cleanly-stopped answer and the rejected is the model's rambling
version. Pair that with applying the inference-time sanitizer to remove decoding artifacts. Both
are concrete next steps rather than open research.
