# Signal Fragility Study: Experimental Design (Pre-Registration Draft)

Status: DRAFT v0.1. This document must be frozen (committed and tagged) BEFORE blind
labels are collected. Once frozen, the rung definitions, signal list, survival rule, and
metrics below may not change for the headline experiment; any change requires a new
versioned design document and must be reported as a deviation.

## Research Claim

Multi-signal safety fusion overstates signal value under naive evaluation. Many signals
that appear to earn runtime weight lose it under leakage-free, blind, and
distribution-shifted protocols. We introduce a signal-admission ladder that makes this
fragility measurable, and we quantify how much adaptive (hard-case-mined) labeling
inflates reported signal value relative to blind random labeling.

This reframes RAMP: not "a better classifier" and not primarily "a cheaper cascade", but
a protocol for deciding which safety signals deserve runtime authority.

Motivating example (already observed in this repository): the v0.1 frozen policy assigned
activation weight 0.75 with mean AUROC 0.9939; the result was an artifact of an
activation probe evaluated largely on its own training rows. Under the cross-fitted
leakage-free protocol the same signal's fused AUROC fell to 0.9609 and it was demoted to
zero runtime weight (`data/fusion_policy/ramp_fusion_policy_v0_2.json`).

## Signals Under Study

| Signal | Score column | Learned on eval data? |
| --- | --- | --- |
| Prompt classifier (Qwen3Guard) | `prompt_risk_score` | No (frozen external model) |
| Embedding proximity (GPT-OSS centroids) | `embedding_prior_score` | Partially (centroids from benchmark corpus, not reviewed labels) |
| Activation probe (GPT-OSS layer 19) | `activation_probability` | Yes (trained on reviewed labels) |
| Output classifier (Qwen3Guard) | `output_risk_score` | No (frozen external model, small eval set) |
| Session scoring | session-level scores | Mixed (formula + frozen model) |
| Tool/action gate | deterministic gate | Not yet evaluated |

v0 of the ladder covers prompt, embedding, and activation (the signals with complete
feature tables). Output and session join the ladder when their feature tables cover the
same rows or a declared session-level analogue. Tool/action remains "not evaluated"
until a benchmark exists; an honest empty cell beats a rushed one.

## The Evaluation Ladder

Each rung evaluates the same signal combinations with strictly less leakage than the rung
below it. A signal's reported value should be monotonically non-increasing in honesty if
the fragility claim is true; the interesting data is where each signal's value collapses.

| Rung | Weights/threshold | Learned signals (probe) | Evaluation rows |
| --- | --- | --- | --- |
| 1. naive | tuned on all rows | trained on all rows, scored in-sample | same rows used for tuning |
| 2. split | tuned on calibration half | trained on ALL rows (in-sample for everyone) | holdout half |
| 3. cross-fit | tuned on calibration half using out-of-fold probe scores | retrained per split on calibration rows only; holdout scored out-of-split | holdout half |
| 4. blind | tuned on full adaptive set (out-of-fold probe scores) | trained on adaptive set only | blind random reviewed set (never seen) |
| 5. shifted | tuned on other sources (out-of-fold probe scores) | trained on other sources only | held-out source |

Rung 2 deliberately reproduces the v0.1 mistake (honest weight split, leaky learned
signal) because it mirrors common published practice. Rung 3 is the current v0.2
protocol. Rungs 1-3 and 5 are runnable today; rung 4 requires the blind label set.

Implementation: `scripts/evaluate_signal_survival_ladder.py`.

## Survival Rule (v0, pre-registered)

For each rung, each signal is judged by its calibrated combination against the
prompt-only calibrated baseline on the same rung:

- embedding -> `prompt_embedding_calibrated`
- activation -> `prompt_activation_calibrated`
- full fusion -> `prompt_embedding_activation_calibrated`

Verdicts per rung (using holdout-mean metrics):

- `survives`: mean AUROC delta > 0 AND mean F1 delta > 0 vs prompt-only
- `mixed`: exactly one of the two deltas > 0
- `fails`: neither delta > 0
- `pending` / `skipped`: rung not yet runnable for that signal

This rule is intentionally simple and symmetric. Statistical significance on the blind
rung is assessed with paired tests (per-row paired bootstrap on AUROC delta and F1
delta, 10,000 resamples, 95% CI excluding zero), because at the current data scale
split-mean comparisons are underpowered (observed AUROC stdev across splits is ~0.013;
the embedding effect of interest is ~0.017).

Shifted-rung verdicts use holdout-row-weighted means across held-out sources, not
unweighted source means: sources differ by an order of magnitude in size and class
balance (e.g. do_not_answer holds only 5 safe rows, making its FPR granularity 0.2),
so an unweighted mean lets the smallest, most degenerate source dominate the verdict.
Per-source results are always reported alongside the weighted aggregate, because
source-dependence of signal value is itself a finding.

## Datasets

| Set | Role | Size | Status |
| --- | --- | --- | --- |
| Adaptive reviewed set | rungs 1-3, 5; calibration for rung 4 | 448 binary rows | exists (`data/reviewed/`) |
| Blind reviewed set | rung 4 evaluation; adaptive-bias quantification | target 500 minimum, 1,000 preferred | v0.1 batch: 500 rows sampled, awaiting labels |
| Benchmark corpus | extraction, shifted-rung populations | 27,718 rows | exists |

Blind batch v0.1 is fixed at 500 rows to match prior review-batch throughput. If review
budget allows, a second 500-row blind batch (same sampler, new seed, prior exclusions)
may be appended BEFORE the headline run; once the headline ladder runs on blind labels,
the blind set is consumed for confirmatory purposes per the single-use rule.

### Blind set requirements (operational definition of "blind")

1. Random sample from the frozen benchmark corpus, stratified by source and domain to
   match the corpus distribution (NOT the adaptive set's distribution), excluding rows
   already reviewed in any prior batch.
2. The sample is drawn and committed (ids + checksums) BEFORE any model scores it.
3. Review CSVs given to reviewers contain NO model scores, NO bucket assignments, NO
   severity hints — prompt text and source metadata only.
4. Labels are frozen with checksums before a single ladder evaluation runs on them.
5. The same blind set is used at most once for the headline table. Subsequent
   iterations require a fresh blind sample or must be reported as post-hoc.

### Adaptive-vs-blind comparison (the novel quantification)

Run identical rung-3 calibration on (a) the adaptive set and (b) the blind set, and
report, per signal: AUROC/F1/FPR deltas, selected weights, and survival verdicts. The
difference is a direct measurement of how much hard-case-mined labeling distorts
reported signal value. The sampling mechanism of each adaptive batch (disagreement
mining, stability-error slices, activation-miss targeting) is documented in
`docs/experimental-design.md` and must be summarized in `data/reviewed/README.md`.

## External Guards

To answer "why not just use a stronger guard": evaluate Qwen3Guard-0.6B (current),
Qwen3Guard-4B, and Llama Guard 3 or WildGuard as prompt-signal alternatives on the blind
set and external corpus. Each strong guard also enters the ladder as a `prompt_risk_score`
variant, which tests whether internal signals add anything once the front-door classifier
is strong.

## Supporting Section: Anytime/Cost

After the survival table exists, a small cost/latency experiment shows that signals
which survive admission can be scheduled efficiently (early exit on decisive cheap
evidence, escalation on unresolved risk states). This is deliberately not the headline.

## Deviations Log

(Empty. Append dated entries here if anything above changes after freeze.)
