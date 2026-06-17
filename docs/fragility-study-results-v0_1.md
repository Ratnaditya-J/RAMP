# Fragility Study: v0.1 Results (blind rung filled)

Status: v0.1 results with the blind rung populated by an LLM judge (silver labels).
See `docs/fragility-study-design.md` for the pre-registered protocol. These numbers are
directional pending the human-audit subset (Cohen's kappa) that validates the judge labels.

## Labeling

- Blind set: 500 rows, random stratified sample from the 27,718-row corpus, all prior
  reviewed ids excluded (`data/reviewed/ramp_blind_review_batch_v0_1.csv`, frozen).
- Judge: `anthropic/claude-opus-4.8` via OpenRouter, temperature 0, strict rubric, JSON
  output. Independent lineage from both the prompt classifier (Qwen3Guard) and the
  embedding/activation source (GPT-OSS).
- Yield: 486/500 labeled, 14 parse errors (~3%, excluded). Binary-usable (safe/unsafe):
  369 rows (222 safe, 147 unsafe). Mean judge confidence 0.84.
- Artifacts: `data/reviewed/ramp_blind_review_batch_v0_1.judge_labeled.{jsonl,csv}`
  (checksummed). Labels are SILVER until human-audited.

## Survival Table

| Signal | naive | split | crossfit | blind | shifted |
| --- | --- | --- | --- | --- | --- |
| embedding | yes (+0.0170) | yes (+0.0183) | yes (+0.0173) | mixed (-0.0203 ns) | yes (+0.0061 ns) |
| activation | yes (+0.0540) | yes (+0.0524) | mixed (+0.0160) | mixed (+0.0338*) | no (-0.0109 ns) |
| full_fusion | yes (+0.0545) | yes (+0.0530) | yes (+0.0205) | mixed (-0.0203 ns) | mixed (+0.0015 ns) |

Deltas are AUROC vs prompt-only on the same rung. `*` = paired-bootstrap 95% CI excludes
zero (blind/shifted only); `ns` = CI includes zero. Verdict = mean-delta sign on AUROC and
F1 (pre-registered). "mixed" = exactly one of AUROC/F1 improves.

## Adaptive-vs-Blind Comparison (the headline)

Same cross-fit-style calibration; adaptive = crossfit holdout (human labels), blind =
LLM-judge holdout. Every condition degrades sharply, and the operating point calibrated on
adaptive data badly miscalibrates on blind data (FPR roughly triples).

| Condition | AUC adaptive | AUC blind | F1 adaptive | F1 blind | FPR adaptive | FPR blind |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prompt only | 0.9448 | 0.8501 | 0.9378 | 0.7861 | 0.1202 | 0.3604 |
| prompt+embedding | 0.9621 | 0.8298 | 0.9433 | 0.7967 | 0.1032 | 0.3378 |
| prompt+activation | 0.9608 | 0.8839 | 0.9354 | 0.7861 | 0.1191 | 0.3604 |
| prompt+emb+act | 0.9653 | 0.8298 | 0.9387 | 0.7967 | 0.1014 | 0.3378 |

Prompt-only AUROC falls 0.945 -> 0.850 and F1 0.938 -> 0.786 from the adaptive hard-case
set to the blind random sample. This is the central fragility result: apparent performance
measured on an adaptively mined set substantially overstates performance on a blind random
draw, and a threshold frozen on the adaptive set yields ~36% FPR on blind data.

## Signal Reversal (the nuance that makes it a real finding)

The protocol does not merely shrink every signal uniformly — the *ranking of which signal
helps* changes with the evaluation set:

- In-distribution (naive/split/crossfit): embedding is the consistent survivor; activation
  looks strong under leaky protocols and fades to "mixed" under cross-fitting.
- Blind random sample: activation is the ONLY signal with a statistically significant
  AUROC lift over prompt-only (+0.0338, CI [+0.0105, +0.0579]); embedding's AUROC
  contribution goes negative (-0.0203, ns) while it adds a small significant F1 gain.
- Under source shift: no internal signal is significant; all CIs include zero.

So "which internal signal earns its weight" is not a stable fact about the signal — it is
a function of the evaluation protocol and the evaluation distribution. That is a stronger
and more defensible claim than "all signals are fragile."

## Caveats (load-bearing)

1. Blind labels are LLM-judge silver labels. The prompt classifier is Qwen; the judge is
   Claude; the activation source is GPT-OSS. The prompt-only AUROC drop on blind partly
   reflects Qwen<->Claude label disagreement on random data, and the activation lift on
   blind may partly reflect GPT-OSS activations aligning with the Claude judge better than
   Qwen does. The adaptive-vs-blind gap therefore conflates sampling shift with a change in
   label source (human -> model).
2. The human-audit subset (~80 random blind rows, report Cohen's kappa vs the judge) is the
   pre-registered fix and is now the gating item for any paper claim built on the blind
   rung. If judge<->human agreement is high, the blind findings stand; if low, they are
   judge artifacts.
3. The calibrated "full fusion" selected embedding weight 0.2 and activation weight 0.0 on
   the adaptive set, so full_fusion == prompt+embedding on the blind holdout (identical
   numbers above). Activation's blind lift comes from the prompt+activation condition.

## Next

- Human-audit ~80 blind rows; report kappa; promote/qualify the blind rung accordingly.
- Add one strong external guard (Qwen3Guard-4B / Llama Guard 3) as a prompt-signal variant
  on the blind set to answer "why not a bigger guard?".
