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

## Hardening Result 1: MLP probe (the "weak linear probe" defense)

Re-running the full ladder with an MLP activation probe (one hidden layer) instead of the
linear probe leaves the verdicts essentially unchanged:

| Signal | crossfit (linear -> mlp) | blind (linear -> mlp) | shifted |
| --- | --- | --- | --- |
| activation | mixed +0.016 -> mixed +0.016 | mixed +0.034* -> mixed +0.028* | fails (both) |
| embedding | survives (both) | mixed -0.020 (both) | survives (both) |

Conclusion: the activation findings — significant lift on blind, collapse under shift — are
NOT an artifact of using a weak linear probe; a stronger probe behaves the same. This closes
the predictable "you used the weakest probe" objection.

## Hardening Result 2: external guards ("why not a bigger guard?")

Two independent-lineage guards scored the eval prompts (not Qwen, not the Claude judge, not
GPT-OSS).

Dedicated guard, operating-point comparison (binary verdict; recall/FPR/F1 at its own point):

| Set | Guard | Recall | FPR | F1 |
| --- | --- | ---: | ---: | ---: |
| Adaptive (human) | Llama Guard 4 12B | 0.623 | 0.048 | 0.752 |
| Adaptive (human) | Qwen3Guard 0.6B | 0.973 | 0.133 | 0.941 |
| Blind (judge) | Llama Guard 4 12B | 0.925 | 0.207 | 0.827 |
| Blind (judge) | Qwen3Guard 0.6B | 1.000 | 0.360 | 0.786 |

Strong general-model guard as the front door (Gemini 2.5 Pro, continuous), prompt-only AUROC:

| Front door | AUC adaptive (human) | AUC blind (judge) |
| --- | ---: | ---: |
| Qwen3Guard 0.6B | 0.945 | 0.850 |
| Gemini-2.5-pro-as-guard | 0.791 | 0.976 |

Conclusions:
- Bigger/stronger is not better as a guard. A 12B dedicated guard misses 38% of unsafe
  prompts on the human-labeled hard-case set; a frontier general model used as a guard is
  worse than the 0.6B dedicated guard on human labels (AUC 0.79 vs 0.945).
- The Gemini-guard's near-perfect blind AUROC (0.976) is a CONFOUND, not a win: the blind
  labels are Claude-judge labels, so guard and labels are both frontier-LLM safety
  judgments sharing method variance. This is a concrete demonstration of why LLM-judge
  blind labels require human validation.
- Internal signals' apparent value depends on the front door: on top of the strong
  (label-aligned) guard they add ~0 on blind; on top of the same guard where it is weak
  (human-labeled), activation adds +0.145 AUROC. "Earning weight" is again protocol- and
  label-dependent.

## Hardening Result 3: inter-judge agreement on the blind labels

A second independent frontier judge (Gemini 2.5 Pro) relabeled the blind set. Agreement
with the Claude Opus 4.8 judge on the shared binary rows:

- n = 338, raw agreement = 0.944, Cohen's kappa = 0.887 ("almost perfect").
- Disagreement is near one-directional: Gemini flags slightly more as unsafe (unsafe rate
  0.476 vs 0.426); only 1 of 338 rows is Claude-unsafe / Gemini-safe.

This makes the silver labels more credible — they are not idiosyncratic to one model. BUT
high inter-LLM agreement does not rule out a shared LLM-vs-human bias (the guard confound
above shows frontier models can agree with each other and still drift from humans). So the
human audit remains the final check, now as a third leg rather than a single point of
failure.

## Hardening Result 4: human audit + the label-standard finding (the big one)

A human labeled a frozen, blinded 80-row subset of the blind set. Agreement with the LLM
judges, by rubric:

| Comparison | Literal rubric | Framing-inclusive rubric |
| --- | ---: | ---: |
| Human vs Claude | kappa 0.19 | kappa 0.57 |
| Human vs Gemini | kappa 0.29 | kappa 0.58 |
| Claude vs Gemini (inter-judge) | kappa 0.89 | kappa 0.81 |

Under the original (literal) rubric the silver labels FAILED the human audit (kappa 0.19-
0.29, "slight"/"fair") even though the two judges agreed with EACH OTHER almost perfectly
(0.89) — a textbook shared-LLM-vs-human bias. The disagreement was 100% one-directional:
the human counts adversarial/jailbreak framing, copyright reproduction, and harmful
ideation as unsafe; the judge rubric scored only the literal underlying request.

Re-running both judges under a framing-inclusive rubric that encodes the human standard
roughly tripled agreement (0.19 -> 0.57, 0.29 -> 0.58), into the range where humans agree
with each other on safety labeling. The residual disagreement is genuine borderline cases
(e.g. "Among Us" gameplay read as veiled violence) plus one human slip ("kissing in a
mirror" labeled unsafe) — not rubric failure.

**The label standard flips a headline finding.** Re-running the blind ladder on the
framing-inclusive (human-aligned) labels:

| Signal | blind dAUC, literal labels | blind dAUC, framing labels |
| --- | --- | --- |
| embedding | -0.0203 (ns) | +0.0268 (significant) |
| activation | +0.0338 (sig) | +0.0484 (significant) |

Under the corrected labels, embedding's blind contribution flips from "negative, not
significant" to "significantly positive" — so the earlier "signal reversal" (only
activation helps on blind) was itself an ARTIFACT of the labeling standard. Under labels
that match human judgment, BOTH embedding and activation significantly help on a blind
random sample. Apparent signal value is sensitive not just to evaluation protocol
(leakage / sampling / shift) but to the label standard too.

Caveat: re-validating on the same 80 audit rows that informed the rubric choice is
confirmatory, not fully independent. The framing-inclusive rubric was written from the
principle the human endorsed, not fitted to individual labels; a fresh second audit batch
under the agreed standard would make it airtight.

Signal Validity Cards (`docs/reports/ramp_signal_validity_cards_framing_v0_1.md`) are stable
to the label correction at the verdict tier: embedding and full_fusion =
`in_distribution_only`, activation = `leak_inflated`. They now carry
`blind_label_provenance=silver_llm_inter_judge_validated`, inter-judge kappa 0.81, and
human-audit kappa 0.57 in the residual-risks block.

## Next

- (Optional, for airtight blind claims) a fresh second human-audit batch under the
  framing-inclusive standard, for an independent kappa.
- (Paper-2 scope, not now) extend the ladder across multiple target models, probe families,
  and an external system.
