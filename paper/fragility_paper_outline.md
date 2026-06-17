# Paper 1 Outline: An Admission Ladder for Safety Signals

Status: DRAFT outline. This is chapter one of a planned arc (see end). It is a methods +
empirical-findings paper, NOT a survey and NOT a "better classifier" paper.

- Primary venue: SaTML (archival, safety-branded, right audience).
- Fallback venue: NeurIPS / ICLR safety or red-teaming workshop.
- Instrument: RAMP. Subject: the evaluation discipline.

## Central Claim

The apparent value of a safety signal is largely an artifact of evaluation protocol.
Under the naive evaluation that is standard practice, several signals appear to earn
runtime weight; under a leakage-free admission ladder, some were illusions (activation)
while others survive (embedding) — and the inflation is large enough to flip a frozen
policy decision.

The contribution is the discipline that tells real signal value from illusory, plus the
measured finding that the protocol — not the signal — determines the verdict. RAMP is the
substrate it runs on, not the thing being sold.

## Why The Discrimination Matters (framing guardrail)

- Do NOT frame as "all safety signals are fragile." Embedding survives every runnable
  rung; that contradicts a blanket-fragility claim and is false to the data.
- A protocol that kills everything looks merely harsh. A protocol that kills activation,
  spares embedding, and renders full fusion mixed demonstrates *resolving power*. The
  mixed/survivor verdicts are the evidence the instrument works.
- Activation is the vivid casualty (the hook). Embedding is the credibility (the method
  is not nihilism). Adaptive-vs-blind is the cleanest novel axis (pending labels).

## Contributions (claim list)

1. The admission ladder: five rungs of strictly increasing evaluation honesty
   (naive -> split -> cross-fit -> blind -> shifted), each defined precisely, with a
   pre-registered survival rule.
2. An empirical demonstration that apparent signal value collapses monotonically with
   protocol honesty for some signals and persists for others — i.e., the protocol, not
   the signal, sets the verdict.
3. A quantification of adaptive-vs-blind sampling inflation for safety-signal weight
   selection (the part not already in the literature). [PENDING blind labels]
4. A concrete consequence: under honest evaluation the frozen runtime policy changes
   (activation weight 0.75 -> 0.0; the v0.1 -> v0.2 flip), showing the protocol is not
   academic — it changes deployed decisions.

## Section Outline

### 1. Introduction
- Multi-signal / cascade safety systems are proliferating (prompt guards + internal
  probes + output + session + tool gates). Apparent per-signal value is usually taken at
  face value from a single evaluation.
- Gap: no discipline that asks whether a signal's apparent value survives honest
  evaluation before it earns runtime authority.
- Our move: an admission ladder; demonstrate on a multi-signal system; report which
  signals survive and by how much the naive protocol inflated them.
- Bounded-scope disclosure up front: one target model (GPT-OSS), the signals RAMP has,
  one benchmark-derived corpus family. This is deliberate (chapter one); the multi-model,
  multi-probe, external-system version is future work.

### 2. Related Work
- Guard classifiers (Llama Guard, WildGuard, Qwen3Guard) — front-door baselines, not the
  contribution; we treat the prompt classifier as one signal.
- Multi-signal / cascade / runtime-monitor safety systems — the systems whose apparent
  signal value our protocol scrutinizes.
- Internal-representation safety signals: embedding proximity, activation probes,
  probe-based safety monitoring (incl. recent dynamic/multi-layer probes). Position: we
  test linear probes honestly; richer probes are explicitly out of scope (paper 2).
- Evaluation methodology / leakage: data leakage in ML-based science; probing control
  tasks. Position: we do NOT claim leakage-exists is novel; we contribute a protocol that
  operationalizes honesty for safety-signal admission and measures the inflation.
- Adaptive / red-teamed evaluation sets: widely used, optimism widely assumed but rarely
  quantified; our adaptive-vs-blind comparison puts a number on it.

### 3. The Admission Ladder (core method)
- Signals and combinations under study; prompt-only as the baseline every signal must beat.
- The five rungs, each defined precisely (reproduce from docs/fragility-study-design.md):
  - naive: weights + probe tuned/scored in-sample on all rows, evaluated on same rows.
  - split: honest weight split, but probe trained on all rows (reproduces common
    published practice and the v0.1 leakage pattern).
  - cross-fit: per-split probe on calibration rows only; out-of-fold calibration scores;
    out-of-split holdout scores (the leakage-free protocol).
  - blind: calibrate on the adaptive set, evaluate on a blind random reviewed set the
    system never saw.
  - shifted: hold out one source at a time; train + tune on the rest.
- Survival rule (pre-registered): beat prompt-only on BOTH mean AUROC and mean F1 =
  survives; one = mixed; none = fails. Shifted uses holdout-row-weighted means.
- Significance on single-evaluation rungs (blind, per-source shifted): per-row paired
  bootstrap on AUROC/F1 deltas, 95% CI excluding zero. [implement before headline run]
- Pre-registration and blinding: design doc frozen + tagged before blind labels; blind
  reviewer CSV carries no scores/buckets/domains/severity; selection manifest + checksums
  committed before labeling. (Reproducibility asset, not just rigor theater.)

### 4. Experimental Setup (the instrument)
- RAMP signals: prompt (Qwen3Guard), embedding proximity (GPT-OSS centroids), activation
  probe (GPT-OSS layer 19, linear), [output/session as extensions if feasible].
- Corpus: 27,718 benchmark-derived rows across four sources (WildGuardMix-dominated);
  taxonomy domains. Honest note on source imbalance.
- Reviewed sets: adaptive 448 binary rows (sampling provenance disclosed: disagreement
  mining, error slices, activation-miss targeting); blind 500-row random sample.
- What each signal is and why it is plausibly a safety signal (so the casualties land).

### 5. Results
- Table 1 (centerpiece): the survival table — signals x rungs, verdicts + AUROC deltas.
  Blind column now filled with LLM-judge silver labels (see docs/fragility-study-results-v0_1.md):
    embedding:   naive +.017 / split +.018 / crossfit +.017 / blind -.020(mixed,ns) / shifted +.006(ns) -> survives in-dist only
    activation:  naive +.054 / split +.052 / crossfit +.016(mixed) / blind +.034(mixed,*) / shifted -.011(fail)
    full_fusion: naive +.055 / split +.053 / crossfit +.021 / blind -.020(mixed,ns) / shifted +.002(mixed,ns)
  Headline nuance: the signal that helps REVERSES across protocol/distribution — embedding
  is the in-distribution survivor, but activation is the only signal with a significant
  AUROC lift on the blind random sample. "Which signal earns its weight" is protocol- and
  distribution-dependent, not a stable property of the signal.
- Figure 1 (money figure): activation's metric trajectory across rungs — the collapse
  curve from naive to shifted. The single most legible artifact in the paper.
- The activation narrative: 0.99 AUROC / weight 0.75 under naive eval -> evaporates under
  honesty. The opening example.
- The embedding narrative: modest but persistent; survives even cross-distribution. The
  method is discriminating, not destroying.
- The policy-flip consequence: v0.1 frozen policy (activation 0.75, AUROC 0.9939) vs v0.2
  (embedding 0.2, activation 0.0) — honest evaluation changed the deployed decision.
- Adaptive-vs-blind: DONE (silver labels). Large gap — prompt-only AUROC 0.945 -> 0.850,
  F1 0.938 -> 0.786, FPR 0.120 -> 0.360 from adaptive hard-case set to blind random sample.
  This is outcome (a): adaptive mining substantially overstates apparent performance, and a
  threshold frozen on adaptive data miscalibrates badly (~36% FPR) on blind data. Caveat:
  the gap conflates sampling shift with a human->model label-source change; the human audit
  disentangles this.
- Per-source shifted detail (Table 3): source-dependence of signal value is itself a
  finding (e.g., embedding helps on WildGuardMix under shift, not on do_not_answer).

### 6. Discussion
- Apparent signal value is protocol-dependent; reporting a single naive number is
  insufficient for safety-signal admission.
- The discrimination (one casualty, one survivor, one mixed) is the evidence the protocol
  resolves real from illusory rather than penalizing uniformly.
- Recommendation to the field: report signal value across honesty rungs, disclose
  adaptive sampling, and require out-of-split / cross-distribution evidence before a
  signal earns runtime weight.

### 7. Limitations and Roadmap (the arc, stated as a feature)
- One target model; linear probes only; one corpus family; modest N; adaptive + single
  blind set. Each is a deliberate boundary of chapter one.
- Paper 2: run the same ladder across multiple target models, multiple probe families,
  and reproduce the inflation on an external published system -> "leakage in safety eval
  is systemic" as a field claim.
- Paper 3: propose a signal/fusion that survives the full ladder, validated end-to-end
  (negative program -> positive result).

### 8. Conclusion
- A reusable admission discipline; a demonstration that protocol sets the verdict; a
  changed policy as proof of consequence; an explicit program it opens.

## Figures and Tables
- Table 1: survival table (centerpiece).
- Figure 1: activation collapse curve across rungs.
- Table 2: adaptive-vs-blind comparison. [PENDING]
- Table 3: per-source shifted breakdown.
- Figure 2 (optional): the v0.1 -> v0.2 policy flip.

## Reproducibility Statement
- Pre-registration: docs/fragility-study-design.md (tags signal-fragility-prereg-v0.1/v0.2).
- Committed reviewed labels + checksums under data/reviewed/; blind manifest frozen
  pre-labeling. Preprint baseline tag preprint-v0.2.
- Ladder: scripts/evaluate_signal_survival_ladder.py; blind sampler:
  scripts/build_blind_review_batch.py. Single-use blind rung.

## Open Decisions (resolve before drafting prose)
1. Include output/session as extension rungs, or hold for paper 2? (Default: mention as
   feasible extensions; keep the headline on prompt/embedding/activation.)
2. External strong guard (Qwen3Guard-4B / Llama Guard 3) in paper 1 or paper 2? (Default:
   one strong guard in paper 1 to answer "why not a bigger guard?"; full sweep is paper 2.)
3. Confirm SaTML as primary target and check the open cycle's deadline.
