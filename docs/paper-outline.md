# RAMP Paper Outline

## Working Title

RAMP: A Multi-Stage Safety Classifier for Accumulating Prompt, Internal, Output, and Session Risk
Signals

## Abstract Shape

Large language model safety systems often rely on one-shot prompt or response classifiers. RAMP
tests a multi-stage alternative: accumulate multiple partial signals into an auditable risk state.
The v0 reference implementation evaluates prompt classification, GPT-OSS input-embedding
proximity, GPT-OSS activation probes, output classification, session classification, and tool/action
gating. The current evidence supports prompt-plus-activation as the strongest reviewed-label
runtime core, while embedding, output, and session signals provide useful audit or escalation
evidence but do not yet justify positive v0 blocking weight.

## 1. Introduction

- Safety classification is often framed as a single prompt or response decision.
- Real deployments have staged evidence: prompt text, internal model state, generated output,
  session trajectory, and tool/action proposals.
- RAMP asks whether accumulating these signals improves safety decisions on hard cases.

## 2. Related Work

- Open-source guard classifiers such as Qwen3Guard, Llama Guard, WildGuard, and ShieldGemma.
- Model-internal safety probes and activation-based monitoring.
- Embedding or centroid-based semantic risk retrieval.
- Multi-turn and jailbreak/session benchmarks such as R-Judge, MHJ, and SafeDialBench.
- Constitutional classifiers and policy-driven safety filters as prior art.

## 3. Method

RAMP is a staged classifier. Each feature emits a versioned score, label, confidence, metadata, and
provenance record.

Stages:

- Prompt classifier: Qwen3Guard prompt risk.
- Input embeddings: GPT-OSS input-embedding proximity to harmful and benign near-neighbor
  centroids.
- Activation probe: linear probe over GPT-OSS hidden states, currently layer 19.
- Output classifier: Qwen3Guard over generated response text.
- Session classifier: compact-state and full-transcript session evidence.
- Tool/action gate: deterministic action-level risk pattern for agent systems.

Fusion policy:

- Primary v0 runtime score: prompt `0.25`, activation `0.75`, embedding `0.00`, threshold `0.53`.
- Severity floor: high-severity prompt findings can raise the final score to a configured floor.
- Output and session are audit/escalation signals in v0, not positive blocking weights.

## 4. Data

- Comprehensive benchmark-derived corpus: 27,718 rows for embeddings and activations.
- Reviewed prompt disagreement set: 448 binary reviewed rows after v0.4 expansion.
- Output eval set: 134 prompt/response rows.
- Session evals:
  - R-Judge: 555 sessions, with safe and unsafe labels.
  - MHJ: 496 unsafe-only multi-turn jailbreak sessions.
  - SafeDialBench: 4,053 unlabeled/ambiguous sessions for mining and future review.

## 5. Experiments

Input-side calibration:

- Compare prompt-only, prompt+embedding, prompt+activation, and prompt+embedding+activation.
- Select weights by deterministic split-stability rather than hand tuning.

Output classifier:

- Score generated responses.
- Evaluate whether output risk improves input-side fusion.

Session classifier:

- Compare Qwen over compact session evidence, Qwen over full transcript, and single-turn max.
- Evaluate R-Judge for AUC/FPR/recall.
- Evaluate MHJ as unsafe-recall stress only.

## 6. Results

Current frozen policy:

- Prompt+activation is selected for v0 runtime.
- Full prompt+embedding+activation ties AUC but has higher FPR and more hard-benign false
  positives in reviewed split stability.

Output result:

- Output classifier is implemented.
- Output-inclusive fusion does not improve the best input-side v0 result.
- Output remains useful as post-generation audit/compliance evidence.

Session result:

- Full-transcript session scoring catches single-turn misses.
- Compact session evidence is currently too lossy.
- Naive max/OR session fusion raises false positives and should not be used as a direct block.

## 7. Negative Results

These findings are part of the contribution:

- Input embeddings are not reliable as a standalone classifier.
- Embedding should not get positive v0 runtime weight without stronger reviewed-label lift.
- Output classification did not improve v0 fusion on the current response set.
- Compact session state did not recover enough full-transcript signal.
- Full-transcript session signal is useful, but naive OR/max fusion is false-positive heavy.
- Tool/action gating still lacks benchmark validation.

## 8. Limitations

- Reviewed labels are targeted hard cases, not a broad production distribution.
- Some datasets are unsafe-only or unlabeled, limiting AUC/FPR claims.
- Activation probes are trained/evaluated on extracted GPT-OSS representations, not a production
  deployment service.
- Output and session results are v0-sized and need more human-reviewed coverage.
- The current policy is a research reference design, not a production moderation guarantee.

## 9. Conclusion

RAMP is feature-complete as a v0 reference implementation. The strongest current finding is not
that every signal deserves runtime weight, but that staged evidence gives a disciplined way to test
which signals add value and which should remain audit/escalation evidence until proven otherwise.

## Appendix Candidates

- Full taxonomy and benign near-neighbor design.
- Artifact registry and provenance.
- Reproducibility commands.
- Calibration search details.
- Error slices by domain, severity, and source.
