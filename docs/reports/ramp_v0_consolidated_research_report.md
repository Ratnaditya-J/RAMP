# RAMP v0 Consolidated Research Report

RAMP v0 is feature-complete as a research multi-stage classifier. The current evidence
supports a prompt-plus-activation primary runtime policy, with embedding, output, and
session signals retained as audit, taxonomy, post-generation, or escalation signals until
larger reviewed datasets justify positive runtime weight.

## Frozen v0 Policy

- Policy artifact: `ramp_multistage_policy_v0.1`
- Decision: `prompt_activation_primary_with_audit_and_escalation_signals`
- Runtime threshold: `0.53`
- Prompt weight: `0.25`
- Activation weight: `0.75`
- Embedding runtime weight: `0.0`
- Output classifier: post-generation audit, no positive v0 runtime weight
- Session classifier: calibrated escalation/audit signal, no naive OR/max blocking
- Tool/action gate: reference deterministic gate, benchmark evaluation still pending

## Input-Side Reviewed Split Stability

| Condition | AUC mean | Recall mean | FPR mean | FP mean | FN mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prompt_activation_calibrated` | 0.9939 | 0.9626 | 0.0397 | 3.7333 | 4.8667 |
| `prompt_embedding_activation_calibrated` | 0.9939 | 0.9633 | 0.0450 | 4.2333 | 4.7667 |

Decision: full prompt+embedding+activation does not beat prompt+activation for the
frozen v0 runtime policy. Embedding remains a useful taxonomy and audit signal.

## Output Classifier

| Condition | AUC mean | Recall mean | FPR mean |
| --- | ---: | ---: | ---: |
| `prompt_embedding_activation_calibrated` | 0.9864 | 0.9443 | 0.0697 |
| `prompt_activation_output_calibrated` | 0.9858 | 0.9457 | 0.0773 |
| `prompt_embedding_activation_output_calibrated` | 0.9851 | 0.9429 | 0.0773 |

Decision: output scoring is implemented and useful for post-generation audit, but
the v0 prompt/response set does not justify positive fusion weight.

## Session Classifier

R-Judge labeled session comparison at threshold `0.55`:

| Condition | AUC | Recall | FPR | Single-turn FNs caught |
| --- | ---: | ---: | ---: | ---: |
| `single_turn_max` | 0.5471 | 0.7413 | 0.7695 | 0 |
| `compact_session_classifier` | 0.5299 | 0.1154 | 0.0558 | 1 |
| `full_transcript_session_classifier` | 0.6105 | 0.5839 | 0.3680 | 14 |
| `max_session_signal` | 0.5570 | 0.7937 | 0.8104 | 15 |

MHJ unsafe-only stress comparison at threshold `0.55`:

| Condition | Recall | Single-turn FNs caught |
| --- | ---: | ---: |
| `single_turn_max` | 0.7984 | 0 |
| `compact_session_classifier` | 0.4133 | 4 |
| `full_transcript_session_classifier` | 0.6956 | 24 |
| `max_session_signal` | 0.8508 | 26 |

Decision: full-transcript session scoring shows real session signal, but compact
state is too lossy and naive max/OR fusion raises false positives. Use session
classification as escalation/audit in v0.

## Negative And Limited Results

- Input embeddings are not a standalone decision feature in v0.
- Output classification does not improve the best input-side v0 fusion yet.
- Compact session evidence does not recover enough full-transcript signal yet.
- Naive OR/max fusion improves unsafe recall but increases false positives.
- Tool/action gating is implemented as a design pattern but lacks benchmark validation.

## Paper Status

This is a feature-complete research v0, not a paper-final claim. The next paper-grade
step is broader reviewed labeling, especially for output responses, session FPR/AUC,
and tool/action examples.
