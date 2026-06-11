# RAMP v0 Consolidated Research Report

RAMP v0 is feature-complete as a research multi-stage classifier. The current evidence
now supersedes the earlier prompt-plus-activation headline: under cross-fitted,
leakage-free reviewed-label calibration, the selected runtime policy is prompt plus
input-embedding proximity. Activation, output, and session signals remain implemented
audit, research, post-generation, or escalation evidence until larger blind holdouts
justify positive runtime weight.

## Frozen v0.2 Input-Side Policy

- Policy artifact: `ramp_fusion_policy_v0.2`
- Supersedes: `ramp_fusion_policy_v0.1`
- Decision: `prompt_embedding_runtime_policy_pending_blind_holdout`
- Runtime threshold: `0.5`
- Prompt weight: `0.8`
- Activation weight: `0.0`
- Embedding runtime weight: `0.2`
- Output classifier: post-generation audit, no positive v0 runtime weight
- Session classifier: calibrated escalation/audit signal, no naive OR/max blocking
- Tool/action gate: reference deterministic gate, benchmark evaluation still pending

## Input-Side Reviewed Split Stability

| Condition | AUC mean | Recall mean | FPR mean | FP mean | FN mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prompt_only_calibrated` | 0.9458 | 0.9631 | 0.1238 | 11.6333 | 4.8000 |
| `prompt_embedding_calibrated` | 0.9627 | 0.9608 | 0.1085 | 10.2000 | 5.1000 |
| `prompt_activation_calibrated` | 0.9609 | 0.9569 | 0.1195 | 11.2333 | 5.6000 |
| `prompt_embedding_activation_calibrated` | 0.9641 | 0.9500 | 0.1078 | 10.1333 | 6.5000 |

Decision: prompt+embedding is the selected v0.2 runtime policy because it improves
AUROC, F1, accuracy, and FPR over prompt-only under the cross-fitted leakage-free
protocol, with a small recall tradeoff. Full prompt+embedding+activation has marginally
higher AUROC but lower recall/F1, so activation remains an audit/research signal pending
blind-holdout validation.

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

- Input embeddings are not a standalone decision feature, but prompt+embedding is the selected v0.2 input-side runtime policy pending blind holdout.
- Output classification does not improve the best input-side v0 fusion yet.
- Compact session evidence does not recover enough full-transcript signal yet.
- Naive OR/max fusion improves unsafe recall but increases false positives.
- Tool/action gating is implemented as a design pattern but lacks benchmark validation.

## Paper Status

This is a feature-complete research v0, not a paper-final claim. The next paper-grade
step is broader reviewed labeling, especially for output responses, session FPR/AUC,
and tool/action examples.
