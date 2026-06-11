# RAMP Artifact Registry

This registry records generated research artifacts that are intentionally not committed to Git.
Large corpora, model vectors, activation tensors, archives, and centroid artifacts live under
`.artifacts/` or RunPod storage and are ignored by the repository.

The registry is the source of record for what was generated, how it was generated, and what
limitations are known at the time of the run.

## GPT-OSS Comprehensive Extraction v0

| Field | Value |
| --- | --- |
| Artifact bundle | `ramp-gpt-oss-comprehensive-artifacts-v0.tgz` |
| Local path | `.artifacts/runpod/comprehensive/ramp-gpt-oss-comprehensive-artifacts-v0.tgz` |
| Source environment | RunPod GPU pod |
| GPU | NVIDIA H200 |
| Model | `openai/gpt-oss-20b` |
| Corpus rows | 27,718 |
| Vector dimension | 2,880 |
| SHA256 | `5a14455573e06d92c79d216cf7c0404531ec369555e7cb8334ac2457bd06a103` |
| Archive verification | `gzip -t` passed locally after copy from RunPod |

Bundle contents:

```text
ramp-artifacts/corpora/ramp_benchmark_comprehensive_v0.jsonl
ramp-artifacts/embeddings/gpt_oss_20b_input_embedding_v0_1/ramp_benchmark_comprehensive_v0.input_embeddings.jsonl
ramp-artifacts/activations/gpt_oss_20b_hidden_state_v0_1/ramp_benchmark_comprehensive_v0.layer_12.jsonl
ramp-artifacts/activations/gpt_oss_20b_hidden_state_v0_1/ramp_benchmark_comprehensive_v0.layer_19.jsonl
ramp-artifacts/activations/gpt_oss_20b_hidden_state_v0_1/ramp_benchmark_comprehensive_v0.layer_final.jsonl
```

Extraction roles:

| File | Representation | Purpose |
| --- | --- | --- |
| `ramp_benchmark_comprehensive_v0.input_embeddings.jsonl` | `input_embedding` | Source of record for input-embedding centroid construction. |
| `ramp_benchmark_comprehensive_v0.layer_12.jsonl` | `hidden_state` | Mid-layer activation probe candidate. |
| `ramp_benchmark_comprehensive_v0.layer_19.jsonl` | `hidden_state` | Late-layer activation probe candidate. |
| `ramp_benchmark_comprehensive_v0.layer_final.jsonl` | `hidden_state` | Final hidden-state probe/ablation candidate. |

The input embedding file uses the project definition of embeddings: GPT-OSS token IDs mapped
through `model.get_input_embeddings()(input_ids)`, followed by attention-mask mean pooling and
L2 normalization.

Hidden-state files are activation artifacts, not the primary embedding-centroid source.

## Activation Probe Comparison v0.1

| Field | Value |
| --- | --- |
| Comparison report | `.artifacts/activation_probes/ramp_activation_probe_layer_comparison_v0_1.json` |
| Selected probe | `.artifacts/activation_probes/ramp_activation_probe_layer_19_v0_1.json` |
| Probe family | `linear_activation_probe_v0.1` |
| Training data | GPT-OSS hidden-state activation files from comprehensive extraction v0 |
| Selection rule | Highest recall at or below 5% false-positive rate, then AUC |
| Selected layer | `19` |

Layer comparison:

| Layer | AUC | Recall at <=5% FPR | FPR | Selected threshold |
| --- | ---: | ---: | ---: | ---: |
| `12` | 0.9926 | 0.9766 | 0.0500 | 0.22 |
| `19` | 0.9953 | 0.9869 | 0.0466 | 0.18 |
| `final` | 0.9940 | 0.9750 | 0.0494 | 0.09 |

The input-embedding linear baseline reached 0.4265 recall at its conservative low-FPR operating
point. This makes the activation-probe result the stronger current evidence for internal model
signals.

Holdout validation reports:

```text
.artifacts/activation_probes/ramp_activation_probe_layer_12_source_holdouts_v0_1.json
.artifacts/activation_probes/ramp_activation_probe_layer_12_domain_holdouts_v0_1.json
.artifacts/activation_probes/ramp_activation_probe_layer_19_source_holdouts_v0_1.json
.artifacts/activation_probes/ramp_activation_probe_layer_19_domain_holdouts_v0_1.json
.artifacts/activation_probes/ramp_activation_probe_layer_final_source_holdouts_v0_1.json
.artifacts/activation_probes/ramp_activation_probe_layer_final_domain_holdouts_v0_1.json
```

Layer 19 remains the best current layer under domain-held-out validation, with mean recall 0.9758
at the train-selected threshold and mean AUC 0.9851 on domains that contain both safe and unsafe
labels. Source-held-out validation exposes dataset-family shift: `do_not_answer` has high safe
false-positive rate and `wildguardmix` unsafe recall drops. Treat this as a strong research signal,
not yet a standalone decision feature.

## Cumulative Internal-Signal Evaluation v0.1

| Field | Value |
| --- | --- |
| Evaluation report | `.artifacts/internal_signal_eval/ramp_internal_signal_ablation_v0_1.json` |
| Markdown summary | `.artifacts/internal_signal_eval/ramp_internal_signal_ablation_v0_1.md` |
| Feature table | `.artifacts/internal_signal_eval/ramp_internal_signal_feature_table_v0_1.jsonl` |
| Embedding input | Centered, domain-conditioned input-embedding centroid scores |
| Activation input | Layer 19 activation probe probabilities |
| Fusion rule | Fixed weighted sum: 0.25 embedding prior, 0.75 activation evidence |

This report evaluates cumulative value, not signal replacement. Embedding proximity is treated as
the early semantic prior and activation probability as later internal-state evidence.

Current ablation result:

| Ablation | AUC | Recall at <=5% FPR | FPR |
| --- | ---: | ---: | ---: |
| embedding only | 0.9465 | 0.7870 | 0.0467 |
| activation only | 0.9959 | 0.9876 | 0.0494 |
| cumulative fixed fusion | 0.9953 | 0.9859 | 0.0465 |

## Prompt Classifier Batch Evaluation v0.1

| Field | Value |
| --- | --- |
| Prompt score artifact | `.artifacts/runpod/prompt_scores_qwen3guard/extracted/ramp-artifacts/prompt_scores/ramp_benchmark_comprehensive_v0.qwen3guard_0_6b_prompt_scores.jsonl` |
| Cumulative report | `.artifacts/cumulative_signal_eval/ramp_qwen3guard_prompt_internal_signal_ablation_v0_1.json` |
| Markdown summary | `.artifacts/cumulative_signal_eval/ramp_qwen3guard_prompt_internal_signal_ablation_v0_1.md` |
| Feature table | `.artifacts/cumulative_signal_eval/ramp_qwen3guard_prompt_internal_signal_feature_table_v0_1.jsonl` |
| Prompt model | `Qwen/Qwen3Guard-Gen-0.6B` |
| Corpus rows joined | 27,718 |

Prompt-label cross-tab:

| Corpus label | Qwen label | Rows |
| --- | --- | ---: |
| `unsafe` | `Unsafe` | 10,720 |
| `unsafe` | `Safe` | 7,953 |
| `unsafe` | `Controversial` | 342 |
| `safe` | `Unsafe` | 5,116 |
| `safe` | `Safe` | 2,004 |
| `safe` | `Controversial` | 1,583 |

Prompt-plus-internal ablation:

| Ablation | AUC | Recall at <=5% FPR | FPR |
| --- | ---: | ---: | ---: |
| prompt only | 0.4520 | 0.0000 | 0.0000 |
| prompt + embedding | 0.6504 | 0.5395 | 0.0483 |
| prompt + activation | 0.7424 | 0.5759 | 0.0444 |
| prompt + embedding + activation | 0.8591 | 0.5753 | 0.0410 |

Known limitation: this is not yet a clean prompt-classifier benchmark. The comprehensive corpus is
span-derived and intentionally skewed toward harmful spans and hard benign near-neighbors. It is
excellent for internal-signal extraction, centroid construction, and probe training, but it should
not be treated as final prompt-level moderation ground truth.

Current interpretation: the Qwen3Guard run is primarily a label-audit artifact. It identifies rows
and source families where corpus labels and prompt-level classifier judgments disagree. The next v0
step is to build an audited prompt-evaluation subset and optionally add a second open classifier as
an agreement/disagreement cross-check.

## Output Classifier Eval Set v0.1

| Field | Value |
| --- | --- |
| Prompt/response JSONL skeleton | `.artifacts/output_eval/ramp_output_eval_set_v0_1.jsonl` |
| Generated response JSONL | `.artifacts/output_eval/ramp_output_eval_set_v0_1.generated.jsonl` |
| Reviewer/generation CSV | `/Users/ratnaditya/Documents/ramp_output_eval_set_v0_1.csv` |
| Output scores | `.artifacts/output_eval/ramp_output_scores_qwen3guard_v0_1.jsonl` |
| Joined output feature table | `.artifacts/output_eval/ramp_prompt_embedding_activation_output_feature_table_v0_1.jsonl` |
| Refined calibration report | `.artifacts/output_eval/ramp_prompt_embedding_activation_output_calibration_refined_v0_1.json` |
| Refined calibration summary | `.artifacts/output_eval/ramp_prompt_embedding_activation_output_calibration_refined_v0_1.md` |
| Summary | `.artifacts/output_eval/ramp_output_eval_set_v0_1.summary.json` |
| Source reviewed CSV | `/Users/ratnaditya/Documents/ramp_prompt_label_review_combined_v0_1_v0_2_v0_4_cleaned.csv` |
| Source feature table | `.artifacts/cumulative_signal_eval/ramp_qwen3guard_prompt_internal_signal_feature_table_expanded_reviewed_activation_v0_1.jsonl` |
| Rows | 134 |
| Output scoring model | `Qwen/Qwen3Guard-Gen-0.6B` |

Selection mix:

| Selection bucket | Rows |
| --- | ---: |
| `high_risk_true_positive` | 60 |
| `low_risk_true_negative` | 60 |
| `prompt_activation_false_negative` | 9 |
| `prompt_false_positive_hard_benign` | 5 |

Domain mix:

| Domain | Rows |
| --- | ---: |
| `regulated_advice` | 25 |
| `weapons_and_physical_violence` | 25 |
| `nonviolent_illegal_activity` | 21 |
| `ip_and_content_rights` | 17 |
| `cyber_abuse` | 16 |
| `self_harm_and_wellbeing` | 9 |
| `child_safety` | 8 |
| `cbrn_and_hazardous_materials` | 5 |
| `privacy_identity_and_secrets` | 4 |
| `sexual_safety_and_content` | 3 |
| `agent_tool_and_system_integrity` | 1 |

The initial skeleton intentionally left `output_text` empty and marked all rows
`output_review_status=needs_generation`. Responses were then generated separately before output
classifier scoring. This prevents prompt text from being reused as response text and keeps the
output classifier experiment scientifically separate from input-side scoring.

Refined calibration result:

| Condition | AUC | Recall | FPR |
| --- | ---: | ---: | ---: |
| `prompt_only_calibrated` | 0.9617 | 0.9400 | 0.0515 |
| `prompt_embedding_calibrated` | 0.9777 | 0.9443 | 0.0697 |
| `prompt_activation_calibrated` | 0.9859 | 0.9457 | 0.0652 |
| `prompt_embedding_activation_calibrated` | 0.9864 | 0.9443 | 0.0697 |
| `prompt_output_calibrated` | 0.9616 | 0.9400 | 0.0591 |
| `prompt_activation_output_calibrated` | 0.9858 | 0.9457 | 0.0773 |
| `prompt_embedding_activation_output_calibrated` | 0.9851 | 0.9429 | 0.0773 |

Interpretation: in this v0 output-response set, the output classifier does not improve the best
input-side fusion. The strongest AUC remains `prompt + embedding + activation`, and adding output
raises FPR without improving recall in the refined run. Output remains valuable as a post-generation
audit and policy-compliance signal, but the current evidence does not justify assigning it positive
weight in the v0 cumulative fusion policy.

## Session-Risk Evaluation v0.1

| Field | Value |
| --- | --- |
| Source manifest | `data/session_corpus/source_manifest_v0_1.json` |
| R-Judge session corpus | `.artifacts/session_eval/ramp_session_eval_corpus_rjudge_v0_1.jsonl` |
| R-Judge flattened turns | `.artifacts/session_eval/ramp_session_eval_corpus_rjudge_v0_1.turns.jsonl` |
| Qwen turn scores | `.artifacts/session_eval/ramp_session_eval_corpus_rjudge_v0_1.qwen_turn_scores.jsonl` |
| Joined scored corpus | `.artifacts/session_eval/ramp_session_eval_corpus_rjudge_v0_1.qwen_scored.jsonl` |
| R-Judge evaluation report | `.artifacts/session_eval/ramp_session_risk_eval_rjudge_qwen_v0_2.json` |
| R-Judge evaluation summary | `.artifacts/session_eval/ramp_session_risk_eval_rjudge_qwen_v0_2.md` |
| MHJ scored corpus | `.artifacts/session_eval/ramp_session_eval_corpus_mhj_v0_1.qwen_scored.jsonl` |
| MHJ evaluation summary | `.artifacts/session_eval/ramp_session_risk_eval_mhj_qwen_v0_1.md` |
| MHJ single-turn misses | `.artifacts/session_eval/ramp_session_mhj_single_turn_misses_qwen_v0_1.md` |
| SafeDialBench scored corpus | `.artifacts/session_eval/ramp_session_eval_corpus_safedialbench_v0_1.qwen_scored.jsonl` |
| SafeDialBench top-risk candidates | `.artifacts/session_eval/ramp_session_safedialbench_top_risk_candidates_qwen_v0_1.md` |
| SafeDialBench review CSV | `/Users/ratnaditya/Documents/ramp_safedialbench_session_label_review_top_200_v0_1.csv` |

R-Judge import summary:

| Metric | Value |
| --- | ---: |
| Sessions | 555 |
| Safe sessions | 269 |
| Unsafe sessions | 286 |
| Flattened turn rows | 2,012 |

Evaluation summary:

| Condition | AUC | Threshold | Recall | FPR | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `single_turn_max` | 0.5471 | 0.55 | 0.7413 | 0.7695 | 212 | 207 | 62 | 74 |
| `session_accumulation` | 0.4910 | 0.55 | 0.0070 | 0.0074 | 2 | 2 | 267 | 284 |

Threshold-sweep summary:

| Condition | Operating point | Threshold | F1 | Recall | FPR |
| --- | --- | ---: | ---: | ---: | ---: |
| `single_turn_max` | best F1 | 0.0000 | 0.6801 | 1.0000 | 1.0000 |
| `single_turn_max` | FPR <= 0.25 | 0.7500 | 0.4269 | 0.3112 | 0.1561 |
| `session_accumulation` | best F1 | 0.1139 | 0.6810 | 1.0000 | 0.9963 |
| `session_accumulation` | FPR <= 0.25 | 0.3529 | 0.3505 | 0.2622 | 0.2491 |

MHJ stress result: MHJ is unsafe-only, so it cannot measure FPR or AUC. At threshold `0.55`,
`single_turn_max` catches `396` of `496` unsafe multi-turn jailbreak sessions (`0.7984` recall), but
the current `session_accumulation` formula catches only `38` (`0.0766` recall). This makes MHJ useful
for unsafe-recall and failure-mining, not for calibration.

SafeDialBench mining result: SafeDialBench imports `4,053` unlabeled sessions and `19,318` flattened
turn rows. It is currently a review/mapping corpus rather than a metrics corpus. The top `200`
Qwen-scored candidates are exported for review at
`/Users/ratnaditya/Documents/ramp_safedialbench_session_label_review_top_200_v0_1.csv`.

Interpretation: this is a benchmark-backed negative v0 result for the current session accumulator.
The formula does not add value on R-Judge when fed Qwen prompt-risk turn scores, and it is too
conservative for MHJ unsafe-recall. The likely mismatch is that this first pass scores isolated user
turn text. The next session experiment should use richer per-turn evidence: assistant output risk,
role-aware transcript scoring, tool/action arguments, and session-specific transition features.
Synthetic session corpora remain useful as mechanism smoke tests, but not as primary evidence.

### Session Classifier v2 Inputs

| Field | Value |
| --- | --- |
| Compact state feature | `src/ramp/features/session_state_risk.py` |
| Input builder | `scripts/build_session_classifier_inputs.py` |
| Session classifier evaluator | `scripts/evaluate_session_classifier_scores.py` |
| Deterministic state evaluator | `scripts/evaluate_session_state_risk.py` |
| RunPod input bundle | `.artifacts/session_eval/runpod_bundle/ramp_session_classifier_inputs_v0_1.tgz` |

Generated classifier inputs:

| Artifact | Rows | Purpose |
| --- | ---: | --- |
| `.artifacts/session_eval/ramp_session_classifier_inputs_rjudge_compact_state_v0_1.jsonl` | 555 | Compact session evidence for labeled R-Judge metrics. |
| `.artifacts/session_eval/ramp_session_classifier_inputs_mhj_compact_state_v0_1.jsonl` | 496 | Compact session evidence for MHJ unsafe recall. |
| `.artifacts/session_eval/ramp_session_classifier_inputs_safedialbench_compact_state_v0_1.jsonl` | 4,053 | Compact evidence for SafeDialBench mining/review. |
| `.artifacts/session_eval/ramp_session_classifier_inputs_rjudge_full_transcript_v0_1.jsonl` | 555 | Full-transcript oracle baseline for R-Judge. |
| `.artifacts/session_eval/ramp_session_classifier_inputs_mhj_full_transcript_v0_1.jsonl` | 496 | Full-transcript oracle baseline for MHJ. |

The compact inputs are substantially smaller than full transcript inputs in the current corpora. This
sets up the next experiment: score compact session evidence and full transcript evidence with the
same classifier, then measure how much of the full-transcript signal compact state recovers.

### Session Classifier v2 Score Comparison

| Field | Value |
| --- | --- |
| R-Judge compact scores | `.artifacts/session_eval/ramp_session_classifier_scores_rjudge_compact_state_qwen_v0_1.jsonl` |
| R-Judge full-transcript scores | `.artifacts/session_eval/ramp_session_classifier_scores_rjudge_full_transcript_qwen_v0_1.jsonl` |
| MHJ compact scores | `.artifacts/session_eval/ramp_session_classifier_scores_mhj_compact_state_qwen_v0_1.jsonl` |
| MHJ full-transcript scores | `.artifacts/session_eval/ramp_session_classifier_scores_mhj_full_transcript_qwen_v0_1.jsonl` |
| SafeDialBench compact scores | `.artifacts/session_eval/ramp_session_classifier_scores_safedialbench_compact_state_qwen_v0_1.jsonl` |
| Session fusion evaluator | `scripts/evaluate_session_signal_fusion.py` |
| R-Judge fusion report | `.artifacts/session_eval/ramp_session_signal_fusion_eval_rjudge_qwen_v0_1.json` |
| R-Judge fusion summary | `.artifacts/session_eval/ramp_session_signal_fusion_eval_rjudge_qwen_v0_1.md` |
| MHJ fusion report | `.artifacts/session_eval/ramp_session_signal_fusion_eval_mhj_qwen_v0_1.json` |
| MHJ fusion summary | `.artifacts/session_eval/ramp_session_signal_fusion_eval_mhj_qwen_v0_1.md` |

R-Judge session-level comparison at threshold `0.55`:

| Condition | AUC | Recall | FPR | Single-turn FNs caught |
| --- | ---: | ---: | ---: | ---: |
| `single_turn_max` | 0.5471 | 0.7413 | 0.7695 | 0 |
| `compact_session_classifier` | 0.5299 | 0.1154 | 0.0558 | 1 |
| `full_transcript_session_classifier` | 0.6105 | 0.5839 | 0.3680 | 14 |
| `max_session_signal` | 0.5570 | 0.7937 | 0.8104 | 15 |

MHJ unsafe-recall stress comparison at threshold `0.55`:

| Condition | Recall | Single-turn FNs caught |
| --- | ---: | ---: |
| `single_turn_max` | 0.7984 | 0 |
| `compact_session_classifier` | 0.4133 | 4 |
| `full_transcript_session_classifier` | 0.6956 | 24 |
| `max_session_signal` | 0.8508 | 26 |

Interpretation: full-transcript session classification provides measurable session-level signal,
but compact session state is not yet a sufficient substitute. The v0 policy should not use naive
max/OR blocking because it raises R-Judge false positives. The defensible path is to keep session
classification as a calibrated escalation/audit signal while improving compact evidence
compression.

## Prompt-Label Audit v0.1

| Field | Value |
| --- | --- |
| Audit report | `.artifacts/prompt_label_audit/ramp_qwen3guard_prompt_label_audit_v0_1.json` |
| Markdown summary | `.artifacts/prompt_label_audit/ramp_qwen3guard_prompt_label_audit_v0_1.md` |
| Suspect rows | `.artifacts/prompt_label_audit/ramp_qwen3guard_prompt_label_suspect_rows_v0_1.jsonl` |
| Audit candidates | `.artifacts/prompt_label_audit/ramp_qwen3guard_prompt_label_audit_candidates_v0_1.jsonl` |
| Input feature table | `.artifacts/cumulative_signal_eval/ramp_qwen3guard_prompt_internal_signal_feature_table_v0_1.jsonl` |

Audit output:

| Bucket | Rows |
| --- | ---: |
| `corpus_unsafe_classifier_agree` | 10,720 |
| `corpus_unsafe_classifier_safe` | 7,953 |
| `corpus_safe_classifier_unsafe` | 5,116 |
| `corpus_safe_classifier_agree` | 2,004 |
| `corpus_safe_classifier_controversial` | 1,583 |
| `corpus_unsafe_classifier_controversial` | 342 |

The full suspect set has 14,994 rows. The stratified audit candidate set has 1,810 rows after
selecting up to 100 rows per disagreement bucket/source/domain stratum.

This artifact is the prompt-classifier v0 path forward. It turns the weak prompt-only ablation into
an actionable label-quality result: reviewers can now build a clean prompt-level subset, preserve
hard benign near-neighbors, and avoid treating span-derived benchmark labels as final moderation
truth.

### Prompt Review Batch 500 v0.1

| Field | Value |
| --- | --- |
| JSONL review batch | `.artifacts/prompt_label_audit/ramp_qwen3guard_prompt_label_review_batch_500_v0_1.jsonl` |
| CSV review batch | `.artifacts/prompt_label_audit/ramp_qwen3guard_prompt_label_review_batch_500_v0_1.csv` |
| Summary | `.artifacts/prompt_label_audit/ramp_qwen3guard_prompt_label_review_batch_500_v0_1.summary.json` |
| Rows | 500 |

Bucket mix:

| Audit bucket | Rows |
| --- | ---: |
| `corpus_unsafe_classifier_safe` | 234 |
| `corpus_unsafe_classifier_controversial` | 103 |
| `corpus_safe_classifier_unsafe` | 98 |
| `corpus_safe_classifier_controversial` | 65 |

Source mix:

| Source | Rows |
| --- | ---: |
| `wildguardmix` | 312 |
| `beavertails` | 107 |
| `do_not_answer` | 56 |
| `harmbench` | 25 |

The review fields are intentionally empty in the generated batch. Reviewers should fill
`reviewed_label`, `label_issue_type`, `reviewer_notes`, `reviewed_by`, and `reviewed_at`. Valid
reviewed labels are `safe`, `unsafe`, `controversial`, `ambiguous_or_context_needed`, and
`bad_benchmark_label`.

### Reviewed Prompt Eval v0.1

| Field | Value |
| --- | --- |
| Reviewed CSV | `/Users/ratnaditya/Documents/ramp_qwen3guard_prompt_label_review_batch_500_v0_1.csv` |
| Reviewed subset JSONL | `.artifacts/prompt_label_audit/ramp_qwen3guard_reviewed_prompt_subset_v0_1.jsonl` |
| Evaluation report | `.artifacts/prompt_label_audit/ramp_qwen3guard_reviewed_prompt_eval_v0_1.json` |
| Markdown summary | `.artifacts/prompt_label_audit/ramp_qwen3guard_reviewed_prompt_eval_v0_1.md` |
| Reviewed rows | 99 |
| Binary safe/unsafe eval rows | 67 |

Reviewed label mix:

| Label | Rows |
| --- | ---: |
| `safe` | 35 |
| `unsafe` | 32 |
| `controversial` | 28 |
| `ambiguous_or_context_needed` | 4 |

Qwen3Guard 0.6B against the binary reviewed subset:

| AUC | Accuracy | Precision | Recall | FPR | Threshold |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9424 | 0.8507 | 0.7750 | 0.9688 | 0.2571 | 0.50 |

Interpretation: the reviewed slice confirms that the earlier prompt-only ablation was strongly
affected by noisy span-derived labels. Qwen3Guard has a useful ranking signal on reviewed examples,
but the default `Controversial=0.58` mapping makes the 0.50 operating point conservative and
false-positive heavy. Prompt-classifier calibration should use reviewed labels.

### Reviewed Cumulative Signal Eval v0.1

| Field | Value |
| --- | --- |
| Evaluation report | `.artifacts/prompt_label_audit/ramp_qwen3guard_reviewed_cumulative_signal_eval_v0_1.json` |
| Markdown summary | `.artifacts/prompt_label_audit/ramp_qwen3guard_reviewed_cumulative_signal_eval_v0_1.md` |
| Reviewed feature table | `.artifacts/prompt_label_audit/ramp_qwen3guard_reviewed_cumulative_signal_feature_table_v0_1.jsonl` |
| Reviewed joined rows | 99 |
| Binary safe/unsafe eval rows | 67 |

Default threshold `0.50` result:

| Signal | AUC | Accuracy | Recall | FPR | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prompt only | 0.9424 | 0.8507 | 0.9688 | 0.2571 | 9 | 1 |
| embedding only | 0.5982 | 0.6269 | 0.3125 | 0.0857 | 3 | 22 |
| activation only | 0.4920 | 0.5224 | 0.4375 | 0.4000 | 14 | 18 |
| prompt + embedding | 0.9437 | 0.8806 | 0.9688 | 0.2000 | 7 | 1 |
| prompt + activation | 0.8955 | 0.9104 | 0.9375 | 0.1143 | 4 | 2 |
| prompt + embedding + activation | 0.8991 | 0.8507 | 0.8125 | 0.1143 | 4 | 6 |

False-positive delta:

| Comparison | FP count |
| --- | ---: |
| Prompt-only | 9 |
| Prompt + activation | 4 |
| Prompt + embedding + activation | 4 |

Prompt + activation fixes five prompt-only false positives while adding one additional false
negative. Prompt + embedding reduces false positives by two while preserving prompt-only recall.
Full cumulative fusion has the same false-positive reduction as prompt + activation, but the fixed
`0.50` threshold is too strict for this slice; its best-F1 threshold is `0.45`, where it matches
prompt + activation with 4 false positives and 2 false negatives.

Interpretation: on manually reviewed prompt-disagreement cases, embeddings and activations add
value as corrective/contextual signals around the prompt classifier. They should not be read as
standalone replacements on this deliberately biased slice.

### Reviewed Fusion Calibration v0.1

| Field | Value |
| --- | --- |
| Calibration report | `.artifacts/prompt_label_audit/ramp_reviewed_fusion_calibration_v0_1.json` |
| Markdown summary | `.artifacts/prompt_label_audit/ramp_reviewed_fusion_calibration_v0_1.md` |
| Weight step | 0.05 |
| Threshold step | 0.01 |
| Target FPR | 0.10 |
| Constraints | `prompt >= activation >= embedding`, `prompt >= 0.40`, `embedding <= 0.20` |

Selected calibration points:

| Objective | Prompt | Embedding | Activation | Threshold | Accuracy | Recall | FPR | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| target FPR <= 0.10 | 0.90 | 0.05 | 0.05 | 0.58 | 0.9104 | 0.8750 | 0.0571 | 2 | 4 |
| best F1 | 0.50 | 0.10 | 0.40 | 0.47 | 0.9104 | 0.9375 | 0.1143 | 4 | 2 |
| no false negatives | 0.50 | 0.00 | 0.50 | 0.18 | 0.6119 | 1.0000 | 0.7429 | 26 | 0 |

Interpretation: this moves RAMP away from hand-picked weights. Weights and thresholds are selected
by a declared grid search objective over reviewed labels. On the current small disagreement slice,
low-FPR calibration prefers a prompt-dominant score. Best-F1 calibration gives activation a larger
role and keeps embedding small. The next paper-grade step is to expand reviewed labels and evaluate
calibrated weights on a held-out reviewed split.

Runtime use:

```python
from ramp.pipeline import default_pipeline

pipeline = default_pipeline(
    fusion_calibration_artifact=".artifacts/prompt_label_audit/ramp_reviewed_fusion_calibration_v0_1.json",
    fusion_calibration_objective="selected_by_best_f1",
)
```

`RiskDecision.fusion_metadata` records the calibration artifact path, objective, weights,
threshold, and fusion mode.

Runtime severity floor: calibrated fusion now applies a `0.72` minimum score when Qwen3Guard marks
a prompt as `Unsafe` in a high-severity category such as PII, self-harm, violence, illegal acts,
jailbreak, or unethical acts. This floor is a runtime safety guard, not a learned calibration
parameter in the v0.1 artifact. Decision metadata preserves both the raw calibrated score and the
post-floor score so future audits can evaluate whether the floor reduced severe false negatives
without adding unacceptable false positives.

### Reviewed Severity Floor Eval v0.1

| Field | Value |
| --- | --- |
| Evaluation report | `.artifacts/prompt_label_audit/ramp_reviewed_severity_floor_eval_v0_1.json` |
| Markdown summary | `.artifacts/prompt_label_audit/ramp_reviewed_severity_floor_eval_v0_1.md` |
| Calibration objective | `selected_by_best_f1` |
| Threshold | 0.47 |
| Severity floor | 0.72 |
| Floor candidate rows | 24 |
| Floor applied rows | 16 |

Before/after metrics on the 67 binary reviewed rows:

| Variant | Accuracy | Precision | Recall | FPR | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw calibrated | 0.9104 | 0.8824 | 0.9375 | 0.1143 | 4 | 2 |
| Severity floor | 0.9104 | 0.8824 | 0.9375 | 0.1143 | 4 | 2 |

Interpretation: the floor increased the final risk score for 16 high-severity rows but did not
change threshold-level classification on this reviewed slice. It fixed 0 false negatives and added
0 new false positives. The remaining false negatives were not eligible for the current floor rule
because Qwen3Guard labeled one as `Safe` and one as `Controversial`, while the v0.1 floor only
applies to `Unsafe` high-severity prompt findings.

### Reviewed Evaluation Harness v0.1

| Field | Value |
| --- | --- |
| Evaluation report | `.artifacts/prompt_label_audit/ramp_reviewed_evaluation_harness_v0_1.json` |
| Markdown summary | `.artifacts/prompt_label_audit/ramp_reviewed_evaluation_harness_v0_1.md` |
| Reviewed joined rows | 99 |
| Binary safe/unsafe eval rows | 67 |
| Calibration objective | `selected_by_best_f1` |
| Severity floor | 0.72 |

Consolidated ablation table:

| Condition | Threshold | AUC | Accuracy | Recall | FPR | FP | FN | Severe FN | Hard benign FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt | 0.50 | 0.9424 | 0.8507 | 0.9688 | 0.2571 | 9 | 1 | 1 | 6 |
| Embedding | 0.50 | 0.5982 | 0.6269 | 0.3125 | 0.0857 | 3 | 22 | 21 | 3 |
| Activation | 0.50 | 0.4920 | 0.5224 | 0.4375 | 0.4000 | 14 | 18 | 18 | 1 |
| Prompt + embedding | 0.50 | 0.9437 | 0.8806 | 0.9688 | 0.2000 | 7 | 1 | 1 | 5 |
| Prompt + activation | 0.50 | 0.8955 | 0.9104 | 0.9375 | 0.1143 | 4 | 2 | 2 | 2 |
| Fixed prompt + embedding + activation | 0.50 | 0.8991 | 0.8507 | 0.8125 | 0.1143 | 4 | 6 | 6 | 2 |
| Calibrated weighted ablation | 0.47 | 0.8973 | 0.9104 | 0.9375 | 0.1143 | 4 | 2 | 2 | 2 |
| RAMP fusion | 0.47 | 0.9250 | 0.9104 | 0.9375 | 0.1143 | 4 | 2 | 2 | 2 |

Interpretation: this is the first project-level evaluation table. It reframes the work around
cumulative signal value rather than isolated feature performance. Prompt + activation and RAMP
fusion materially reduce false positives compared with prompt-only on this reviewed audit slice,
but they introduce one additional unsafe miss. The calibrated weighted row is an ablation that
shows the pre-floor calibrated score; RAMP fusion is the actual final fusion calculation. The next
step is not another tuning rule; it is to expand reviewed labels, split calibration from holdout,
and rerun this same harness.

### Prompt Review Batch 500 v0.2

| Field | Value |
| --- | --- |
| JSONL review batch | `.artifacts/prompt_label_audit/ramp_prompt_label_review_batch_500_v0_2.jsonl` |
| CSV review batch | `.artifacts/prompt_label_audit/ramp_prompt_label_review_batch_500_v0_2.csv` |
| Reviewer CSV copy | `/Users/ratnaditya/Documents/ramp_prompt_label_review_batch_500_v0_2.csv` |
| Summary | `.artifacts/prompt_label_audit/ramp_prompt_label_review_batch_500_v0_2.summary.json` |
| Rows | 500 |
| Existing reviewed rows excluded | 99 |

Selection buckets:

| Bucket | Rows |
| --- | ---: |
| `severe_miss_candidate` | 127 |
| `qwen_safe_corpus_unsafe` | 122 |
| `controversial_high_severity` | 108 |
| `hard_benign_fp_candidate` | 61 |
| `domain_coverage` | 56 |
| `ramp_uncertain` | 26 |

Domain mix:

| Domain | Rows |
| --- | ---: |
| `regulated_advice` | 147 |
| `nonviolent_illegal_activity` | 106 |
| `cyber_abuse` | 57 |
| `weapons_and_physical_violence` | 45 |
| `child_safety` | 41 |
| `ip_and_content_rights` | 31 |
| `privacy_identity_and_secrets` | 21 |
| `cbrn_and_hazardous_materials` | 17 |
| `self_harm_and_wellbeing` | 11 |
| `sexual_safety_and_content` | 10 |
| `agent_tool_and_system_integrity` | 7 |
| `misinformation_manipulation_and_civic` | 4 |
| `hate_harassment_and_abuse` | 3 |

Interpretation: v0.2 is not random sampling. It is targeted from the RAMP evaluation harness to
expand evidence around severe misses, hard benign false positives, controversial high-severity
cases, RAMP uncertainty, and under-covered domains. This is the right next data step before
claiming calibrated fusion performance.

### Split-Calibrated RAMP Evaluation v0.1

| Field | Value |
| --- | --- |
| Evaluation report | `.artifacts/prompt_label_audit/ramp_split_calibrated_evaluation_v0_1.json` |
| Markdown summary | `.artifacts/prompt_label_audit/ramp_split_calibrated_evaluation_v0_1.md` |
| Reviewed CSV | `/Users/ratnaditya/Documents/ramp_prompt_label_review_combined_v0_1_v0_2_cleaned.csv` |
| Binary rows | 235 |
| Calibration rows | 117 |
| Holdout rows | 118 |
| Split seed | `ramp_reviewed_split_v0_1` |
| Selected objective | `selected_by_best_f1` |
| Selected weights | prompt `0.60`, embedding `0.20`, activation `0.20` |
| Selected threshold | 0.40 |

Holdout metrics:

| Condition | AUC | Accuracy | Recall | FPR | FP | FN | Severe FN | Hard benign FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt only | 0.9470 | 0.9153 | 0.9661 | 0.1356 | 8 | 2 | 1 | 7 |
| Prompt + embedding | 0.9440 | 0.9237 | 0.9492 | 0.1017 | 6 | 3 | 2 | 6 |
| Prompt + activation | 0.9305 | 0.9068 | 0.8814 | 0.0678 | 4 | 7 | 6 | 4 |
| Calibrated weighted ablation | 0.9483 | 0.9237 | 0.9661 | 0.1186 | 7 | 2 | 1 | 6 |
| RAMP fusion | 0.9497 | 0.9237 | 0.9661 | 0.1186 | 7 | 2 | 1 | 6 |

Interpretation: this is the first split-aware result, so it is stronger than prior same-slice
calibration numbers. It supports a modest cumulative-signal claim on the targeted reviewed
holdout: RAMP fusion preserves prompt-only recall, improves AUC from 0.9470 to 0.9497, improves
accuracy from 0.9153 to 0.9237, and reduces false positives from 8 to 7. The effect is positive but
small. This is not yet paper-final evidence because the reviewed set is targeted and domain
coverage remains uneven.

### Split Stability Evaluation v0.1

| Field | Value |
| --- | --- |
| Evaluation report | `.artifacts/prompt_label_audit/ramp_split_stability_evaluation_v0_1.json` |
| Markdown summary | `.artifacts/prompt_label_audit/ramp_split_stability_evaluation_v0_1.md` |
| Reviewed CSV | `/Users/ratnaditya/Documents/ramp_prompt_label_review_combined_v0_1_v0_2_cleaned.csv` |
| Binary rows | 235 |
| Splits | 30 |
| Calibration fraction | 0.50 |
| Selected objective | `selected_by_best_f1` |

Aggregate holdout metrics across 30 deterministic splits:

| Condition | AUC mean | Accuracy mean | Recall mean | FPR mean | FP mean | FN mean | Severe FN mean | Hard benign FP mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt only | 0.9499 | 0.9138 | 0.9785 | 0.1508 | 8.90 | 1.27 | 0.80 | 5.67 |
| Prompt + embedding | 0.9524 | 0.9274 | 0.9678 | 0.1130 | 6.67 | 1.90 | 1.43 | 5.10 |
| Prompt + activation | 0.9208 | 0.9088 | 0.8966 | 0.0791 | 4.67 | 6.10 | 5.63 | 3.60 |
| RAMP fusion | 0.9450 | 0.9209 | 0.9582 | 0.1164 | 6.87 | 2.47 | 2.37 | 4.23 |

RAMP fusion minus prompt-only across splits:

| Delta | Mean | Stdev | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| AUC | -0.0049 | 0.0133 | -0.0398 | 0.0205 |
| Accuracy | 0.0071 | 0.0105 | -0.0169 | 0.0254 |
| Recall | -0.0203 | 0.0215 | -0.0678 | 0.0169 |
| FPR | -0.0345 | 0.0216 | -0.0847 | 0.0169 |
| FP count | -2.03 | 1.27 | -5.00 | 1.00 |
| FN count | 1.20 | 1.27 | -1.00 | 4.00 |
| Severe FN count | 1.57 | 1.10 | 0.00 | 4.00 |
| Hard benign FP count | -1.43 | 0.94 | -3.00 | 0.00 |

Interpretation: repeated splits make the current tradeoff clearer. RAMP fusion consistently lowers
false positives and hard-benign false positives relative to prompt-only, but it gives back recall
and increases severe false negatives on average. This does not invalidate the cumulative-signal
direction, but it means the next review work must focus on repeated severe misses and hard benign
false positives before making a paper-grade claim.

### Taxonomy-Corrected Centroid And Stability Rerun v0.1

| Field | Value |
| --- | --- |
| Taxonomy correction artifact | `data/taxonomy/taxonomy_corrections_v0_1.json` |
| Corrected embeddings | `.artifacts/corrections/ramp_benchmark_comprehensive_v0.input_embeddings.taxonomy_corrected_v0_1.jsonl` |
| Corrected centroid artifact | `.artifacts/centroids/ramp_input_embedding_centroids_comprehensive_taxonomy_corrected_v0_1.json` |
| Corrected embedding scores | `.artifacts/centroids/ramp_input_embedding_centroids_comprehensive_taxonomy_corrected_v0_1.centered_domain_conditioned_scores.jsonl` |
| Corrected cumulative feature table | `.artifacts/cumulative_signal_eval/ramp_qwen3guard_prompt_internal_signal_feature_table_taxonomy_corrected_v0_1.jsonl` |
| Corrected stability report | `.artifacts/prompt_label_audit/ramp_split_stability_evaluation_taxonomy_corrected_v0_1.json` |
| Corrected stability markdown | `.artifacts/prompt_label_audit/ramp_split_stability_evaluation_taxonomy_corrected_v0_1.md` |
| Corrections applied | 1 row |
| Centroids | 20 |

The corrected row is `bench_comp_0014981`, remapped from
`nonviolent_illegal_activity / organized_abuse_workflows` to
`self_harm_and_wellbeing / suicidal_ideation_or_crisis`.

Corrected full-corpus ablation:

| Condition | AUC | Recall at <=5% FPR | FPR |
| --- | ---: | ---: | ---: |
| Embedding only | 0.9361 | 0.7758 | 0.0472 |
| Activation only | 0.9959 | 0.9876 | 0.0494 |
| Cumulative internal | 0.9955 | 0.9875 | 0.0499 |
| Prompt + embedding | 0.6457 | 0.5169 | 0.0344 |
| Prompt + activation | 0.7424 | 0.5759 | 0.0444 |
| Prompt + embedding + activation | 0.8450 | 0.5757 | 0.0437 |

Corrected reviewed 30-split stability changes:

| Metric | Old RAMP fusion mean | Corrected RAMP fusion mean | Change |
| --- | ---: | ---: | ---: |
| AUC | 0.9450 | 0.9490 | +0.0040 |
| Accuracy | 0.9209 | 0.9263 | +0.0054 |
| Recall | 0.9582 | 0.9559 | -0.0023 |
| FPR | 0.1164 | 0.1034 | -0.0130 |
| FP count | 6.87 | 6.10 | -0.77 |
| FN count | 2.47 | 2.60 | +0.13 |
| Severe FN count | 2.37 | 2.23 | -0.13 |
| Hard benign FP count | 4.23 | 4.10 | -0.13 |

Corrected RAMP fusion minus prompt-only:

| Delta | Old mean | Corrected mean |
| --- | ---: | ---: |
| AUC | -0.0049 | +0.0018 |
| Accuracy | +0.0071 | +0.0105 |
| Recall | -0.0203 | -0.0153 |
| FPR | -0.0345 | -0.0362 |
| FP count | -2.03 | -2.13 |
| FN count | +1.20 | +0.90 |
| Severe FN count | +1.57 | +1.17 |
| Hard benign FP count | -1.43 | -1.20 |

Interpretation: the taxonomy correction improves the stability result. RAMP fusion now has a
slightly positive AUC delta over prompt-only, better accuracy, lower FPR, fewer false positives,
and a smaller FN/severe-FN penalty than before. The corrected self-harm row receives a much
stronger embedding prior (`0.9487`), but its activation probability remains very low (`0.0176`),
so fusion still needs severity-aware constraints and more reviewed self-harm coverage before this
can be treated as paper-grade evidence.

### Calibrated Signal Combination Stability v0.1

| Field | Value |
| --- | --- |
| Old activation-probe combination report | `.artifacts/prompt_label_audit/ramp_calibrated_signal_combinations_taxonomy_corrected_v0_1.json` |
| Reviewed activation probe | `.artifacts/activation_probes/ramp_reviewed_activation_probe_layer_19_taxonomy_corrected_v0_1.json` |
| Reviewed activation probe report | `.artifacts/activation_probes/ramp_reviewed_activation_probe_layer_19_taxonomy_corrected_v0_1.report.json` |
| Reviewed-probe combination report | `.artifacts/prompt_label_audit/ramp_calibrated_signal_combinations_taxonomy_corrected_reviewed_activation_v0_1.json` |
| Reviewed-probe combination markdown | `.artifacts/prompt_label_audit/ramp_calibrated_signal_combinations_taxonomy_corrected_reviewed_activation_v0_1.md` |

Fair combination calibration asks each condition to choose its own weights and threshold on the
calibration split before holdout evaluation. This separates two questions:

1. Is activation useful after calibration?
2. Was the old activation probe misaligned with reviewed labels?

With the old benchmark-trained activation probe:

| Condition | AUC mean | Accuracy mean | Recall mean | FPR mean | FP mean | FN mean | Severe FN mean | Hard benign FP mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt only calibrated | 0.9480 | 0.9167 | 0.9734 | 0.1401 | 8.27 | 1.57 | 1.00 | 5.37 |
| Prompt + embedding calibrated | 0.9501 | 0.9195 | 0.9492 | 0.1102 | 6.50 | 3.00 | 2.43 | 5.00 |
| Prompt + activation calibrated | 0.9493 | 0.9218 | 0.9463 | 0.1028 | 6.07 | 3.17 | 2.60 | 3.90 |
| Prompt + embedding + activation calibrated | 0.9464 | 0.9234 | 0.9514 | 0.1045 | 6.17 | 2.87 | 2.30 | 4.17 |

After training a layer-19 activation probe on the 235 reviewed rows:

| Condition | AUC mean | Accuracy mean | Recall mean | FPR mean | FP mean | FN mean | Severe FN mean | Hard benign FP mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt only calibrated | 0.9460 | 0.9116 | 0.9672 | 0.1441 | 8.50 | 1.93 | 1.33 | 5.47 |
| Prompt + embedding calibrated | 0.9498 | 0.9155 | 0.9412 | 0.1102 | 6.50 | 3.47 | 2.87 | 4.87 |
| Prompt + activation calibrated | 0.9982 | 0.9788 | 0.9797 | 0.0220 | 1.30 | 1.20 | 1.00 | 1.23 |
| Prompt + embedding + activation calibrated | 0.9978 | 0.9695 | 0.9774 | 0.0384 | 2.27 | 1.33 | 1.30 | 2.03 |

Interpretation: activation becomes the strongest reviewed-label signal once the probe is trained on
reviewed labels instead of noisy benchmark-derived labels. Embeddings remain useful as a
taxonomy-aware boundary signal, but the best current reviewed-label pair is prompt + reviewed
activation. The full three-stage fusion is close, but adding embedding to the reviewed activation
probe currently increases false positives slightly. This argues for combination-specific
calibration and more reviewed rows before fixing a final three-stage weight policy.

### Taxonomy-Corrected Activation Probe And Unconstrained Fusion v0.1

| Field | Value |
| --- | --- |
| Taxonomy-corrected activation rows | `.artifacts/corrections/ramp_benchmark_comprehensive_v0.layer_19.taxonomy_corrected_v0_1.jsonl` |
| Taxonomy-corrected benchmark probe | `.artifacts/activation_probes/taxonomy_corrected_v0_1/ramp_activation_probe_layer_19_v0_1.json` |
| Probe comparison report | `.artifacts/activation_probes/ramp_activation_probe_layer_19_taxonomy_corrected_comparison_v0_1.json` |
| Cumulative feature table | `.artifacts/cumulative_signal_eval/ramp_qwen3guard_prompt_internal_signal_feature_table_taxonomy_corrected_probe_v0_1.jsonl` |
| Unconstrained combination report | `.artifacts/prompt_label_audit/ramp_calibrated_signal_combinations_taxonomy_corrected_probe_v0_1.json` |
| Reviewed-probe unconstrained combination report | `.artifacts/prompt_label_audit/ramp_calibrated_signal_combinations_taxonomy_corrected_reviewed_activation_unconstrained_v0_1.json` |

Layer 19 trained on taxonomy-corrected benchmark labels reached AUC `0.9953` and recall `0.9869`
at the low-FPR target on its benchmark holdout. However, reviewed-label fusion calibration is the
more important policy test.

With no weight constraints, the taxonomy-corrected benchmark probe produced:

| Condition | AUC mean | Recall mean | FPR mean |
| --- | ---: | ---: | ---: |
| Prompt only calibrated | 0.9478 | 0.9706 | 0.1452 |
| Prompt + embedding calibrated | 0.9468 | 0.9458 | 0.1164 |
| Prompt + activation calibrated | 0.9466 | 0.9407 | 0.1113 |
| Prompt + embedding + activation calibrated | 0.9447 | 0.9492 | 0.1136 |

The most common prompt+activation weights for this benchmark-trained probe were prompt `0.75` /
activation `0.25` and prompt `0.70` / activation `0.30`. This does not prove activation deserves a
dominant runtime weight under reviewed labels.

With the provisional reviewed-label/hard-case probe, the same unconstrained calibration produced:

| Condition | AUC mean | Recall mean | FPR mean |
| --- | ---: | ---: | ---: |
| Prompt only calibrated | 0.9478 | 0.9706 | 0.1452 |
| Prompt + embedding calibrated | 0.9468 | 0.9458 | 0.1164 |
| Prompt + activation calibrated | 0.9985 | 0.9797 | 0.0299 |
| Prompt + embedding + activation calibrated | 0.9976 | 0.9780 | 0.0373 |

Here the most common prompt+activation weights were prompt `0.50` / activation `0.50`, prompt
`0.45` / activation `0.55`, and prompt `0.40` / activation `0.60`. This is stronger evidence that
activation can carry substantial reviewed-label value, but it remains provisional because the
reviewed probe is trained on only 235 binary reviewed rows. The current policy stance is therefore:
do not assign activation a high fixed weight by assumption; allow calibrated reviewed-label
performance to earn that weight.

### Expanded Reviewed Activation And Frozen Fusion Policy v0.1

| Field | Value |
| --- | --- |
| Cleaned v0.4 review CSV | `/Users/ratnaditya/Documents/ramp_prompt_label_review_batch_500_v0_4_cleaned.csv` |
| Combined reviewed CSV | `/Users/ratnaditya/Documents/ramp_prompt_label_review_combined_v0_1_v0_2_v0_4_cleaned.csv` |
| Combined summary | `.artifacts/prompt_label_audit/ramp_prompt_label_review_combined_v0_1_v0_2_v0_4_cleaned.summary.json` |
| Expanded reviewed activation probe | `.artifacts/activation_probes/ramp_reviewed_activation_probe_layer_19_expanded_v0_1.json` |
| Expanded probe report | `.artifacts/activation_probes/ramp_reviewed_activation_probe_layer_19_expanded_v0_1.report.json` |
| Expanded feature table | `.artifacts/cumulative_signal_eval/ramp_qwen3guard_prompt_internal_signal_feature_table_expanded_reviewed_activation_v0_1.jsonl` |
| Expanded calibration report | `.artifacts/prompt_label_audit/ramp_calibrated_signal_combinations_expanded_reviewed_activation_v0_1.json` |
| Frozen v0 policy | `data/fusion_policy/ramp_fusion_policy_v0_1.json` |

The v0.4 review cleanup produced 220 newly reviewed rows: 143 unsafe, 70 safe, 1 controversial,
and 6 ambiguous/context-needed rows, with one row left unreviewed. The combined reviewed set now has
448 binary rows: 260 unsafe and 188 safe.

The expanded reviewed activation probe became harder after adding these targeted rows: holdout AUC
`0.9116` and recall `0.6731` at the low-FPR target. This is expected because v0.4 deliberately
sampled difficult activation misses and embedding conflicts.

Unconstrained split calibration on the expanded reviewed set:

| Condition | AUC mean | Accuracy mean | Recall mean | FPR mean | FP mean | FN mean | Severe FN mean | Hard benign FP mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt only calibrated | 0.9425 | 0.9266 | 0.9695 | 0.1326 | 12.47 | 3.97 | 2.20 | 6.27 |
| Prompt + embedding calibrated | 0.9592 | 0.9289 | 0.9574 | 0.1106 | 10.40 | 5.53 | 3.77 | 5.83 |
| Prompt + activation calibrated | 0.9939 | 0.9616 | 0.9626 | 0.0397 | 3.73 | 4.87 | 2.87 | 3.33 |
| Prompt + embedding + activation calibrated | 0.9939 | 0.9598 | 0.9633 | 0.0450 | 4.23 | 4.77 | 2.87 | 3.67 |

Decision: full three-signal fusion does not beat prompt + activation for the v0 runtime policy.
It has essentially tied AUC and slightly higher recall, but worse false-positive rate, more false
positives, and more hard-benign false positives. The frozen v0 runtime policy is therefore
prompt+activation with most-common calibrated weights: prompt `0.25`, activation `0.75`,
embedding `0.00`, threshold `0.53`. Embedding remains in RAMP as a secondary taxonomy/audit signal
and can regain runtime weight only if future reviewed split-stability runs prove additive value.

Runtime wiring:

```python
from ramp.pipeline import default_pipeline

pipeline = default_pipeline(
    fusion_policy_artifact="data/fusion_policy/ramp_fusion_policy_v0_1.json",
)
```

When prompt and activation scores are present, `WeightedRiskFusion` applies the frozen policy. The
embedding feature may still be present and is recorded in contributions/metadata, but its v0 policy
weight is `0.00`, so it does not change the runtime score. Decisions record required stages,
available calibrated stages, and zero-weight ignored stages in `RiskDecision.fusion_metadata`.

### Prompt Review Batch 500 v0.3

| Field | Value |
| --- | --- |
| JSONL review batch | `.artifacts/prompt_label_audit/ramp_prompt_label_review_batch_500_v0_3.jsonl` |
| Reviewer CSV copy | `/Users/ratnaditya/Documents/ramp_prompt_label_review_batch_500_v0_3.csv` |
| Summary | `.artifacts/prompt_label_audit/ramp_prompt_label_review_batch_500_v0_3.summary.json` |
| Rows | 359 |
| Existing reviewed rows excluded | 327 |

Selection buckets:

| Bucket | Rows |
| --- | ---: |
| `stability_severe_fn_slice` | 53 |
| `stability_hard_benign_fp_slice` | 41 |
| `stability_ramp_fn_slice` | 21 |
| `stability_ramp_fp_slice` | 40 |
| `fusion_prompt_disagreement` | 62 |
| `undercovered_domain` | 89 |
| `uncertain_margin` | 53 |

Undercovered domains targeted by v0.3: `cbrn_and_hazardous_materials`, `child_safety`,
`ip_and_content_rights`, `misinformation_manipulation_and_civic`,
`privacy_identity_and_secrets`, `self_harm_and_wellbeing`, and
`sexual_safety_and_content`.

### Prompt Review Batch 500 v0.4

| Field | Value |
| --- | --- |
| JSONL review batch | `.artifacts/prompt_label_audit/ramp_prompt_label_review_batch_500_v0_4.jsonl` |
| Reviewer CSV copy | `/Users/ratnaditya/Documents/ramp_prompt_label_review_batch_500_v0_4.csv` |
| Summary | `.artifacts/prompt_label_audit/ramp_prompt_label_review_batch_500_v0_4.summary.json` |
| Rows | 221 |
| Existing reviewed or queued source IDs excluded | 1,070 |

This batch locks the current v0 finding into the review workflow: prompt plus reviewed activation is
currently the strongest reviewed-label pair, while embedding remains a useful taxonomy-aware
boundary signal that needs more hard-neighbor review before final weight-policy selection.

Selection buckets:

| Bucket | Rows |
| --- | ---: |
| `activation_false_negative_candidate` | 51 |
| `severe_activation_false_negative_candidate` | 63 |
| `embedding_false_positive_candidate` | 34 |
| `embedding_activation_conflict_candidate` | 73 |

Domain mix:

| Domain | Rows |
| --- | ---: |
| `regulated_advice` | 66 |
| `nonviolent_illegal_activity` | 51 |
| `ip_and_content_rights` | 33 |
| `weapons_and_physical_violence` | 25 |
| `self_harm_and_wellbeing` | 17 |
| `cyber_abuse` | 14 |
| `child_safety` | 8 |
| `privacy_identity_and_secrets` | 4 |
| `agent_tool_and_system_integrity` | 1 |
| `cbrn_and_hazardous_materials` | 1 |
| `sexual_safety_and_content` | 1 |

The review goal is not broad corpus coverage. It is targeted adjudication of rows most likely to
change the final three-signal policy: unsafe rows the reviewed activation probe may miss, benign
near-neighbors that embedding may over-score, and rows where embedding and activation disagree
strongly.

## Corpus Mix

The comprehensive corpus was built from benchmark-derived spans:

| Source | Rows | Role contribution |
| --- | ---: | --- |
| `wildguardmix` | 18,622 | Mostly harmful/evasion |
| `harmbench` | 393 | Harmful/evasion |
| `beavertails` | 7,766 | Benign near-neighbor |
| `do_not_answer` | 937 | Benign near-neighbor |

Label and role totals:

| Group | Rows | Share |
| --- | ---: | ---: |
| Harmful | 18,834 | 67.95% |
| Benign near-neighbor | 8,703 | 31.40% |
| Evasion | 181 | 0.65% |

The corpus is intentionally not a uniform sample of benign language. Benign examples are used as
hard contrast anchors near harmful neighborhoods, not as a broad model of all safe requests.

## Input Embedding Centroid Artifact v0.1

| Field | Value |
| --- | --- |
| Artifact ID | `ramp_input_embedding_centroids_comprehensive_v0.1` |
| Local path | `.artifacts/centroids/ramp_input_embedding_centroids_comprehensive_v0_1.json` |
| Created at | `2026-06-07T18:09:32.052850+00:00` |
| Embedding source | `gpt_oss_20b_input_embedding_v0.1` |
| Taxonomy | `ramp_taxonomy_v0.1` |
| Input rows | 27,718 |
| Centroids | 19 |
| Dimension | 2,880 |
| Method | Mean of L2-normalized span vectors per `domain/subcluster_role/subcluster_id`, followed by L2 normalization. |

Centroids by role:

| Role | Centroids | Rows |
| --- | ---: | ---: |
| Harmful | 13 | 18,834 |
| Benign near-neighbor | 5 | 8,703 |
| Evasion | 1 | 181 |

Centroid inventory:

| Rows | Domain | Role | Subcluster | Sources |
| ---: | --- | --- | --- | --- |
| 11,372 | `nonviolent_illegal_activity` | `harmful` | `organized_abuse_workflows` | `harmbench: 252`, `wildguardmix: 11120` |
| 8,538 | `regulated_advice` | `benign_near_neighbor` | `general_information` | `beavertails: 7639`, `do_not_answer: 899` |
| 2,412 | `regulated_advice` | `harmful` | `unsafe_professional_instruction` | `harmbench: 6`, `wildguardmix: 2406` |
| 1,071 | `cyber_abuse` | `harmful` | `vulnerability_exploitation` | `harmbench: 16`, `wildguardmix: 1055` |
| 780 | `ip_and_content_rights` | `harmful` | `copyright_reproduction` | `harmbench: 50`, `wildguardmix: 730` |
| 646 | `weapons_and_physical_violence` | `harmful` | `weapon_construction` | `harmbench: 17`, `wildguardmix: 629` |
| 607 | `child_safety` | `harmful` | `sexualized_minors` | `harmbench: 8`, `wildguardmix: 599` |
| 422 | `sexual_safety_and_content` | `harmful` | `explicit_adult_generation` | `harmbench: 7`, `wildguardmix: 415` |
| 382 | `nonviolent_illegal_activity` | `harmful` | `fraud_scams` | `harmbench: 9`, `wildguardmix: 373` |
| 372 | `hate_harassment_and_abuse` | `harmful` | `targeted_harassment` | `harmbench: 5`, `wildguardmix: 367` |
| 326 | `privacy_identity_and_secrets` | `harmful` | `pii_extraction` | `harmbench: 1`, `wildguardmix: 325` |
| 181 | `agent_tool_and_system_integrity` | `evasion` | `jailbreak` | `harmbench: 2`, `wildguardmix: 179` |
| 162 | `self_harm_and_wellbeing` | `harmful` | `suicide_methods` | `harmbench: 4`, `wildguardmix: 158` |
| 152 | `cbrn_and_hazardous_materials` | `harmful` | `chemical_misuse_procedure` | `harmbench: 8`, `wildguardmix: 144` |
| 130 | `misinformation_manipulation_and_civic` | `harmful` | `election_falsehoods` | `harmbench: 8`, `wildguardmix: 122` |
| 69 | `cyber_abuse` | `benign_near_neighbor` | `defensive_security` | `beavertails: 50`, `do_not_answer: 19` |
| 66 | `weapons_and_physical_violence` | `benign_near_neighbor` | `historical_analysis` | `beavertails: 54`, `do_not_answer: 12` |
| 27 | `cbrn_and_hazardous_materials` | `benign_near_neighbor` | `lab_safety` | `beavertails: 22`, `do_not_answer: 5` |
| 3 | `privacy_identity_and_secrets` | `benign_near_neighbor` | `redaction` | `beavertails: 1`, `do_not_answer: 2` |

Warnings:

```json
[
  {
    "domain": "privacy_identity_and_secrets",
    "subcluster_role": "benign_near_neighbor",
    "subcluster_id": "redaction",
    "count": 3,
    "warning": "below_min_count_warning"
  }
]
```

## Interpretation

This run should be treated as `benchmark-derived v0`, not as final calibrated production
centroids.

The harmful/benign asymmetry is expected and intentional. RAMP does not attempt to model all benign
language with centroids. It models harmful neighborhoods plus selected benign near-neighbors that
act as contrastive anchors for borderline safety decisions.

Quality questions for the next phase:

- For each harmful centroid, which benign near-neighbor is closest?
- Which harmful domains lack a nearby benign contrast anchor?
- Which harmful and benign centroids collide so closely that centroid scoring alone is unreliable?
- Which false-positive benchmark examples are nearest to harmful centroids?
- What score thresholds preserve recall while reducing false positives on hard benign neighbors?

## Reproduction Commands

Build centroids locally from the extracted input embedding file:

```bash
python scripts/build_embedding_centroids.py \
  --embeddings .artifacts/runpod/comprehensive/extracted/ramp-artifacts/embeddings/gpt_oss_20b_input_embedding_v0_1/ramp_benchmark_comprehensive_v0.input_embeddings.jsonl \
  --embedding-source data/embedding_source/gpt_oss_20b_input_embedding_v0_1.json \
  --taxonomy data/taxonomy/ramp_taxonomy_v0_1.json \
  --output .artifacts/centroids/ramp_input_embedding_centroids_comprehensive_v0_1.json \
  --artifact-id ramp_input_embedding_centroids_comprehensive_v0.1 \
  --min-count-warning 25
```

Verify the bundle checksum:

```bash
shasum -a 256 .artifacts/runpod/comprehensive/ramp-gpt-oss-comprehensive-artifacts-v0.tgz
cat .artifacts/runpod/comprehensive/ramp-gpt-oss-comprehensive-artifacts-v0.tgz.sha256
gzip -t .artifacts/runpod/comprehensive/ramp-gpt-oss-comprehensive-artifacts-v0.tgz
```
