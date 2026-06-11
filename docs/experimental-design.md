# RAMP Experimental Design

This document is the project spine for RAMP as a research system. It separates the central
research question from implementation details so future work does not get trapped in tuning one
feature at a time.

## Research Claim

RAMP tests whether a multi-stage classifier that accumulates prompt, embedding, activation,
output, session, and tool/action signals can make better safety decisions than a single-stage
classifier.

The intended claim is not that every signal is independently strong. The intended claim is that
the accumulated evidence improves decisions on hard cases:

- fewer false positives on benign near-neighbors
- fewer severe false negatives on harmful prompts
- better uncertainty handling through escalation or continued evaluation
- auditable decisions with feature-level provenance
- usable runtime behavior where cheap signals can act early and expensive signals are reserved for
  ambiguous or high-stakes cases

## Non-Claims

RAMP should not claim:

- to replace prompt classifiers such as Qwen3Guard, Llama Guard, WildGuard, or commercial
  moderation systems
- that input embeddings alone are a reliable safety classifier
- that activation probes alone are production-ready safety guards
- that current v0 weights, floors, or thresholds are final
- that benchmark-derived span labels are clean prompt-level ground truth

The paper should instead describe these as signals inside a cumulative classifier.

## Feature Stages

RAMP evaluates evidence in stages. Each stage emits a versioned feature result with score,
confidence, label, metadata, and provenance.

| Stage | Role | Current status | Research question |
| --- | --- | --- | --- |
| Prompt classifier | Front-door semantic and policy classifier | Qwen3Guard v0.6B implemented and audited | How much does a prompt classifier contribute after reviewed-label calibration? |
| Input embedding proximity | Early internal semantic prior | GPT-OSS input-embedding centroids built | Does embedding proximity reduce prompt-classifier false positives on hard benign neighbors? |
| Activation probe | Internal model-state signal | GPT-OSS layer 19 linear probe selected | Do activations add useful signal beyond prompt text? |
| Output classifier | Post-generation safety check | Qwen3Guard output scoring built and evaluated | Does response-level evidence catch failures missed pre-generation? |
| Session signal | Cross-turn accumulation | Compact and full-transcript classifiers built and evaluated | Does risk drift or multi-prompt composition improve detection? |
| Tool/action gate | Agent action safety | Runtime scaffolded, benchmark validation pending | Do proposed actions expose risk not visible in text alone? |

## Evaluation Conditions

The core evaluation should compare cumulative signal value, not isolated replacement value.

Primary ablations:

| Condition | Purpose |
| --- | --- |
| Prompt only | Baseline open classifier behavior |
| Embedding only | Weak prior sanity check, not a decision target |
| Activation only | Internal-signal standalone upper/lower bound |
| Prompt + embedding | Tests whether embedding helps on prompt-classifier false positives |
| Prompt + activation | Tests whether activation helps on hard prompt cases |
| Prompt + embedding + activation | Full pre-generation fusion candidate |
| Prompt + embedding + activation + output | Adds post-generation evidence |
| Full RAMP | Adds session and tool/action evidence |

Secondary ablations:

- calibrated weighted fusion before runtime policy floors
- RAMP fusion with severity floors included in the fusion calculation
- same-domain vs any-domain benign embedding contrast
- raw cosine vs centered cosine embedding scoring
- mid-layer vs late-layer vs final-layer activation probes

## Datasets

RAMP needs multiple datasets because one dataset cannot answer every question.

| Dataset | Purpose | Status |
| --- | --- | --- |
| Comprehensive benchmark-derived corpus | Broad extraction corpus for embeddings and activations | Built, 27,718 rows |
| Human-reviewed disagreement set | Clean prompt-level evaluation for classifier/fusion decisions | Started, 99 reviewed rows, 67 binary |
| Hard benign near-neighbor set | Borderline benign examples close to harmful clusters | Partially represented by benchmarks |
| Severe harm set | Stress test for severe false negatives | Needs explicit construction |
| Output-risk set | Prompt/response pairs for output classifier evaluation | Built, 134 generated response rows |
| Session-risk set | Multi-turn risk accumulation, drift, and composition | Built for R-Judge, MHJ, and SafeDialBench mining |
| Tool/action set | Proposed actions, arguments, and permission context | Scaffolded, needs benchmark data |

## Metrics

Use metrics that match the safety classifier goal.

Ranking metrics:

- AUROC
- AUPRC when class balance is skewed

Operating-point metrics:

- recall at fixed false-positive rate
- false-positive rate at fixed recall
- severe false-negative count
- false-positive reduction on hard benign near-neighbors
- escalation/abstention rate

Runtime metrics:

- latency by stage
- cost tier by stage
- percent of examples resolved before expensive stages
- percent routed to escalation or continued evaluation

Audit metrics:

- feature disagreement count
- decision provenance completeness
- score calibration by reviewed label
- category-level error slices

## Scientific Calibration Protocol

Weights and thresholds should be selected by declared protocol, not hand tuning.

For each reviewed-data milestone:

1. Freeze the feature table and provenance.
2. Split reviewed rows into calibration and holdout partitions when the reviewed set is large
   enough.
3. Grid-search or fit weights on calibration rows only.
4. Select operating points using declared objectives, such as target false-positive rate, best F1,
   or no severe false negatives under an allowed false-positive budget.
5. Report all final metrics on holdout rows.
6. Preserve artifacts in the registry.

Until the reviewed set is large enough to split, RAMP should call results "v0 audit findings", not
paper-grade final metrics.

## Current Evidence

Current v0 findings:

- Qwen3Guard looked weak against noisy benchmark-derived labels but strong on the first reviewed
  disagreement slice.
- Embeddings are weak as a standalone classifier but can add useful context.
- Activation probes, especially GPT-OSS layer 19, remain useful internal-model research signals,
  but the earlier activation-heavy policy was invalidated by leakage-free cross-fitting.
- The frozen v0.2 input-side runtime core is prompt `0.80` plus embedding `0.20`, threshold `0.50`.
- Calibrated fusion is better justified than fixed hand-picked weights.
- Cross-fitted repeated split evaluation shows prompt+embedding has the strongest current selected
  runtime tradeoff.
- Embedding earns positive v0.2 runtime weight on the reviewed hard-case set, pending blind holdout.
- Output scoring is implemented, but output-inclusive fusion did not improve the best v0 input-side
  result.
- Full-transcript session scoring shows real session signal, but compact state is too lossy and
  naive OR/max session fusion raises false positives.
- Severity floors are useful as a runtime safety constraint, but the current reviewed slice shows
  score-severity adjustment rather than threshold-level classification improvement.

The evidence supports calling RAMP feature-complete as a research v0. It does not yet support a
paper-final or production-grade claim.

## Build Roadmap

### Phase 1: Make Internal-Signal Evaluation Paper-Ready

Goal: convert prompt + embedding + activation from promising v0 artifacts into a clean evaluation
harness.

Tasks:

- expand the reviewed disagreement set
- create calibration and holdout splits
- rerun calibrated fusion on split data
- report prompt, embedding, activation, and fused ablations
- slice errors by domain, source, severity, and benign near-neighbor status

Exit criterion:

- a stable table showing whether internal signals add value beyond prompt classifier alone on
  reviewed holdout rows

### Phase 2: Add Output Classifier

Goal: test whether post-generation text adds cumulative value.

Tasks:

- use the Qwen3Guard output-risk backend for generated response text
- build prompt/response evaluation rows with human labels
- score responses into `output_risk_score`
- add output classifier ablations to the harness with `--include-output`
- evaluate whether output evidence catches missed harmful behavior or mostly duplicates prompt
  evidence

Exit criterion:

- output-risk ablation table and error analysis

Implementation status: the runtime output classifier path is wired through
`Qwen3GuardOutputRiskFeature` and `default_pipeline(output_risk_backend="qwen3guard")`. The
calibrated-combination evaluator now supports output-inclusive combinations when the feature table
contains `output_risk_score`.

Prompt/response eval set v0.1 has been initialized at:

- `.artifacts/output_eval/ramp_output_eval_set_v0_1.jsonl`
- `/Users/ratnaditya/Documents/ramp_output_eval_set_v0_1.csv`

The set has 134 reviewed-label rows selected from input-side true positives, true negatives, false
negatives, and hard benign false positives. The skeleton leaves `output_text` blank by design;
responses are generated separately before output scoring. The completed v0.1 artifacts are:

- `.artifacts/output_eval/ramp_output_eval_set_v0_1.generated.jsonl`
- `.artifacts/output_eval/ramp_output_scores_qwen3guard_v0_1.jsonl`
- `.artifacts/output_eval/ramp_prompt_embedding_activation_output_feature_table_v0_1.jsonl`
- `.artifacts/output_eval/ramp_prompt_embedding_activation_output_calibration_refined_v0_1.md`

The refined output-inclusive calibration does not improve the best input-side fusion:

| Condition | AUC | Recall | FPR |
| --- | ---: | ---: | ---: |
| `prompt_embedding_activation_calibrated` | 0.9864 | 0.9443 | 0.0697 |
| `prompt_activation_output_calibrated` | 0.9858 | 0.9457 | 0.0773 |
| `prompt_embedding_activation_output_calibrated` | 0.9851 | 0.9429 | 0.0773 |

Current v0 interpretation: output scoring is useful for post-generation audit and compliance
measurement, but it is not currently a positive runtime fusion weight. The project should keep the
output feature interface and evaluation harness, while freezing the v0 input-side fusion around
prompt, embedding, and activation evidence unless later response-level data shows output lift.

Reproduction command pattern:

```bash
.venv/bin/python scripts/batch_generate_output_eval_responses.py \
  --input .artifacts/output_eval/ramp_output_eval_set_v0_1.jsonl \
  --output-jsonl .artifacts/output_eval/ramp_output_eval_set_v0_1.generated.jsonl \
  --output-csv /Users/ratnaditya/Documents/ramp_output_eval_set_v0_1_generated.csv \
  --provider openrouter \
  --model "${RAMP_GENERATION_MODEL:-openai/gpt-oss-20b}" \
  --max-tokens 512 \
  --temperature 0 \
  --resume

.venv/bin/python scripts/batch_score_output_classifier.py \
  --input .artifacts/output_eval/ramp_output_eval_set_v0_1.generated.jsonl \
  --output .artifacts/output_eval/ramp_output_scores_qwen3guard_v0_1.jsonl \
  --provider qwen3guard \
  --model "${RAMP_OUTPUT_RISK_MODEL:-Qwen/Qwen3Guard-Gen-0.6B}" \
  --batch-size 8 \
  --resume

.venv/bin/python scripts/join_output_scores.py \
  --feature-table .artifacts/cumulative_signal_eval/ramp_qwen3guard_prompt_internal_signal_feature_table_expanded_reviewed_activation_v0_1.jsonl \
  --output-scores .artifacts/output_eval/ramp_output_scores_qwen3guard_v0_1.jsonl \
  --output .artifacts/output_eval/ramp_prompt_embedding_activation_output_feature_table_v0_1.jsonl \
  --summary-output .artifacts/output_eval/ramp_output_score_join_v0_1.summary.json

.venv/bin/python scripts/evaluate_calibrated_signal_combinations.py \
  --review-csv /Users/ratnaditya/Documents/ramp_prompt_label_review_combined_v0_1_v0_2_v0_4_cleaned.csv \
  --feature-table .artifacts/output_eval/ramp_prompt_embedding_activation_output_feature_table_v0_1.jsonl \
  --output-json .artifacts/output_eval/ramp_prompt_embedding_activation_output_calibration_v0_1.json \
  --output-md .artifacts/output_eval/ramp_prompt_embedding_activation_output_calibration_v0_1.md \
  --include-output \
  --num-splits 30 \
  --weight-step 0.05 \
  --threshold-step 0.01 \
  --min-prompt-weight 0.0 \
  --max-embedding-weight 1.0 \
  --max-output-weight 1.0
```

### Phase 3: Add Session Signal

Goal: evaluate risk accumulation across turns.

Tasks:

- define session windows when a session identifier exists
- define sliding-window grouping when no session identifier exists
- create multi-turn examples for severity accumulation, harm drift, and composition
- score prompt-level and session-level risk separately
- evaluate whether session evidence catches cases that single-turn classifiers miss

Exit criterion:

- session-risk benchmark and ablation showing whether session evidence adds value

Implementation status: RAMP now has a benchmark-backed session corpus path and a session-risk
accumulator. The current sources are recorded in `data/session_corpus/source_manifest_v0_1.json`.
R-Judge is the first labeled session benchmark imported into the artifact flow; SafeDialBench is
available as an unlabeled or review-mapping corpus, and MHJ remains gated on Hugging Face.

Current R-Judge artifacts:

- `.artifacts/session_eval/ramp_session_eval_corpus_rjudge_v0_1.jsonl`
- `.artifacts/session_eval/ramp_session_eval_corpus_rjudge_v0_1.turns.jsonl`
- `.artifacts/session_eval/ramp_session_eval_corpus_rjudge_v0_1.qwen_turn_scores.jsonl`
- `.artifacts/session_eval/ramp_session_eval_corpus_rjudge_v0_1.qwen_scored.jsonl`
- `.artifacts/session_eval/ramp_session_risk_eval_rjudge_qwen_v0_1.md`

Current v0 result: session accumulation is **not** validated as a positive runtime signal yet.
Using Qwen3Guard turn-level prompt scores on R-Judge, `single_turn_max` has AUC `0.5473` and the
current `session_accumulation` formula has AUC `0.4885`. At a fixed `0.55` threshold,
`single_turn_max` has recall `0.7448` with FPR `0.7732`, while `session_accumulation` has recall
`0.0070` with FPR `0.0074`. The calibrated threshold sweep does not rescue the session formula:
the best-F1 operating point predicts almost every session unsafe, and low-FPR operating points lose
most unsafe sessions.

Interpretation: this is a useful negative result, not a failure of the project. R-Judge labels
multi-turn agent behavior and risk trajectories, while the current turn scorer mostly sees isolated
user text. The next session research step is therefore to improve the per-turn evidence used inside
the session accumulator: output risk, tool/action risk, role-aware transcript scoring, and
session-specific features should be evaluated before assigning a positive v0 weight to session
accumulation. Synthetic sessions remain useful only as mechanism smoke tests.

### Phase 4: Add Tool/Action Evidence

Goal: evaluate agent-specific risk that appears in proposed actions or tool arguments.

Tasks:

- define tool/action schema for evaluation
- create benign and harmful action examples
- test text-safe/action-risk disagreements
- evaluate gating decisions separately from text moderation

Exit criterion:

- tool/action evaluation table and examples of action-level risk not visible from prompt text alone

### Phase 5: Paper Results

Goal: assemble the final RAMP evidence package.

Tasks:

- freeze datasets and artifacts
- run all ablations
- produce result tables and error slices
- document limitations and prior-art positioning
- write the paper narrative around cumulative signal value

## Immediate Next Step

The first evaluation harness now exists around the reviewed feature table:

```bash
python scripts/evaluate_ramp_harness.py \
  --review-csv /Users/ratnaditya/Documents/ramp_qwen3guard_prompt_label_review_batch_500_v0_1.csv \
  --feature-table .artifacts/prompt_label_audit/ramp_qwen3guard_reviewed_cumulative_signal_feature_table_v0_1.jsonl \
  --calibration-artifact .artifacts/prompt_label_audit/ramp_reviewed_fusion_calibration_v0_1.json \
  --calibration-objective selected_by_best_f1 \
  --output-json .artifacts/prompt_label_audit/ramp_reviewed_evaluation_harness_v0_1.json \
  --output-md .artifacts/prompt_label_audit/ramp_reviewed_evaluation_harness_v0_1.md \
  --floor 0.72
```

It produces one consolidated report with:

- prompt, embedding, activation, fixed-fusion, calibrated-fusion, and severity-floor ablations
- calibration objective
- threshold used
- false positives and false negatives
- severe false negatives
- hard benign near-neighbor false positives
- per-domain slices
- artifact IDs and source paths

Once that harness exists, each new feature becomes one more column in the same evaluation table
instead of a separate side quest.

The next implementation step is to expand the harness input data, not tune another local rule:

1. Add more reviewed rows, especially severe harms and hard benign near-neighbors.
2. Split reviewed rows into calibration and holdout partitions.
3. Re-run the same harness on the held-out split.
4. Only then tune weights, floors, or thresholds.

The v0.2 review batch was generated for step 1 with:

```bash
python scripts/build_review_batch_v0_2.py \
  --feature-table .artifacts/cumulative_signal_eval/ramp_qwen3guard_prompt_internal_signal_feature_table_v0_1.jsonl \
  --review-csv /Users/ratnaditya/Documents/ramp_qwen3guard_prompt_label_review_batch_500_v0_1.csv \
  --calibration-artifact .artifacts/prompt_label_audit/ramp_reviewed_fusion_calibration_v0_1.json \
  --calibration-objective selected_by_best_f1 \
  --output-jsonl .artifacts/prompt_label_audit/ramp_prompt_label_review_batch_500_v0_2.jsonl \
  --output-csv .artifacts/prompt_label_audit/ramp_prompt_label_review_batch_500_v0_2.csv \
  --summary-output .artifacts/prompt_label_audit/ramp_prompt_label_review_batch_500_v0_2.summary.json \
  --max-rows 500 \
  --max-per-bucket 250 \
  --max-per-stratum 40 \
  --floor 0.72
```

The generated CSV for review is:

```text
/Users/ratnaditya/Documents/ramp_prompt_label_review_batch_500_v0_2.csv
```

After v0.2 cleanup, the combined reviewed set has 235 binary rows. The first split-aware
calibration/holdout run is:

```bash
python scripts/evaluate_split_calibrated_ramp.py \
  --review-csv /Users/ratnaditya/Documents/ramp_prompt_label_review_combined_v0_1_v0_2_cleaned.csv \
  --feature-table .artifacts/cumulative_signal_eval/ramp_qwen3guard_prompt_internal_signal_feature_table_v0_1.jsonl \
  --output-json .artifacts/prompt_label_audit/ramp_split_calibrated_evaluation_v0_1.json \
  --output-md .artifacts/prompt_label_audit/ramp_split_calibrated_evaluation_v0_1.md \
  --calibration-fraction 0.5 \
  --split-seed ramp_reviewed_split_v0_1 \
  --weight-step 0.05 \
  --threshold-step 0.01 \
  --target-fpr 0.10 \
  --min-prompt-weight 0.40 \
  --max-embedding-weight 0.20 \
  --require-prompt-gte-activation \
  --require-activation-gte-embedding \
  --calibration-objective selected_by_best_f1 \
  --floor 0.72
```

Selected split-calibrated fusion:

```text
prompt weight = 0.60
embedding weight = 0.20
activation weight = 0.20
threshold = 0.40
```

Holdout result:

| Condition | AUC | Accuracy | Recall | FPR | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt only | 0.9470 | 0.9153 | 0.9661 | 0.1356 | 8 | 2 |
| Prompt + embedding | 0.9440 | 0.9237 | 0.9492 | 0.1017 | 6 | 3 |
| Prompt + activation | 0.9305 | 0.9068 | 0.8814 | 0.0678 | 4 | 7 |
| RAMP fusion | 0.9497 | 0.9237 | 0.9661 | 0.1186 | 7 | 2 |

This is the first defensible directionally split result. It supports a narrower claim than the
original ambition: on the targeted reviewed holdout, RAMP fusion preserves prompt-only recall,
slightly improves AUC and accuracy, and reduces false positives from 8 to 7. The improvement is
real but modest, so the next research step should be more reviewed coverage and error slicing, not
a strong final claim.

The single split was followed by a 30-split stability run:

```bash
python scripts/evaluate_split_stability.py \
  --review-csv /Users/ratnaditya/Documents/ramp_prompt_label_review_combined_v0_1_v0_2_cleaned.csv \
  --feature-table .artifacts/cumulative_signal_eval/ramp_qwen3guard_prompt_internal_signal_feature_table_v0_1.jsonl \
  --output-json .artifacts/prompt_label_audit/ramp_split_stability_evaluation_v0_1.json \
  --output-md .artifacts/prompt_label_audit/ramp_split_stability_evaluation_v0_1.md \
  --num-splits 30 \
  --seed-prefix ramp_stability_v0_1 \
  --calibration-fraction 0.5 \
  --weight-step 0.05 \
  --threshold-step 0.01 \
  --target-fpr 0.10 \
  --min-prompt-weight 0.40 \
  --max-embedding-weight 0.20 \
  --require-prompt-gte-activation \
  --require-activation-gte-embedding \
  --calibration-objective selected_by_best_f1 \
  --floor 0.72 \
  --top-k-errors 50
```

Across 30 deterministic splits, RAMP fusion changes the prompt-only baseline as follows:

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

This is more conservative than the single split. RAMP currently buys lower false-positive pressure
at the cost of recall. That is still useful for the research program, because it identifies the
precise tension that the next review batch must resolve.

The v0.3 review batch was generated from the repeated error distribution:

```bash
python scripts/build_review_batch_v0_3.py \
  --feature-table .artifacts/cumulative_signal_eval/ramp_qwen3guard_prompt_internal_signal_feature_table_v0_1.jsonl \
  --review-csv /Users/ratnaditya/Documents/ramp_prompt_label_review_combined_v0_1_v0_2_cleaned.csv \
  --stability-artifact .artifacts/prompt_label_audit/ramp_split_stability_evaluation_v0_1.json \
  --output-jsonl .artifacts/prompt_label_audit/ramp_prompt_label_review_batch_500_v0_3.jsonl \
  --output-csv /Users/ratnaditya/Documents/ramp_prompt_label_review_batch_500_v0_3.csv \
  --summary-output .artifacts/prompt_label_audit/ramp_prompt_label_review_batch_500_v0_3.summary.json \
  --max-rows 500 \
  --max-per-bucket 125 \
  --max-per-stratum 20 \
  --threshold 0.40 \
  --uncertainty-window 0.08 \
  --min-domain-binary-reviewed 10 \
  --max-error-slices 12
```

The generated batch has 359 rows: 53 stable severe-FN slice rows, 41 stable hard-benign-FP slice
rows, 21 RAMP-FN slice rows, 40 RAMP-FP slice rows, 62 fusion/prompt disagreement rows, 89
undercovered-domain rows, and 53 uncertain-margin rows.

After the `bench_comp_0014981` review finding, the taxonomy was patched with
`self_harm_and_wellbeing / suicidal_ideation_or_crisis`, and the centroid path was rerun with the
explicit correction overlay:

```bash
python scripts/apply_taxonomy_corrections.py \
  --input .artifacts/runpod/comprehensive/extracted/ramp-artifacts/embeddings/gpt_oss_20b_input_embedding_v0_1/ramp_benchmark_comprehensive_v0.input_embeddings.jsonl \
  --corrections data/taxonomy/taxonomy_corrections_v0_1.json \
  --output .artifacts/corrections/ramp_benchmark_comprehensive_v0.input_embeddings.taxonomy_corrected_v0_1.jsonl

python scripts/build_embedding_centroids.py \
  --embeddings .artifacts/corrections/ramp_benchmark_comprehensive_v0.input_embeddings.taxonomy_corrected_v0_1.jsonl \
  --embedding-source data/embedding_source/gpt_oss_20b_input_embedding_v0_1.json \
  --taxonomy data/taxonomy/ramp_taxonomy_v0_1.json \
  --output .artifacts/centroids/ramp_input_embedding_centroids_comprehensive_taxonomy_corrected_v0_1.json \
  --artifact-id ramp_input_embedding_centroids_comprehensive_taxonomy_corrected_v0.1
```

The corrected stability run improves the cumulative-signal story:

| Delta vs prompt-only | Old mean | Corrected mean |
| --- | ---: | ---: |
| AUC | -0.0049 | +0.0018 |
| Accuracy | +0.0071 | +0.0105 |
| Recall | -0.0203 | -0.0153 |
| FPR | -0.0345 | -0.0362 |
| FP count | -2.03 | -2.13 |
| FN count | +1.20 | +0.90 |
| Severe FN count | +1.57 | +1.17 |
| Hard benign FP count | -1.43 | -1.20 |

This is exactly why stable error review should feed the taxonomy. The correction does not magically
solve fusion, but it moves the result in the right direction: better AUC, better accuracy, lower
FPR, fewer false positives, and a smaller false-negative penalty.

The follow-up calibrated-combination run tested whether the apparent activation weakness was caused
by the signal itself or by an activation probe trained on noisy benchmark-derived labels. Each
condition was calibrated independently on repeated reviewed-label splits: prompt only, prompt plus
embedding, prompt plus activation, and prompt plus embedding plus activation.

With the original benchmark-trained activation probe, the activation signal was only marginally
helpful:

| Condition | AUC mean | Accuracy mean | Recall mean | FPR mean | FP mean | FN mean | Severe FN mean | Hard benign FP mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt only calibrated | 0.9480 | 0.9167 | 0.9734 | 0.1401 | 8.27 | 1.57 | 1.00 | 5.37 |
| Prompt + embedding calibrated | 0.9501 | 0.9195 | 0.9492 | 0.1102 | 6.50 | 3.00 | 2.43 | 5.00 |
| Prompt + activation calibrated | 0.9493 | 0.9218 | 0.9463 | 0.1028 | 6.07 | 3.17 | 2.60 | 3.90 |
| Prompt + embedding + activation calibrated | 0.9464 | 0.9234 | 0.9514 | 0.1045 | 6.17 | 2.87 | 2.30 | 4.17 |

Before the leakage review, training the layer-19 activation probe on the 235 manually reviewed rows
made activation appear to be the strongest reviewed-label internal signal:

| Condition | AUC mean | Accuracy mean | Recall mean | FPR mean | FP mean | FN mean | Severe FN mean | Hard benign FP mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt only calibrated | 0.9460 | 0.9116 | 0.9672 | 0.1441 | 8.50 | 1.93 | 1.33 | 5.47 |
| Prompt + embedding calibrated | 0.9498 | 0.9155 | 0.9412 | 0.1102 | 6.50 | 3.47 | 2.87 | 4.87 |
| Prompt + activation calibrated | 0.9982 | 0.9788 | 0.9797 | 0.0220 | 1.30 | 1.20 | 1.00 | 1.23 |
| Prompt + embedding + activation calibrated | 0.9978 | 0.9695 | 0.9774 | 0.0384 | 2.27 | 1.33 | 1.30 | 2.03 |

That interpretation is now historical. The later leakage-free rerun showed that this activation
headline depended on in-sample activation probabilities for calibration rows. It remains useful as a
diagnostic for why activation looked promising, but it is not the current policy evidence.
Embeddings still carry taxonomy-aware boundary information, and final paper claims should report
combination-specific leakage-free calibration rather than a single hand-chosen weight vector.

The next reviewed-label expansion is intentionally narrow. Review batch v0.4 samples 221 unreviewed
rows focused on activation false-negative candidates, severe activation false-negative candidates,
embedding false-positive candidates, and embedding/activation conflicts. Once those labels are
complete, rerun reviewed activation-probe training and calibrated-combination stability before
freezing any final prompt/embedding/activation weight policy.

The taxonomy-corrected benchmark-label activation probe is useful as a corpus-scale probe, but it
does not by itself justify a dominant activation weight on reviewed labels. In unconstrained
reviewed-label calibration, the benchmark-trained probe mostly earns modest activation weight
(`0.25` to `0.30` in the most common prompt+activation configurations). The provisional
reviewed-label probe earns much more activation weight and much better reviewed-label performance,
but it is still trained on only 235 binary reviewed rows. This is the current rule for the paper:
activation weight must be earned by reviewed-label calibration, not assigned because later-layer
signals are theoretically appealing.

After the v0.4 review expansion, the reviewed set increased to 448 binary rows. A subsequent
leakage review found that the activation-heavy result was not a valid final policy claim: the
calibration rows had been scored by activation probes trained on the same reviewed rows. The
leakage-free rerun retrains activation probes inside each split, uses out-of-fold activation
predictions for calibration rows, and uses out-of-split predictions for holdout rows.

Under that cross-fitted protocol, the earlier prompt+activation headline is superseded:

| Condition | AUC mean | F1 mean | Recall mean | FPR mean | FP mean | FN mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt only calibrated | 0.9458 | 0.9382 | 0.9631 | 0.1238 | 11.63 | 4.80 |
| Prompt + embedding calibrated | 0.9627 | 0.9423 | 0.9608 | 0.1085 | 10.20 | 5.10 |
| Prompt + activation calibrated | 0.9609 | 0.9365 | 0.9569 | 0.1195 | 11.23 | 5.60 |
| Prompt + embedding + activation calibrated | 0.9641 | 0.9367 | 0.9500 | 0.1078 | 10.13 | 6.50 |

The frozen v0.2 input-side runtime policy is therefore prompt `0.80`, embedding `0.20`,
activation `0.00`, threshold `0.50`. Full prompt+embedding+activation has marginally higher AUROC,
but lower recall and F1, so activation remains an audit/research signal until it proves value on a
blind holdout.
