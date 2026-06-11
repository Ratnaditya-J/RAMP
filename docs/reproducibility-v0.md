# RAMP v0 Reproducibility Guide

This guide documents the current research-v0 reproduction path. It assumes generated artifacts live
under `.artifacts/` and model weights live under ignored local directories or RunPod storage.

## Environment

```bash
cd /Users/ratnaditya/RAMP
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,qwen]"
pytest
ruff check .
```

For local Qwen3Guard prompt/output scoring:

```bash
hf auth login
hf download Qwen/Qwen3Guard-Gen-0.6B --local-dir .models/qwen3guard-gen-0.6b
export RAMP_PROMPT_RISK_MODEL=.models/qwen3guard-gen-0.6b
export RAMP_OUTPUT_RISK_MODEL=.models/qwen3guard-gen-0.6b
```

Model binaries are intentionally ignored by Git.

## Frozen v0 Policy

The current policy artifacts are:

- `data/fusion_policy/ramp_fusion_policy_v0_1.json`
- `data/fusion_policy/ramp_multistage_policy_v0_1.json`

The v0 primary runtime score is prompt `0.25` plus activation `0.75`, threshold `0.53`, with
embedding runtime weight `0.00`. Output and session classifiers are retained as audit/escalation
signals rather than positive v0 blocking weights.

## Consolidated Report

Build the consolidated report from the current artifact set:

```bash
.venv/bin/python scripts/build_v0_consolidated_report.py \
  --output-md docs/reports/ramp_v0_consolidated_research_report.md \
  --output-json .artifacts/reports/ramp_v0_consolidated_research_report.json
```

The committed Markdown report is the reader-facing summary. The JSON output is ignored and can be
regenerated from local artifacts.

## Input-Side Fusion

The selected input-side policy comes from reviewed-label split stability:

```bash
.venv/bin/python scripts/evaluate_calibrated_signal_combinations.py \
  --features .artifacts/cumulative_signal_eval/ramp_qwen3guard_prompt_internal_signal_feature_table_expanded_reviewed_activation_v0_1.jsonl \
  --review-csv /Users/ratnaditya/Documents/ramp_prompt_label_review_combined_v0_1_v0_2_v0_4_cleaned.csv \
  --output-json .artifacts/prompt_label_audit/ramp_calibrated_signal_combinations_expanded_reviewed_activation_v0_1.json \
  --output-md .artifacts/prompt_label_audit/ramp_calibrated_signal_combinations_expanded_reviewed_activation_v0_1.md \
  --num-splits 30 \
  --weight-step 0.05 \
  --threshold-step 0.01
```

## Output Classifier

The output classifier experiment uses generated prompt/response rows:

```bash
.venv/bin/python scripts/batch_score_output_classifier.py \
  --input .artifacts/output_eval/ramp_output_eval_set_v0_1.generated.jsonl \
  --output .artifacts/output_eval/ramp_output_scores_qwen3guard_v0_1.jsonl \
  --provider qwen3guard \
  --model "$RAMP_OUTPUT_RISK_MODEL" \
  --batch-size 8 \
  --resume
```

Then run output-inclusive calibration:

```bash
.venv/bin/python scripts/evaluate_calibrated_signal_combinations.py \
  --features .artifacts/output_eval/ramp_prompt_embedding_activation_output_feature_table_v0_1.jsonl \
  --review-csv /Users/ratnaditya/Documents/ramp_output_eval_set_v0_1.csv \
  --output-json .artifacts/output_eval/ramp_prompt_embedding_activation_output_calibration_refined_v0_1.json \
  --output-md .artifacts/output_eval/ramp_prompt_embedding_activation_output_calibration_refined_v0_1.md \
  --include-output \
  --num-splits 20 \
  --weight-step 0.10 \
  --threshold-step 0.01 \
  --min-prompt-weight 0.0 \
  --max-embedding-weight 0.3 \
  --max-output-weight 0.3
```

## Session Classifier

Build compact and full transcript session classifier inputs:

```bash
.venv/bin/python scripts/build_session_classifier_inputs.py \
  --session-corpus .artifacts/session_eval/ramp_session_eval_corpus_rjudge_v0_1.qwen_scored.jsonl \
  --output .artifacts/session_eval/ramp_session_classifier_inputs_rjudge_compact_state_v0_1.jsonl \
  --mode compact_state

.venv/bin/python scripts/build_session_classifier_inputs.py \
  --session-corpus .artifacts/session_eval/ramp_session_eval_corpus_rjudge_v0_1.qwen_scored.jsonl \
  --output .artifacts/session_eval/ramp_session_classifier_inputs_rjudge_full_transcript_v0_1.jsonl \
  --mode full_transcript
```

Score those inputs with Qwen3Guard on a GPU instance when needed, then evaluate fusion locally:

```bash
.venv/bin/python scripts/evaluate_session_signal_fusion.py \
  --session-corpus .artifacts/session_eval/ramp_session_eval_corpus_rjudge_v0_1.qwen_scored.jsonl \
  --compact-scores .artifacts/session_eval/ramp_session_classifier_scores_rjudge_compact_state_qwen_v0_1.jsonl \
  --full-transcript-scores .artifacts/session_eval/ramp_session_classifier_scores_rjudge_full_transcript_qwen_v0_1.jsonl \
  --output-json .artifacts/session_eval/ramp_session_signal_fusion_eval_rjudge_qwen_v0_1.json \
  --output-md .artifacts/session_eval/ramp_session_signal_fusion_eval_rjudge_qwen_v0_1.md \
  --threshold 0.55 \
  --weight-step 0.05 \
  --threshold-step 0.01 \
  --target-fpr 0.25
```

## Verification

Before publishing a new v0 artifact:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
```
