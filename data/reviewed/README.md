# Reviewed Label Artifacts

This directory contains the human-reviewed prompt-label CSVs used for RAMP v0
calibration and audit experiments. The files are committed so the reviewed-label
population is no longer dependent on local spreadsheet state.

## Files

- `ramp_qwen3guard_prompt_label_review_batch_500_v0_1.csv`: first reviewed
  prompt-label audit batch.
- `ramp_prompt_label_review_batch_500_v0_2_cleaned.csv`: mechanically cleaned
  second review batch.
- `ramp_prompt_label_review_batch_500_v0_4_cleaned.csv`: mechanically cleaned
  expanded hard-case review batch.
- `ramp_prompt_label_review_combined_v0_1_v0_2_v0_4_cleaned.csv`: combined
  prompt-label review set used by the latest prompt/internal-signal calibration.
- `SHA256SUMS`: SHA-256 digests for the committed CSV files.

## Label Columns

- `review_status`: use rows with `reviewed`.
- `reviewed_label`: one of `safe`, `unsafe`, `controversial`,
  `ambiguous_or_context_needed`, or `bad_benchmark_label`.
- `label_issue_type`: optional reviewer diagnosis for corpus/model-label issues.
- `source_id`: source row id used to join against generated feature tables.

Only binary `safe` and `unsafe` reviewed labels are used for AUROC/FPR/recall
calibration. Non-binary reviewed labels remain in the CSVs for provenance and
future taxonomy analysis.
