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

## Sampling Provenance

These CSVs are targeted review artifacts, not a random deployment-distribution
test set.

- `v0_1` started from Qwen3Guard prompt-label audit candidates where benchmark
  labels, prompt-classifier scores, or source metadata suggested likely prompt
  label noise.
- `v0_2` expanded the reviewed set with additional disagreement and uncertain
  rows from the prompt-label audit workflow, then received mechanical CSV
  cleanup before use.
- `v0_4` was intentionally adaptive: it focused on activation false-negative
  candidates, severe activation false-negative candidates, embedding false
  positives, and embedding/activation conflict candidates found during split
  stability analysis.
- The combined file is the union of those reviewed batches after mechanical
  cleanup. It is appropriate for hard-case calibration and error analysis, but
  any paper or deployment claim must describe it as adaptively sampled and
  should validate the selected policy on a future blind holdout.
