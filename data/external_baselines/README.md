# External Guard & Inter-Judge Artifacts

Committed because they cost API spend to regenerate and are provenance for the fragility
study's hardening results (see `docs/fragility-study-results-v0_1.md`). Scores cover the
union of source_ids in the adaptive reviewed set and the blind set.

- `ramp_guard_gemini25pro_eval_scores_v0_1.jsonl`: Gemini 2.5 Pro used as a continuous
  guard (unsafe-probability). Independent lineage from Qwen (prompt), Claude (blind judge),
  and GPT-OSS (activations). Used as the strong-guard front door in the survival ladder.
- `ramp_guard_llamaguard4_eval_scores_v0_1.jsonl`: Llama Guard 4 12B, binary verdict
  (0.0/1.0) + category codes. Dedicated-guard operating-point baseline.
- `ramp_blind_interjudge_agreement_v0_1.json`: Cohen's kappa between the Claude Opus 4.8
  and Gemini 2.5 Pro judges on the blind set (kappa = 0.887, n = 338).
- `SHA256SUMS`: digests for the above.

Regenerate with `scripts/score_prompts_external_guard.py` and
`scripts/compare_label_agreement.py`. These are model outputs, not human ground truth.
