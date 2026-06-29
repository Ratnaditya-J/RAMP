# Fragility Study: v0.2 Results (supplement)

> v0.2 supplement to `docs/fragility-study-results-v0_1.md`, covering results added after
> v0.1: cross-family multi-model generalization, a positive control, and a decomposition
> of the headline Qwen3Guard AUROC drop. v0.1 remains the source for the survival table,
> the adaptive-vs-blind headline, and the label-standard (human-audit) findings.

This document consolidates three result sets that, prior to this file, lived only in
gitignored `.artifacts/`. Each section gives the headline numbers, the provenance (the
artifact file / script that produced them), the exact reproduction command(s), and the
paper section the result backs. No number here is introduced that is not present in the
cited source artifact; verdicts are quoted exactly.

---

## 1. Multi-model cross-family generalization (backs paper §3, closes §7 single-family limitation)

**Source:** `.artifacts/multimodel/MULTIMODEL_RESULTS.md`
(extractor `scripts/extract_multimodel_activations.py`; runner `scripts/run_multimodel_ladder.py`;
ladder reports + SVC cards under `.artifacts/multimodel/ladder/<model>/`).

**Question.** Does GPT-OSS-20b's `leak_inflated` activation-probe verdict generalize to other
model families, or is it GPT-OSS-specific?

**Method (as recorded in the source).** Layer hidden states (masked mean-pool, L2-norm, bf16,
raw `span_text`, max_len 512) extracted from 3 open models on the SAME 1,559 reviewed prompts
(adaptive 1,059 + blind 500) used in the main study, on a RunPod H100 (~19 min wall, pod
terminated immediately after download). Identical survival ladder vs the identical Qwen3Guard
prompt-only baseline + labels; ONLY the `--activation` source changed (n = 448 adaptive / 369
blind, same as GPT-OSS; prompt-only baseline byte-identical across all four, crossfit AUROC
0.9448). Best layer selected by leakage-free crossfit AUROC.

### Per-model verdicts

| Model | Params / arch | Selected layer | Per-rung (activation) | SVC verdict | `distribution_robust`? |
|---|---|---|---|---|---|
| GPT-OSS-20b | 20B MoE | 19 | naive✓ split✓ crossfit~mixed blind~mixed shifted✗ | `leak_inflated` | NO |
| Llama-3.1-8B | 8B dense | 10 | ✓ ✓ mixed mixed ✗ | `leak_inflated` | NO |
| Mistral-7B | 7B dense | 10 | ✓ ✓ mixed mixed ✗ | `leak_inflated` | NO |
| Phi-4 | 14B dense | 20 | ✓ ✓ ✓ ✓ mixed | `in_distribution_only` | NO |

In-distribution (crossfit) activation AUROC ≈ 0.96–0.97 for every model.

**Key claim (verified, quoted from source).** Across 4 families (OpenAI / Meta / Mistral /
Microsoft), 7–20B params, MoE + dense: the activation probe is **NEVER `distribution_robust`** —
it never survives the distribution-shift rung. Its near-perfect in-distribution AUROC is an
evaluation artifact in every family tested.

**Tier is layer-sensitive, not a per-model property.** The `leak_inflated` vs
`in_distribution_only` tier turns on a sub-0.002 crossfit-F1 delta (≈0.15σ of per-split noise);
a one-probed-layer step (10→16) flips Llama and Mistral from `leak_inflated` to
`in_distribution_only`, neither of which is robust:
- Llama L16: ✓ ✓ ✓ blind=mixed shifted=mixed → `in_distribution_only`
- Mistral L16: ✓ ✓ ✓ blind=✓ shifted=fails → `in_distribution_only`

So the robust, model-independent claim is the **ceiling** (never `distribution_robust`), not the
exact failure tier.

**Verification (from source).** The harness reproduces GPT-OSS's canonical `leak_inflated`
verdict exactly (correctness gate, `"match": true`). A critical agent independently re-derived
every verdict from the reports, confirmed run integrity (all 5 rungs completed, n = 448/369,
byte-identical baseline), and caught that the per-model tier-ranking is a layer-selection
artifact (corrected above).

**Caveats (from source).** Tier layer-sensitivity (above); single labeled benchmark with silver
(LLM-judge) blind labels (human-validated κ 0.62 in the main study); layer selection by crossfit
AUROC, with not-robust confirmed at each model's robustness-candidate (F1-positive) layer.

### Exact commands to reproduce

```bash
cd /Users/ratnaditya/RAMP
# 1) extract per-model mean-pooled activations on the reviewed prompts (GPU)
python scripts/extract_multimodel_activations.py
# 2) run the survival ladder + SVC for each model
python scripts/run_multimodel_ladder.py
```

(The source records the GPU extraction was run on a RunPod H100; the ladder/SVC step is local.
Activations are written under `.artifacts/multimodel/activations/{llama31_8b,mistral7b_v03,phi4}/*.layer_*.jsonl`;
ladder reports + SVC cards under `.artifacts/multimodel/ladder/<model>/`.)

---

## 2. Positive control — a genuinely robust signal certified `distribution_robust` (backs paper §3)

**Source:** `.artifacts/positive_control/POSITIVE_CONTROL_RESULTS.md`
(SVC produced by `ramp.audit`; bundle at `.artifacts/positive_control/bundle/positive_control.bundle.json`,
audit at `.artifacts/positive_control/bundle/positive_control.audit.json`;
data manifest `.artifacts/positive_control/data_manifest.json`).

**Purpose.** The mirror image of the safety activation probe. A demotion tool is only
trustworthy if it also *certifies* — i.e. if a genuinely robust signal comes back
`distribution_robust`. The object under test is the **same KIND** as the safety probe — a
linear logistic-regression probe on a small generative model's mean-pooled mid-layer
activations, extracted with the byte-identical pooling recipe (masked mean over non-pad tokens
→ L2-normalize).

**Object under test (from source).** Property/label: is the text **English (1)** vs
**Spanish (0)** — near-domain-invariant, strongly linearly decodable. Candidate signal: pure-numpy
L2-regularized logistic-regression probe (standardized features) on **`gpt2` layer-7**
(depth 0.583, mid-band), mean-pooled + L2-normalized, vector dim 768. Baseline to beat:
**`ascii_letter_fraction`** (fraction of alphabetic chars that are ASCII a–z), a-priori
AUROC ≈ 0.78 — informative-but-imperfect, not a strawman. Dataset: `mteb/amazon_massive_scenario`
(parquet mirror of AmazonScience/MASSIVE), configs `en` + `es`, `train` split. **2,880** examples,
balanced **1,440 English / 1,440 Spanish**, **160 per domain** (80 EN + 80 ES) across the **18
MASSIVE scenarios** (parallel locales, so leave-one-domain-out tests property transfer, not source
memorization).

### Per-rung table (candidate = probe, baseline = ascii-fraction heuristic)

AUROC/F1 and the OOD paired-bootstrap significance are re-derived by RAMP's own machinery inside
`LadderBundle.from_raw_scores` (`ramp.audit.stats._fast_auc` / `_f1_at` / seeded
`paired_bootstrap`, 2,000 resamples). "OOD sig" = bootstrap 95% CI on ΔAUROC excludes zero.

| rung | cand AUROC | base AUROC | ΔAUROC | cand F1 | base F1 | ΔF1 | OOD ΔAUROC CI95 | OOD sig | survival |
|---|---|---|---|---|---|---|---|---|---|
| naive    | 1.0000 | 0.7715 | +0.2285 | 0.9986 | 0.8140 | +0.1846 | — | — | **survives** |
| split    | 0.9997 | 0.7729 | +0.2268 | 0.9921 | 0.8055 | +0.1866 | — | — | **survives** |
| crossfit | 0.9998 | 0.7715 | +0.2283 | 0.9955 | 0.8140 | +0.1815 | — | — | **survives** |
| blind    | 0.9999 | 0.7600 | +0.2399 | 0.9954 | 0.8100 | +0.1854 | [+0.2153, +0.2629] | **yes** | **survives** |
| shifted  | 0.9998 | 0.7715 | +0.2283 | 0.9955 | 0.8140 | +0.1815 | [+0.2159, +0.2421] | **yes** | **survives** |

The probe is at AUROC ≈ 1.0 on every rung, including leakage-free crossfit and
leave-one-domain-out shifted. It dominates the (non-trivial) baseline by ΔAUROC ≈ +0.23 and
ΔF1 ≈ +0.18 at every rung, with both OOD lifts bootstrap-significant.

### Final `ramp.audit` SVC verdict (quoted exactly)

```
verdict : distribution_robust
status  : ok
reasons :
  - survives leakage-free in-distribution (crossfit)
  - robustly passes blind and shifted (significant AUROC lift)
allowed   : the signal adds value under leakage-free, blind, and shifted evaluation
disallowed: claim causal sufficiency (out of scope for this axis; see SIEVE)
residual  : (none)
```

Contrast with the safety activation probe on the same tool: `leak_inflated` (survives
naive/split, fails crossfit). Same machinery, opposite verdict — RAMP's ladder discriminates.

### Integrity checks (both pass, from source)

1. **Per-domain transfer is uniform** — every one of the 18 held-out domains gives AUROC
   **0.9997–1.0000** (min 0.9997, mean 0.9999), each balanced 80 EN / 80 ES. The robust verdict
   is not driven by one easy domain.
2. **Negative control (shuffled-label check, no pipeline leakage)** — shuffling the labels
   collapses crossfit AUROC to **0.4919** (≈ chance), vs **0.9998** with real labels. The
   pipeline is reading a genuine property, not manufacturing one.

**Honesty notes / caveats (from source).** This is a *certification-capability* control, not a
clean one-variable swap: the control also differs from the safety audit in dataset
(MASSIVE n=2,880 vs reviewed prompts n=448/369), label provenance (objective EN/ES,
`blind_label_provenance="human"`, vs silver LLM-judge labels), model scale (gpt2 124M vs 7–20B),
dtype (fp32 vs bf16), and max-len (128 vs 512). Language ID is near-perfectly separable from
surface text (a char-frequency baseline reaches AUROC ≈ 0.98), so the control shows the ladder
*can* certify a robust same-KIND signal — not that the activation probe beats cheap text features.
Single property, single small model, single dataset family.

### Exact commands to reproduce

```bash
cd /Users/ratnaditya/RAMP

# 0) one-time: isolated CPU extraction venv (the RAMP .venv cannot import transformers)
python3 -m venv .artifacts/positive_control/extract_venv
.artifacts/positive_control/extract_venv/bin/python -m pip install --upgrade pip
.artifacts/positive_control/extract_venv/bin/python -m pip install \
    torch transformers "numpy<2" \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

# 1) build the domain-tagged language-ID corpus (RAMP .venv: has datasets + numpy)
PYTHONPATH=src .venv/bin/python .artifacts/positive_control/scripts/01_prepare_data.py

# 2) extract mean-pooled gpt2 layer-7 activations on CPU (isolated venv)
.artifacts/positive_control/extract_venv/bin/python \
    .artifacts/positive_control/scripts/02_extract_activations.py \
    --model gpt2 --layer 7 \
    --corpus .artifacts/positive_control/corpus_langid.jsonl \
    --output .artifacts/positive_control/activations/gpt2.layer_7.jsonl

# 3) run the survival ladder + ramp.audit SVC (RAMP .venv: numpy + ramp.audit)
PYTHONPATH=src .venv/bin/python \
    .artifacts/positive_control/scripts/03_run_ladder_audit.py \
    --activation .artifacts/positive_control/activations/gpt2.layer_7.jsonl
```

**Environment (from source).** Extraction: isolated CPU venv (py3.9.6), torch 2.8.0 (CPU),
transformers 4.57.6, numpy 1.26.4. Analysis: RAMP `.venv` (py3.12.13), numpy 2.4.6,
datasets 5.0.0, `ramp.audit`. No GPU; gpt2 forward passes on CPU (~1 min for 2,880 utterances).

---

## 3. Confound decomposition of the 0.945 → 0.850 drop (backs paper §4.1 and §7)

**Source:** `.artifacts/confound_decomp/decomp_result.json`
(produced by `.artifacts/confound_decomp/decompose_qwen3guard_drop.py`).
This is the only artifact for this result — there is no companion `.md`; the headline numbers,
the apportionment, and the corrected-framing note are all fields inside the JSON.

**Question.** Of Qwen3Guard's headline AUROC drop from 0.945 (adaptive × human) to 0.850
(blind × judge), how much is genuine sampling/distribution shift versus a change in label
source (human → LLM judge)?

**Estimands (from JSON).** Per-row guard score = `prompt_risk_score`; binary label mapping
`unsafe->1, safe->0, else dropped`. Four AUROCs:
- **A = adaptive × human:** AUROC **0.9437** (`A_adaptive_human.auc`), CI95 [0.9204, 0.9651], n=448 (260 unsafe / 188 safe).
- **B = blind × judge (full):** AUROC **0.8501** (`B_blind_judge_full.auc`), CI95 [0.8194, 0.8802], n=369 (147 unsafe / 222 safe).
- **C = blind × human (160-row audit):** AUROC **0.9231** (`C_blind_human_160.auc`), n=132 (117 unsafe / 15 safe); paired-basis C = **0.9121** (`auc_C_paired_basis`).
- **D = blind × judge on the same ids as C:** AUROC **0.7768** (`D_blind_judge_same_ids.auc`), n=105 (49 unsafe / 56 safe).

The self-gate passed: A reproduces the paper's ~0.945 and B the ~0.850 (`self_gate.passed = true`,
A_target 0.945, B_target 0.85).

**Headline drop (from JSON).** A − B (mixed basis) = **0.0935** (`decomposition.headline_A_minus_B_mixed_basis.delta`),
CI95 [0.0563, 0.1305], `significant: true`, 10,000 resamples. This contrast mixes a sampling
shift AND a label-source change on different row counts (448 vs 369); A−C and C−D are the
single-factor contrasts.

**Single-factor contrasts (from JSON).**
- **Sampling shift, A − C (unpaired):** delta **0.0206** (`sampling_shift_A_minus_C_unpaired.delta`), CI95 [−0.0182, 0.0604], `significant: false`.
- **Label source, C − D (paired, on the audited subset):** delta **0.1353** (`label_source_C_minus_D_paired.delta`), CI95 [0.0407, 0.2287], `significant: true`, on the n=105 paired intersection (source_ids binary under BOTH human and judge).
- Paired label flips human→judge (`paired_label_flips_human_to_judge`): human-safe/judge-safe **14**, human-unsafe/judge-unsafe **49**, human-unsafe/judge-safe **42**.
- Additivity check A − D (paired): **0.1669** (`additivity_check_A_minus_D_paired`).

**Apportionment (~73% / ~27%) and the correction (quoted from `validation_note`).**

> CORRECTED after adversarial review (2026-06-27): the label-source-dominates framing holds ONLY
> on the disagreement-dense 160-row audited subset. Post-stratified to the full blind set,
> label-source accounts for only ~0.025 of the 0.094 drop (~27 percent); ~73 percent is genuine
> sampling/distribution shift. The paper section 4.1 distribution-shift framing STANDS; see
> section 7.

So: on the **full** blind set, of the ≈0.094 drop, **~0.025 (~27%) is label source** and
**~73% is sampling/distribution shift**. (Arithmetic check: 0.025 / 0.0935 = 0.267, leaving
0.733.) This reconciles the two bases — the **C−D paired contrast on the disagreement-dense
audited subset** shows label source *dominating* there (0.1353), but **post-stratified to the
full blind set** the label-source share shrinks to ~27%, with sampling shift the dominant ~73%.

**Caveat (load-bearing).** The full-set ~73%/~27% apportionment is **approximate**: it is a
post-stratification (extrapolation) from the 160-row human audit to the full blind set, not a
direct full-set measurement. The clean single-factor contrasts (A−C sampling, C−D label source)
are measured only on their respective row sets; A−C is not statistically significant while C−D is
(on the audited subset). The middle anchor (C) is the 160-row human audit; the paper's §4.1
distribution-shift framing stands, and §7 carries this caveat.

**Bootstrap config (from JSON).** 10,000 resamples, seed prefix `ramp_confound_decomp_v0.1`.

### Exact command to reproduce

```bash
cd /Users/ratnaditya/RAMP
# the script inserts src/ and scripts/ on sys.path itself; writes decomp_result.json next to it
.venv/bin/python .artifacts/confound_decomp/decompose_qwen3guard_drop.py
```

Inputs (module constants in the script): feature table
`.artifacts/cumulative_signal_eval/ramp_qwen3guard_prompt_internal_signal_feature_table_expanded_reviewed_activation_v0_1.jsonl`;
labels `data/reviewed/ramp_prompt_label_review_combined_v0_1_v0_2_v0_4_cleaned.csv` (A),
`data/reviewed/ramp_blind_review_batch_v0_1.judge_labeled.csv` (B/D),
`data/reviewed/ramp_blind_audit_combined_v0_1_v0_2.labeled.csv` (C). AUROC (`_fast_auc`) and the
paired bootstrap come from `src/ramp/audit/stats.py` — the same primitives the survival ladder used.

---

## Paper-section cross-reference

| Result | Backs paper section |
|---|---|
| Multi-model cross-family generalization (§1 above) | §3 (multi-model demotion), closes §7 single-target-family limitation |
| Positive control `distribution_robust` (§2 above) | §3 (the certification positive control) |
| Confound decomposition 0.945→0.850, ~73%/~27% (§3 above) | §4.1 (headline drop) and §7 (decomposition + caveat) |
