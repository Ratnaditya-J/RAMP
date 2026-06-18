# Signal Validity Card — embedding

- axis: `evaluation_robustness`  (vocabulary: no_value, leak_inflated, in_distribution_only, distribution_robust)
- verdict: **in_distribution_only**  (status: ok)
- scope: GPT-OSS input-embedding centroid proximity, n=448, front_door=qwen3guard_0.6b, probe=linear
- blind labels: silver_llm_inter_judge_validated (inter-judge kappa=0.810)

## Allowed claims
- the signal adds value under leakage-free in-distribution evaluation

## Disallowed claims
- claim the signal generalizes under distribution shift
- deploy an in-distribution-tuned threshold without shift testing
- claim validation against human-labeled blind data

## Residual risks
- value does not robustly pass BOTH out-of-distribution rungs (blind random sample and held-out source); see per-rung diagnostics
- blind rung uses LLM-judge silver labels (inter-judge kappa=0.810); human-audited at kappa=0.573 (moderate; residual is genuine borderline-case variance, not rubric failure)

config_hash: `sha256:db6c8059a687bf4929529a76d2738e96d0a93067e0c5dad0259a95effd30c1fd`
inputs_hash: `sha256:065a70c099e8a028aa385af4e103893f5ad00b5349e0d9d6145bee48246fd237`

---

# Signal Validity Card — activation

- axis: `evaluation_robustness`  (vocabulary: no_value, leak_inflated, in_distribution_only, distribution_robust)
- verdict: **leak_inflated**  (status: ok)
- scope: GPT-OSS layer-19 linear probe, n=448, front_door=qwen3guard_0.6b, probe=linear
- blind labels: silver_llm_inter_judge_validated (inter-judge kappa=0.810)

## Allowed claims
- the signal's apparent value appears only under leakage-prone evaluation

## Disallowed claims
- claim the signal earns runtime weight
- report naive/split numbers as the signal's value

## Residual risks
- leaky-rung metrics overstate the signal's value

config_hash: `sha256:db6c8059a687bf4929529a76d2738e96d0a93067e0c5dad0259a95effd30c1fd`
inputs_hash: `sha256:065a70c099e8a028aa385af4e103893f5ad00b5349e0d9d6145bee48246fd237`

---

# Signal Validity Card — full_fusion

- axis: `evaluation_robustness`  (vocabulary: no_value, leak_inflated, in_distribution_only, distribution_robust)
- verdict: **in_distribution_only**  (status: ok)
- scope: prompt + embedding + activation calibrated fusion, n=448, front_door=qwen3guard_0.6b, probe=linear
- blind labels: silver_llm_inter_judge_validated (inter-judge kappa=0.810)

## Allowed claims
- the signal adds value under leakage-free in-distribution evaluation

## Disallowed claims
- claim the signal generalizes under distribution shift
- deploy an in-distribution-tuned threshold without shift testing
- claim validation against human-labeled blind data

## Residual risks
- value does not robustly pass BOTH out-of-distribution rungs (blind random sample and held-out source); see per-rung diagnostics
- blind rung uses LLM-judge silver labels (inter-judge kappa=0.810); human-audited at kappa=0.573 (moderate; residual is genuine borderline-case variance, not rubric failure)

config_hash: `sha256:db6c8059a687bf4929529a76d2738e96d0a93067e0c5dad0259a95effd30c1fd`
inputs_hash: `sha256:065a70c099e8a028aa385af4e103893f5ad00b5349e0d9d6145bee48246fd237`
