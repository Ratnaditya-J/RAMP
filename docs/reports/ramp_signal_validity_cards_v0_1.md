# Signal Validity Card — embedding

- axis: `evaluation_robustness`  (vocabulary: no_value, leak_inflated, in_distribution_only, distribution_robust)
- verdict: **in_distribution_only**  (status: ok)
- scope: GPT-OSS input-embedding centroid proximity, n=448, front_door=qwen3guard_0.6b, probe=linear
- blind labels: silver_llm (inter-judge kappa=0.887)

## Allowed claims
- the signal adds value under leakage-free in-distribution evaluation

## Disallowed claims
- claim the signal generalizes under distribution shift
- deploy an in-distribution-tuned threshold without shift testing
- claim validation against human-labeled blind data

## Residual risks
- value not demonstrated out-of-distribution / on a blind random sample
- blind rung uses LLM-judge silver labels (inter-judge kappa=0.887); human audit pending

config_hash: `sha256:db6c8059a687bf4929529a76d2738e96d0a93067e0c5dad0259a95effd30c1fd`
inputs_hash: `sha256:dca1f0b634d4aabc2fb1cbb9d0997e5eb6d01c9f5c4a3228856b03dab74a5829`

---

# Signal Validity Card — activation

- axis: `evaluation_robustness`  (vocabulary: no_value, leak_inflated, in_distribution_only, distribution_robust)
- verdict: **leak_inflated**  (status: ok)
- scope: GPT-OSS layer-19 linear probe, n=448, front_door=qwen3guard_0.6b, probe=linear
- blind labels: silver_llm (inter-judge kappa=0.887)

## Allowed claims
- the signal's apparent value appears only under leakage-prone evaluation

## Disallowed claims
- claim the signal earns runtime weight
- report naive/split numbers as the signal's value

## Residual risks
- leaky-rung metrics overstate the signal's value

config_hash: `sha256:db6c8059a687bf4929529a76d2738e96d0a93067e0c5dad0259a95effd30c1fd`
inputs_hash: `sha256:dca1f0b634d4aabc2fb1cbb9d0997e5eb6d01c9f5c4a3228856b03dab74a5829`

---

# Signal Validity Card — full_fusion

- axis: `evaluation_robustness`  (vocabulary: no_value, leak_inflated, in_distribution_only, distribution_robust)
- verdict: **in_distribution_only**  (status: ok)
- scope: prompt + embedding + activation calibrated fusion, n=448, front_door=qwen3guard_0.6b, probe=linear
- blind labels: silver_llm (inter-judge kappa=0.887)

## Allowed claims
- the signal adds value under leakage-free in-distribution evaluation

## Disallowed claims
- claim the signal generalizes under distribution shift
- deploy an in-distribution-tuned threshold without shift testing
- claim validation against human-labeled blind data

## Residual risks
- value not demonstrated out-of-distribution / on a blind random sample
- blind rung uses LLM-judge silver labels (inter-judge kappa=0.887); human audit pending

config_hash: `sha256:db6c8059a687bf4929529a76d2738e96d0a93067e0c5dad0259a95effd30c1fd`
inputs_hash: `sha256:dca1f0b634d4aabc2fb1cbb9d0997e5eb6d01c9f5c4a3228856b03dab74a5829`
