# Activation Probe Feature

The activation probe feature tests whether GPT-OSS hidden states add useful internal-state evidence
to RAMP's accumulated risk estimate. It does not replace prompt or embedding signals. Embedding
proximity provides an early semantic neighborhood prior; the activation probe asks whether the
model's later internal processing makes unsafe intent more separable.

The probe is intentionally simple:

```text
p(unsafe | activation_vector) = sigmoid(W * standardized_activation + b)
```

This keeps the experiment focused on representation quality rather than classifier complexity.

## Training

Train and compare probes for the three extracted GPT-OSS layers:

```bash
python scripts/train_activation_probes.py \
  --activation .artifacts/runpod/comprehensive/extracted/ramp-artifacts/activations/gpt_oss_20b_hidden_state_v0_1/ramp_benchmark_comprehensive_v0.layer_12.jsonl \
  --layer 12 \
  --activation .artifacts/runpod/comprehensive/extracted/ramp-artifacts/activations/gpt_oss_20b_hidden_state_v0_1/ramp_benchmark_comprehensive_v0.layer_19.jsonl \
  --layer 19 \
  --activation .artifacts/runpod/comprehensive/extracted/ramp-artifacts/activations/gpt_oss_20b_hidden_state_v0_1/ramp_benchmark_comprehensive_v0.layer_final.jsonl \
  --layer final \
  --embedding-baseline .artifacts/centroids/ramp_input_embedding_linear_baseline_v0_1.json \
  --output-dir .artifacts/activation_probes \
  --report-output .artifacts/activation_probes/ramp_activation_probe_layer_comparison_v0_1.json \
  --epochs 30 \
  --learning-rate 0.10 \
  --l2 0.001 \
  --seed 7
```

The script writes one probe artifact per layer and a comparison report:

```text
.artifacts/activation_probes/ramp_activation_probe_layer_12_v0_1.json
.artifacts/activation_probes/ramp_activation_probe_layer_19_v0_1.json
.artifacts/activation_probes/ramp_activation_probe_layer_final_v0_1.json
.artifacts/activation_probes/ramp_activation_probe_layer_comparison_v0_1.json
```

## Current Result

The random-split v0.1 result selects layer 19:

| Feature | AUC | Recall at <=5% FPR | FPR | Threshold |
| --- | ---: | ---: | ---: | ---: |
| input embedding linear baseline | n/a | 0.4265 | 0.0023 | 0.51 |
| activation layer 12 | 0.9926 | 0.9766 | 0.0500 | 0.22 |
| activation layer 19 | 0.9953 | 0.9869 | 0.0466 | 0.18 |
| activation final layer | 0.9940 | 0.9750 | 0.0494 | 0.09 |

Interpretation: GPT-OSS hidden-state activations add strong later evidence on the current
benchmark-derived corpus. Layer 19 is the best candidate under the selection rule: highest recall at
or below 5% false-positive rate, then AUC.

## Holdout Validation

Run source-held-out or domain-held-out validation with:

```bash
python scripts/validate_activation_probe_holdouts.py \
  --activation .artifacts/runpod/comprehensive/extracted/ramp-artifacts/activations/gpt_oss_20b_hidden_state_v0_1/ramp_benchmark_comprehensive_v0.layer_19.jsonl \
  --layer 19 \
  --holdout-key source \
  --output .artifacts/activation_probes/ramp_activation_probe_layer_19_source_holdouts_v0_1.json \
  --epochs 30 \
  --learning-rate 0.10 \
  --l2 0.001 \
  --seed 7
```

The source mix is asymmetric:

| Source | Labels |
| --- | --- |
| `beavertails` | safe only |
| `do_not_answer` | safe only |
| `harmbench` | unsafe only |
| `wildguardmix` | unsafe only |

Because of that, source-held-out validation is a label-specific stress test rather than an AUC
test. A safe-only source reports false-positive rate; an unsafe-only source reports recall.

Layer 19 source-held-out result:

| Held-out source | Held-out label | Recall | FPR |
| --- | --- | ---: | ---: |
| `beavertails` | safe | 0.0000 | 0.0595 |
| `do_not_answer` | safe | 0.0000 | 0.4290 |
| `harmbench` | unsafe | 0.9669 | 0.0000 |
| `wildguardmix` | unsafe | 0.6583 | 0.0000 |

Layer 19 domain-held-out summary:

| Layer | Mean recall at train-selected threshold | Mean AUC on domains with both labels |
| --- | ---: | ---: |
| `12` | 0.9463 | 0.9534 |
| `19` | 0.9758 | 0.9851 |
| `final` | 0.9217 | 0.8636 |

Layer 19 remains the best current layer after holdout stress. The result is strong, but not
production-final:

- `do_not_answer` safe examples false-positive heavily under source holdout.
- `wildguardmix` unsafe recall drops under source holdout.
- small benign-neighbor domains such as privacy and weapons have unstable FPR because they contain
  very few safe examples.

The defensible interpretation is:

```text
Layer 19 activations add materially stronger later-stage evidence to the accumulated risk state,
but the probe needs source-balanced and domain-balanced validation before being used as a
standalone decision feature.
```

## Cumulative Internal-Signal Evaluation

RAMP's internal-signal thesis is cumulative, not winner-takes-all. Evaluate embedding and activation
signals together with:

```bash
python scripts/evaluate_cumulative_internal_signals.py \
  --prompt-scores .artifacts/prompt_scores/ramp_benchmark_comprehensive_v0.qwen3guard_0_6b_prompt_scores.jsonl \
  --embedding-scores .artifacts/centroids/ramp_input_embedding_centroids_comprehensive_v0_1.centered_domain_conditioned_scores.jsonl \
  --activation .artifacts/runpod/comprehensive/extracted/ramp-artifacts/activations/gpt_oss_20b_hidden_state_v0_1/ramp_benchmark_comprehensive_v0.layer_19.jsonl \
  --activation-probe .artifacts/activation_probes/ramp_activation_probe_layer_19_v0_1.json \
  --output-json .artifacts/cumulative_signal_eval/ramp_prompt_internal_signal_ablation_v0_1.json \
  --output-md .artifacts/cumulative_signal_eval/ramp_prompt_internal_signal_ablation_v0_1.md \
  --feature-table .artifacts/cumulative_signal_eval/ramp_prompt_internal_signal_feature_table_v0_1.jsonl
```

The evaluator reports:

- embedding-only semantic prior performance
- activation-only internal-state evidence performance
- fixed cumulative fusion performance
- prompt-only and prompt-plus-internal ablations when `--prompt-scores` is provided
- hard benign false-positive rows
- domain and source slices

This report should be read as incremental-value evidence: embedding proximity identifies the
semantic neighborhood and activation probability adds later model-state evidence.

Current cumulative internal-signal result:

| Ablation | AUC | Recall at <=5% FPR | FPR |
| --- | ---: | ---: | ---: |
| embedding only | 0.9465 | 0.7870 | 0.0467 |
| activation only | 0.9959 | 0.9876 | 0.0494 |
| cumulative fixed fusion | 0.9953 | 0.9859 | 0.0465 |

The fixed fusion score is not meant to prove that every metric improves monotonically. It preserves
embedding as semantic prior/context and activation as stronger later evidence inside the same
accumulated risk estimate. The cumulative score keeps activation-level recall while slightly
reducing FPR at the selected operating point.

A local keyword prompt-classifier dry run validates the full prompt + embedding + activation
pipeline, but is not a research classifier result:

| Ablation | AUC | Recall at <=5% FPR | FPR |
| --- | ---: | ---: | ---: |
| prompt only | 0.5151 | 0.0686 | 0.0387 |
| prompt + embedding | 0.9163 | 0.6992 | 0.0477 |
| prompt + activation | 0.9950 | 0.9776 | 0.0334 |
| prompt + embedding + activation | 0.9947 | 0.9828 | 0.0476 |

The next research artifact should replace the keyword prompt scores with Qwen3Guard batch scores.

## Runtime Feature

`ActivationProbeFeature` loads a trained linear probe artifact and scores a supplied activation
vector:

```python
from ramp.features import ActivationProbeFeature, FeatureInput

feature = ActivationProbeFeature.from_artifact(
    ".artifacts/activation_probes/ramp_activation_probe_layer_19_v0_1.json"
)

result = feature.extract(
    FeatureInput(
        prompt="...",
        context={"activation_vector": layer_19_vector},
    ),
    state,
)
```

The default provider is `ContextActivationProvider`, which reads `activation_vector` from
`FeatureInput.context`. A live GPT-OSS runtime adapter can provide the same vector from an
instrumented forward pass once the target model runtime is wired into RAMP.

Feature metadata records:

- `probe_artifact_id`
- `selected_layer`
- `activation_provider_version`
- `unsafe_probability`
- `selected_threshold`
- `activated`
- `training_summary`

## Research Use

The activation result is the strongest current evidence for RAMP's cumulative internal-signal thesis:

```text
Input embeddings locate broad semantic neighborhoods.
Later activations add internal model-state evidence.
RAMP fuses both rather than choosing one as the only signal.
```

The next research steps are:

1. Build a source-balanced held-out set with both safe and unsafe examples per source family.
2. Build domain-balanced benign near-neighbors for sparse domains.
3. Train domain-specific activation probes where enough data exists.
4. Calibrate activation probe output into the final RAMP score fusion.
5. Wire live GPT-OSS activation extraction for the selected layer.
