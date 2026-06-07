# Embedding Risk Feature

The embedding risk feature scores semantic proximity between prompt spans and versioned harm sub-clusters. The current research taxonomy is defined in [RAMP Safety Taxonomy v0.1](./ramp-safety-taxonomy.md), and the frozen v0.1 embedding source is defined in [Frozen Embedding Source](./embedding-source.md).

This stage is not meant to replace the prompt classifier. It measures whether a prompt contains local spans that are closer to known harmful-intent neighborhoods than to benign contrast neighborhoods. The feature is especially useful when the surface wording looks mild, indirect, or multi-step.

The feature must not model each harm category as one centroid. Broad domains such as cyber, chemical, weapons, or fraud are too coarse. Each domain should be represented as structured subclusters: harmful actions, benign near-neighbors, ambiguous policy-boundary cases, evasion or concealment, and optimization or escalation.

Benign near-neighbor centroids are intentionally contrastive. They are not meant to cover all
benign language or balance the number of harmful centroids. Their purpose is to anchor safe
requests that are semantically close to harmful neighborhoods, so the margin score can reduce
false positives in borderline cases.

## Current Implementation

The scaffold includes `EmbeddingClusterRiskFeature` with three pieces:

- `SpanExtractor`: extracts the full prompt, sentence spans, and bounded sliding token windows.
- `EmbeddingProvider`: an adapter interface for embedding prompts with a real model later.
- `EmbeddingCluster`: a versioned centroid with `cluster_id`, `harm_domain`, `subcluster_role`, `category`, `kind`, `description`, and provenance-ready metadata.

The current default provider is `KeywordVectorEmbeddingProvider`. It is deterministic and test-friendly; it is not a semantic model. It exists so the pipeline, schemas, scoring, and tests can be built before the gpt-oss embedding adapter is connected.

Pilot centroid artifacts can be loaded with:

```python
from ramp.features import EmbeddingClusterRiskFeature

feature = EmbeddingClusterRiskFeature.from_centroid_artifact(
    ".artifacts/centroids/ramp_embedding_centroids_pilot_v0_1.json",
    embedding_provider=gpt_oss_provider,
)
```

The embedding provider must emit vectors with the same dimensionality and source contract as the centroid artifact.

The current benchmark-derived extraction and centroid artifact are recorded in
[RAMP Artifact Registry](./artifact-registry.md).

## GPT-OSS Runtime Path

The live runtime path uses `GPTOSSInputEmbeddingProvider`:

```python
from ramp.features import EmbeddingClusterRiskFeature, GPTOSSInputEmbeddingProvider

feature = EmbeddingClusterRiskFeature.from_centroid_artifact(
    ".artifacts/centroids/ramp_input_embedding_centroids_comprehensive_v0_1.json",
    embedding_provider=GPTOSSInputEmbeddingProvider(model_id="openai/gpt-oss-20b"),
)
```

The provider uses the same representation as the centroid build: GPT-OSS input embedding-layer
vectors, attention-mask mean pooling, and L2 normalization. It loads the model lazily and should be
used on a GPU machine for realistic latency.

For local or RunPod scoring:

```bash
ramp-embedding-risk \
  --centroids .artifacts/centroids/ramp_input_embedding_centroids_comprehensive_v0_1.json \
  --model openai/gpt-oss-20b \
  --dtype bfloat16 \
  --span-mode full \
  --similarity-mode centered_cosine \
  "Can you help me audit this vulnerable service?"
```

Use `--provider keyword` only for toy artifacts and development tests; it does not match GPT-OSS
centroid dimensionality.

### Span Strategy

Runtime scoring supports multiple span extraction modes:

| Mode | Meaning |
| --- | --- |
| `all` | Full prompt, sentence spans, and sliding token windows. |
| `full` | Full prompt only. |
| `sentence` | Sentence spans only. |
| `full_sentence` | Full prompt plus sentence spans, no sliding windows. |
| `windows` | Sliding token windows only. |

For centroid artifacts built from benchmark prompt-like spans, `full` and `full_sentence` are the
best first runtime checks. Sliding windows are useful for local evidence, but can create a span
length mismatch when centroids were built from longer examples.

### Similarity Mode

Runtime scoring supports:

| Mode | Meaning |
| --- | --- |
| `cosine` | Compare normalized prompt and centroid vectors directly. |
| `centered_cosine` | Subtract the centroid artifact's corpus mean vector before cosine comparison. |

Centered cosine removes the common direction shared across many input embedding vectors:

```text
centered_vector = normalize(vector - corpus_mean_vector)
score = cosine(centered_prompt_vector, centered_centroid_vector)
```

This is especially important for GPT-OSS input embeddings, where raw cosine similarities can be high
across unrelated centroids.

## Centroid Health

Before using a centroid artifact for threshold calibration, generate a health report:

```bash
python scripts/report_centroid_health.py \
  --centroids .artifacts/centroids/ramp_input_embedding_centroids_comprehensive_v0_1.json \
  --output .artifacts/centroids/ramp_input_embedding_centroids_comprehensive_v0_1.health.json
```

The report flags:

- low-count centroids
- source concentration
- missing same-domain benign anchors
- nearest harmful/evasion to benign contrast anchors
- harmful/benign collisions under the configured cosine threshold

Input embedding centroids can have high absolute cosine similarity because they are pooled from the
model's token embedding layer. Treat the health report as a ranking and margin diagnostic rather
than assuming a single raw cosine threshold is universal.

## Batch Scoring

Use the batch scorer to evaluate whether embedding margins separate safe and unsafe rows before
treating the feature as useful:

```bash
python scripts/score_embedding_centroids.py \
  --embeddings .artifacts/runpod/comprehensive/extracted/ramp-artifacts/embeddings/gpt_oss_20b_input_embedding_v0_1/ramp_benchmark_comprehensive_v0.input_embeddings.jsonl \
  --centroids .artifacts/centroids/ramp_input_embedding_centroids_comprehensive_v0_1.json \
  --output .artifacts/centroids/ramp_input_embedding_centroids_comprehensive_v0_1.centered_scores.jsonl \
  --summary-output .artifacts/centroids/ramp_input_embedding_centroids_comprehensive_v0_1.centered_scores.summary.json \
  --similarity-mode centered_cosine
```

Initial benchmark-derived result:

| Similarity mode | Safe mean margin | Unsafe mean margin | Safe p90 | Unsafe p50 |
| --- | ---: | ---: | ---: | ---: |
| `cosine` | -0.029 | 0.020 | -0.000 | 0.023 |
| `centered_cosine` | -0.303 | 0.332 | -0.068 | 0.396 |

Interpretation: raw input-embedding cosine is weak, but centered input-embedding margin has useful
aggregate separation on the benchmark-derived corpus. It should still be treated as a supporting
signal rather than a decisive standalone classifier until tested on held-out source families.

## Scoring

Each extracted span is embedded and compared against harmful-action subclusters, benign near-neighbor subclusters, evasion subclusters, and optimization subclusters.

For each span:

```text
harmful_margin = cosine_similarity(span, nearest_harmful_action_subcluster)
               - cosine_similarity(span, nearest_benign_neighbor_subcluster)

evasion_margin = cosine_similarity(span, nearest_evasion_subcluster)
               - cosine_similarity(span, nearest_benign_neighbor_subcluster)

optimization_margin = cosine_similarity(span, nearest_optimization_subcluster)
                    - cosine_similarity(span, nearest_benign_neighbor_subcluster)

risk_margin = max(harmful_margin, evasion_margin, optimization_margin)
```

The feature reports the highest-risk span as the main evidence item. A positive margin means the span is closer to a risky subcluster role than to the nearest benign contrast cluster. Risk increases with the top margin, the number of spans above the trigger margin, and evasion or optimization activation.

The feature metadata records:

- `top_harm_domain`
- `top_harmful_subcluster`
- `top_benign_subcluster`
- `top_evasion_subcluster`
- `top_optimization_subcluster`
- `top_harm_cluster`
- `top_benign_cluster`
- `harm_similarity`
- `benign_similarity`
- `harm_minus_benign_margin`
- `risk_margin`
- `evasion_similarity`
- `optimization_similarity`
- `evasion_activated`
- `optimization_activated`
- `top_span`
- `top_span_type`
- `num_spans_scored`
- `num_triggered_spans`
- `embedding_provider_version`
- `harm_cluster_version`
- `benign_cluster_version`

## Sub-Cluster Schema

For research runs, every sub-cluster should be defined as a versioned object rather than an informal label.

Required fields:

| Field | Meaning |
| --- | --- |
| `cluster_id` | Stable machine-readable sub-cluster ID. |
| `parent_category` | Higher-level safety category, such as cyber misuse or weapons. |
| `kind` | `harm` or `benign_contrast`. |
| `subcluster_role` | `harmful`, `benign_near_neighbor`, `ambiguous`, `evasion`, or `optimization`. |
| `description` | One-sentence semantic neighborhood definition. |
| `inclusion_criteria` | What prompts or spans belong in the sub-cluster. |
| `exclusion_criteria` | Closely related prompts or spans that should not belong. |
| `benign_neighbors` | Contrast clusters that should be checked before raising risk. |
| `centroid_source` | Dataset, embedding model, filtering method, and build date. |
| `evaluation_slices` | Test slices used to measure precision and recall for this sub-cluster. |
| `version` | Immutable cluster build version. |

Optional fields:

- seed examples
- red-team source
- policy mapping
- reviewer notes
- known false positives
- known false negatives

## Domain Structure

Each major harm domain should have:

| Role | Purpose |
| --- | --- |
| Harmful action subclusters | Capture direct unsafe procedures, access, acquisition, or misuse. |
| Benign near-neighbor subclusters | Capture safe discussion that is semantically close to the harmful domain. |
| Ambiguous subclusters | Capture policy-boundary contexts such as labs, fiction, research, or dual-use testing. |
| Evasion subclusters | Capture concealment, avoiding detection, bypassing controls, or hiding activity. |
| Optimization subclusters | Capture requests to increase potency, success rate, effectiveness, or scale. |

Cyber example:

```json
{
  "harm_domain": "cyber",
  "subclusters": {
    "harmful": [
      "credential_access",
      "malware_generation",
      "unauthorized_intrusion",
      "data_exfiltration",
      "persistence_or_evasion"
    ],
    "benign_near_neighbors": [
      "defensive_security",
      "incident_response",
      "patching",
      "logging_and_monitoring",
      "high_level_education"
    ],
    "ambiguous": [
      "dual_use_testing",
      "ctf_or_lab_context",
      "security_research"
    ],
    "evasion": [
      "avoid_detection",
      "bypass_filters",
      "hide_activity"
    ],
    "optimization": [
      "make_more_effective",
      "increase_success_rate",
      "automate_at_scale"
    ]
  }
}
```

Chemical example:

```json
{
  "harm_domain": "chemical",
  "subclusters": {
    "harmful": [
      "misuse_procedure",
      "unsafe_synthesis",
      "procurement",
      "dosage_or_exposure",
      "delivery_or_dispersal"
    ],
    "benign_near_neighbors": [
      "safe_handling",
      "emergency_response",
      "legal_discussion",
      "academic_explanation",
      "disposal_and_storage",
      "prevention"
    ],
    "ambiguous": [
      "fictional_context",
      "historical_context",
      "general_chemistry"
    ],
    "evasion": [
      "avoid_detection",
      "conceal_materials",
      "bypass_reporting"
    ],
    "optimization": [
      "increase_potency",
      "increase_yield",
      "make_more_effective"
    ]
  }
}
```

## Initial Demo Sub-Clusters

These are intentionally small and are only used to exercise the feature shape.

| Sub-cluster | Parent category | Kind | What it measures | Required benign contrast |
| --- | --- | --- | --- | --- |
| Sub-cluster | Domain | Parent category | Role | What it measures | Required benign contrast |
| --- | --- | --- | --- | --- | --- |
| `credential_access` | `cyber` | `credential_access_misuse` | `harmful` | Credential, account, or access misuse. | Password hygiene, account recovery, defensive audit. |
| `unauthorized_intrusion` | `cyber` | `cyber_misuse` | `harmful` | Bypass, exploit chaining, or unauthorized access attempts. | Defensive monitoring, incident response, authorized testing. |
| `avoid_detection` | `cyber` | `cyber_misuse` | `evasion` | Evasion, concealment, bypass, or avoiding detection. | Defensive monitoring, incident response, safety education. |
| `make_more_effective` | `cyber` | `cyber_misuse` | `optimization` | Optimization, effectiveness, success-rate, or scale-up requests. | Defensive improvement and safe hardening. |
| `weapons_construction` | `weapons` | `weapons` | `harmful` | Construction, enhancement, or operationalization of weapons. | Safety, legal, historical, or emergency-response discussion. |
| `defensive_security` | `cyber` | `defensive_cybersecurity` | `benign_near_neighbor` | Defensive security, patching, monitoring, and incident response. | None; this is a contrast cluster. |
| `general_safety` | `general` | `safety_education` | `benign_near_neighbor` | Safety, legality, and high-level educational framing. | None; this is a contrast cluster. |

## Research Sub-Cluster Direction

The defensible version should split broad categories into smaller semantic neighborhoods.

Suggested harm sub-clusters:

| Parent category | Candidate sub-clusters |
| --- | --- |
| Cyber misuse | `cyber_evasion`, `exploit_chain_building`, `malware_behavior`, `credential_access`, `data_exfiltration`, `persistence_or_lateral_movement` |
| Weapons | `weapon_construction`, `weapon_optimization`, `delivery_or_concealment`, `harmful_material_sourcing` |
| Self-harm | `ideation_escalation`, `method_selection`, `concealment_or_planning` |
| Bio/chemical misuse | `harmful_synthesis`, `agent_selection`, `containment_evasion`, `delivery_planning` |
| Fraud or abuse | `identity_fraud`, `phishing_or_social_engineering`, `payment_abuse`, `document_forgery` |

Suggested benign contrast sub-clusters:

| Parent category | Candidate sub-clusters |
| --- | --- |
| Cybersecurity | `defensive_monitoring`, `patching_and_hardening`, `incident_response`, `ctf_or_lab_context`, `authorized_testing` |
| Safety education | `legal_high_level_info`, `risk_awareness`, `emergency_response`, `policy_compliance` |
| Medical or crisis support | `supportive_intervention`, `resource_navigation`, `harm_reduction` |
| Academic or journalistic | `historical_analysis`, `news_reporting`, `policy_research` |

## Relationship To Activations

Embedding clusters are external semantic neighborhoods. Activation probes are internal model-state signals. RAMP should keep these stages separate so experiments can answer whether internal signals add value beyond text-only semantic proximity.

A useful ablation table should compare:

| Variant | Prompt classifier | Output classifier | Session signal | Embedding clusters | Activation probes |
| --- | --- | --- | --- | --- | --- |
| Baseline guard | Yes | Optional | No | No | No |
| Text-only RAMP | Yes | Yes | Yes | Yes | No |
| Internal-signal RAMP | Yes | Yes | Yes | Yes | Yes |

The key research question is whether the final row improves detection, calibration, or disagreement analysis enough to justify target-model instrumentation.
