# RAMP Artifact Registry

This registry records generated research artifacts that are intentionally not committed to Git.
Large corpora, model vectors, activation tensors, archives, and centroid artifacts live under
`.artifacts/` or RunPod storage and are ignored by the repository.

The registry is the source of record for what was generated, how it was generated, and what
limitations are known at the time of the run.

## GPT-OSS Comprehensive Extraction v0

| Field | Value |
| --- | --- |
| Artifact bundle | `ramp-gpt-oss-comprehensive-artifacts-v0.tgz` |
| Local path | `.artifacts/runpod/comprehensive/ramp-gpt-oss-comprehensive-artifacts-v0.tgz` |
| Source environment | RunPod GPU pod |
| GPU | NVIDIA H200 |
| Model | `openai/gpt-oss-20b` |
| Corpus rows | 27,718 |
| Vector dimension | 2,880 |
| SHA256 | `5a14455573e06d92c79d216cf7c0404531ec369555e7cb8334ac2457bd06a103` |
| Archive verification | `gzip -t` passed locally after copy from RunPod |

Bundle contents:

```text
ramp-artifacts/corpora/ramp_benchmark_comprehensive_v0.jsonl
ramp-artifacts/embeddings/gpt_oss_20b_input_embedding_v0_1/ramp_benchmark_comprehensive_v0.input_embeddings.jsonl
ramp-artifacts/activations/gpt_oss_20b_hidden_state_v0_1/ramp_benchmark_comprehensive_v0.layer_12.jsonl
ramp-artifacts/activations/gpt_oss_20b_hidden_state_v0_1/ramp_benchmark_comprehensive_v0.layer_19.jsonl
ramp-artifacts/activations/gpt_oss_20b_hidden_state_v0_1/ramp_benchmark_comprehensive_v0.layer_final.jsonl
```

Extraction roles:

| File | Representation | Purpose |
| --- | --- | --- |
| `ramp_benchmark_comprehensive_v0.input_embeddings.jsonl` | `input_embedding` | Source of record for input-embedding centroid construction. |
| `ramp_benchmark_comprehensive_v0.layer_12.jsonl` | `hidden_state` | Mid-layer activation probe candidate. |
| `ramp_benchmark_comprehensive_v0.layer_19.jsonl` | `hidden_state` | Late-layer activation probe candidate. |
| `ramp_benchmark_comprehensive_v0.layer_final.jsonl` | `hidden_state` | Final hidden-state probe/ablation candidate. |

The input embedding file uses the project definition of embeddings: GPT-OSS token IDs mapped
through `model.get_input_embeddings()(input_ids)`, followed by attention-mask mean pooling and
L2 normalization.

Hidden-state files are activation artifacts, not the primary embedding-centroid source.

## Corpus Mix

The comprehensive corpus was built from benchmark-derived spans:

| Source | Rows | Role contribution |
| --- | ---: | --- |
| `wildguardmix` | 18,622 | Mostly harmful/evasion |
| `harmbench` | 393 | Harmful/evasion |
| `beavertails` | 7,766 | Benign near-neighbor |
| `do_not_answer` | 937 | Benign near-neighbor |

Label and role totals:

| Group | Rows | Share |
| --- | ---: | ---: |
| Harmful | 18,834 | 67.95% |
| Benign near-neighbor | 8,703 | 31.40% |
| Evasion | 181 | 0.65% |

The corpus is intentionally not a uniform sample of benign language. Benign examples are used as
hard contrast anchors near harmful neighborhoods, not as a broad model of all safe requests.

## Input Embedding Centroid Artifact v0.1

| Field | Value |
| --- | --- |
| Artifact ID | `ramp_input_embedding_centroids_comprehensive_v0.1` |
| Local path | `.artifacts/centroids/ramp_input_embedding_centroids_comprehensive_v0_1.json` |
| Created at | `2026-06-07T18:09:32.052850+00:00` |
| Embedding source | `gpt_oss_20b_input_embedding_v0.1` |
| Taxonomy | `ramp_taxonomy_v0.1` |
| Input rows | 27,718 |
| Centroids | 19 |
| Dimension | 2,880 |
| Method | Mean of L2-normalized span vectors per `domain/subcluster_role/subcluster_id`, followed by L2 normalization. |

Centroids by role:

| Role | Centroids | Rows |
| --- | ---: | ---: |
| Harmful | 13 | 18,834 |
| Benign near-neighbor | 5 | 8,703 |
| Evasion | 1 | 181 |

Centroid inventory:

| Rows | Domain | Role | Subcluster | Sources |
| ---: | --- | --- | --- | --- |
| 11,372 | `nonviolent_illegal_activity` | `harmful` | `organized_abuse_workflows` | `harmbench: 252`, `wildguardmix: 11120` |
| 8,538 | `regulated_advice` | `benign_near_neighbor` | `general_information` | `beavertails: 7639`, `do_not_answer: 899` |
| 2,412 | `regulated_advice` | `harmful` | `unsafe_professional_instruction` | `harmbench: 6`, `wildguardmix: 2406` |
| 1,071 | `cyber_abuse` | `harmful` | `vulnerability_exploitation` | `harmbench: 16`, `wildguardmix: 1055` |
| 780 | `ip_and_content_rights` | `harmful` | `copyright_reproduction` | `harmbench: 50`, `wildguardmix: 730` |
| 646 | `weapons_and_physical_violence` | `harmful` | `weapon_construction` | `harmbench: 17`, `wildguardmix: 629` |
| 607 | `child_safety` | `harmful` | `sexualized_minors` | `harmbench: 8`, `wildguardmix: 599` |
| 422 | `sexual_safety_and_content` | `harmful` | `explicit_adult_generation` | `harmbench: 7`, `wildguardmix: 415` |
| 382 | `nonviolent_illegal_activity` | `harmful` | `fraud_scams` | `harmbench: 9`, `wildguardmix: 373` |
| 372 | `hate_harassment_and_abuse` | `harmful` | `targeted_harassment` | `harmbench: 5`, `wildguardmix: 367` |
| 326 | `privacy_identity_and_secrets` | `harmful` | `pii_extraction` | `harmbench: 1`, `wildguardmix: 325` |
| 181 | `agent_tool_and_system_integrity` | `evasion` | `jailbreak` | `harmbench: 2`, `wildguardmix: 179` |
| 162 | `self_harm_and_wellbeing` | `harmful` | `suicide_methods` | `harmbench: 4`, `wildguardmix: 158` |
| 152 | `cbrn_and_hazardous_materials` | `harmful` | `chemical_misuse_procedure` | `harmbench: 8`, `wildguardmix: 144` |
| 130 | `misinformation_manipulation_and_civic` | `harmful` | `election_falsehoods` | `harmbench: 8`, `wildguardmix: 122` |
| 69 | `cyber_abuse` | `benign_near_neighbor` | `defensive_security` | `beavertails: 50`, `do_not_answer: 19` |
| 66 | `weapons_and_physical_violence` | `benign_near_neighbor` | `historical_analysis` | `beavertails: 54`, `do_not_answer: 12` |
| 27 | `cbrn_and_hazardous_materials` | `benign_near_neighbor` | `lab_safety` | `beavertails: 22`, `do_not_answer: 5` |
| 3 | `privacy_identity_and_secrets` | `benign_near_neighbor` | `redaction` | `beavertails: 1`, `do_not_answer: 2` |

Warnings:

```json
[
  {
    "domain": "privacy_identity_and_secrets",
    "subcluster_role": "benign_near_neighbor",
    "subcluster_id": "redaction",
    "count": 3,
    "warning": "below_min_count_warning"
  }
]
```

## Interpretation

This run should be treated as `benchmark-derived v0`, not as final calibrated production
centroids.

The harmful/benign asymmetry is expected and intentional. RAMP does not attempt to model all benign
language with centroids. It models harmful neighborhoods plus selected benign near-neighbors that
act as contrastive anchors for borderline safety decisions.

Quality questions for the next phase:

- For each harmful centroid, which benign near-neighbor is closest?
- Which harmful domains lack a nearby benign contrast anchor?
- Which harmful and benign centroids collide so closely that centroid scoring alone is unreliable?
- Which false-positive benchmark examples are nearest to harmful centroids?
- What score thresholds preserve recall while reducing false positives on hard benign neighbors?

## Reproduction Commands

Build centroids locally from the extracted input embedding file:

```bash
python scripts/build_embedding_centroids.py \
  --embeddings .artifacts/runpod/comprehensive/extracted/ramp-artifacts/embeddings/gpt_oss_20b_input_embedding_v0_1/ramp_benchmark_comprehensive_v0.input_embeddings.jsonl \
  --embedding-source data/embedding_source/gpt_oss_20b_input_embedding_v0_1.json \
  --taxonomy data/taxonomy/ramp_taxonomy_v0_1.json \
  --output .artifacts/centroids/ramp_input_embedding_centroids_comprehensive_v0_1.json \
  --artifact-id ramp_input_embedding_centroids_comprehensive_v0.1 \
  --min-count-warning 25
```

Verify the bundle checksum:

```bash
shasum -a 256 .artifacts/runpod/comprehensive/ramp-gpt-oss-comprehensive-artifacts-v0.tgz
cat .artifacts/runpod/comprehensive/ramp-gpt-oss-comprehensive-artifacts-v0.tgz.sha256
gzip -t .artifacts/runpod/comprehensive/ramp-gpt-oss-comprehensive-artifacts-v0.tgz
```
