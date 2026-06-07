from __future__ import annotations

import json
from pathlib import Path

INPUT_EMBEDDING_SOURCE_PATH = Path("data/embedding_source/gpt_oss_20b_input_embedding_v0_1.json")
HIDDEN_STATE_SOURCE_PATH = Path("data/embedding_source/gpt_oss_20b_hidden_state_v0_1.json")
TAXONOMY_PATH = Path("data/taxonomy/ramp_taxonomy_v0_1.json")


def test_embedding_source_freezes_gpt_oss_20b_input_embedding_contract() -> None:
    source = json.loads(INPUT_EMBEDDING_SOURCE_PATH.read_text(encoding="utf-8"))
    taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))

    assert source["embedding_source_id"] == "gpt_oss_20b_input_embedding_v0.1"
    assert source["model"]["huggingface_model_id"] == "openai/gpt-oss-20b"
    assert source["model"]["role_in_ramp"]
    assert source["runtime"]["primary_environment"] == "RunPod GPU pod"
    assert source["runtime"]["extraction_runtime_for_embeddings"]
    assert source["extraction"]["representation"] == "input_embedding"
    assert source["extraction"]["embedding_layer_policy"]
    assert source["extraction"]["pooling"] == "attention-mask mean pooling over non-padding tokens"
    assert (
        source["extraction"]["normalization"]
        == "l2_normalize pooled vector before centroid construction"
    )
    assert source["centroid_build"]["taxonomy"] == "data/taxonomy/ramp_taxonomy_v0_1.json"
    assert source["centroid_build"]["taxonomy"].endswith("ramp_taxonomy_v0_1.json")
    assert taxonomy["taxonomy_id"] == "ramp_taxonomy_v0.1"

    required_provenance = {
        "embedding_source_id",
        "representation",
        "huggingface_model_id",
        "model_revision",
        "tokenizer_revision",
        "transformers_version",
        "torch_version",
        "cuda_version",
        "gpu_name",
        "dtype",
        "pooling",
        "normalization",
        "source_corpus_version",
        "taxonomy_id",
        "created_at",
    }
    assert required_provenance <= set(source["provenance_required_in_outputs"])


def test_hidden_state_source_remains_available_for_ablation() -> None:
    source = json.loads(HIDDEN_STATE_SOURCE_PATH.read_text(encoding="utf-8"))

    assert source["embedding_source_id"] == "gpt_oss_20b_hidden_state_v0.1"
    assert source["model"]["huggingface_model_id"] == "openai/gpt-oss-20b"
    assert source["extraction"]["hidden_state_layer_policy"]
