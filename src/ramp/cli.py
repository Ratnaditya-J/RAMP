from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from enum import Enum

from ramp.features import (
    DEFAULT_QWEN3GUARD_MODEL,
    EmbeddingClusterRiskFeature,
    FeatureInput,
    GPTOSSInputEmbeddingProvider,
    Qwen3GuardPromptRiskFeature,
    SpanExtractor,
)
from ramp.pipeline import default_pipeline
from ramp.risk_state import RiskState
from ramp.schemas.provenance import RuntimeProvenance


def _encode(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {_encode(key): _encode(item) for key, item in value.items()}
    return value


def single_turn_demo() -> None:
    request_id = "req_cli_demo"
    session_id = "sess_cli_demo"
    state = RiskState(
        request_id=request_id,
        session_id=session_id,
        provenance=RuntimeProvenance(request_id=request_id, session_id=session_id),
    )
    decision = default_pipeline().evaluate(
        FeatureInput(prompt="Can you explain safe password manager best practices?"),
        state,
    )
    print(json.dumps(_encode(asdict(decision)), indent=2, sort_keys=True))


def prompt_risk() -> None:
    parser = argparse.ArgumentParser(description="Score one prompt with Qwen3Guard.")
    parser.add_argument("prompt", help="Prompt text to classify")
    parser.add_argument(
        "--model",
        default=DEFAULT_QWEN3GUARD_MODEL,
        help=f"Hugging Face model id. Defaults to {DEFAULT_QWEN3GUARD_MODEL}.",
    )
    args = parser.parse_args()

    request_id = "req_prompt_risk_cli"
    session_id = "sess_prompt_risk_cli"
    state = RiskState(
        request_id=request_id,
        session_id=session_id,
        provenance=RuntimeProvenance(request_id=request_id, session_id=session_id),
    )
    feature = Qwen3GuardPromptRiskFeature(model_id=args.model).extract(
        FeatureInput(prompt=args.prompt),
        state,
    )
    print(json.dumps(_encode(asdict(feature)), indent=2, sort_keys=True))


def embedding_risk() -> None:
    parser = argparse.ArgumentParser(description="Score one prompt with embedding centroids.")
    parser.add_argument("prompt", help="Prompt text to score")
    parser.add_argument("--centroids", required=True, help="Centroid artifact JSON path")
    parser.add_argument(
        "--provider",
        choices=["gpt-oss", "keyword"],
        default="gpt-oss",
        help="Embedding provider to use. Defaults to the GPT-OSS input-embedding provider.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Hugging Face model ID or local model path for GPT-OSS scoring.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--trigger-margin", type=float, default=0.18)
    parser.add_argument(
        "--span-mode",
        choices=["all", "full", "sentence", "full_sentence", "windows"],
        default="all",
        help="Span extraction strategy for runtime scoring.",
    )
    parser.add_argument(
        "--similarity-mode",
        choices=["cosine", "centered_cosine"],
        default="cosine",
        help="Similarity space for centroid scoring.",
    )
    parser.add_argument(
        "--benign-contrast-mode",
        choices=["domain_conditioned", "any_domain"],
        default="domain_conditioned",
        help="How to choose the benign contrast centroid for risk margins.",
    )
    args = parser.parse_args()

    if args.provider == "gpt-oss":
        embedding_provider = GPTOSSInputEmbeddingProvider(
            model_id=args.model or "openai/gpt-oss-20b",
            dtype=args.dtype,
            device_map=args.device_map,
            max_length=args.max_length,
        )
    else:
        embedding_provider = None

    request_id = "req_embedding_risk_cli"
    session_id = "sess_embedding_risk_cli"
    state = RiskState(
        request_id=request_id,
        session_id=session_id,
        provenance=RuntimeProvenance(request_id=request_id, session_id=session_id),
    )
    feature = EmbeddingClusterRiskFeature.from_centroid_artifact(
        args.centroids,
        embedding_provider=embedding_provider,
        span_extractor=SpanExtractor.from_mode(args.span_mode),
        trigger_margin=args.trigger_margin,
        similarity_mode=args.similarity_mode,
        benign_contrast_mode=args.benign_contrast_mode,
    ).extract(FeatureInput(prompt=args.prompt), state)
    print(json.dumps(_encode(asdict(feature)), indent=2, sort_keys=True))
