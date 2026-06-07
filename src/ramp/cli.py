from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from enum import Enum

from ramp.features import DEFAULT_QWEN3GUARD_MODEL, FeatureInput, Qwen3GuardPromptRiskFeature
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
