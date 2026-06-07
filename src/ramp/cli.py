from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum

from ramp.features import FeatureInput
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

