from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum

from ramp.risk_state import RiskState
from ramp.schemas.provenance import RuntimeProvenance


def make_state(
    request_id: str,
    session_id: str,
    *,
    action_stakes: str = "low",
) -> RiskState:
    provenance = RuntimeProvenance(
        request_id=request_id,
        session_id=session_id,
        policy_version="policy_v0",
        risk_fusion_version="risk_fusion_v0",
    )
    return RiskState(
        request_id=request_id,
        session_id=session_id,
        provenance=provenance,
        action_stakes=action_stakes,
    )


def print_decision(decision) -> None:
    def encode(value):
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, tuple):
            return [encode(item) for item in value]
        if isinstance(value, dict):
            return {encode(key): encode(item) for key, item in value.items()}
        return value

    print(json.dumps(encode(asdict(decision)), indent=2, sort_keys=True))
