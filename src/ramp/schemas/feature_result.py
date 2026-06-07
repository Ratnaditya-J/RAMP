from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FeatureStage(str, Enum):
    POLICY_HEURISTICS = "policy_heuristics"
    PROMPT_RISK = "prompt_risk_feature"
    EMBEDDING_RISK = "embedding_risk_feature"
    ACTIVATION_PROBE = "activation_probe_feature"
    OUTPUT_RISK = "output_risk_feature"
    SESSION_DRIFT = "session_drift_feature"
    TOOL_ACTION_RISK = "tool_action_risk_feature"


@dataclass(frozen=True)
class FeatureResult:
    """One partial signal emitted by a RAMP feature extractor."""

    stage: FeatureStage
    risk_score: float
    confidence: float
    label: str | None = None
    harm_category: str | None = None
    latency_ms: int | None = None
    cost_tier: int = 1
    version: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError("risk_score must be between 0 and 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.cost_tier < 0:
            raise ValueError("cost_tier must be non-negative")

