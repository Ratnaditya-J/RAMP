from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ramp.schemas.feature_result import FeatureStage


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecommendedAction(str, Enum):
    ALLOW = "allow"
    ALLOW_FAST_PATH = "allow_fast_path"
    CONTINUE_EVALUATION = "continue_evaluation"
    CAUTION = "caution"
    SAFE_COMPLETE = "safe_complete"
    REWRITE = "rewrite"
    BLOCK = "block"
    ESCALATE = "escalate"
    RESTRICT_TOOLS = "restrict_tools"
    REQUIRE_HUMAN_APPROVAL = "require_human_approval"
    UPDATE_FUTURE_SESSION_RISK = "update_future_session_risk"
    LOG_FOR_AUDIT = "log_for_audit"


@dataclass(frozen=True)
class RiskDecision:
    request_id: str
    session_id: str
    current_risk: float
    confidence: float
    risk_level: RiskLevel
    recommended_action: RecommendedAction
    features_seen: tuple[FeatureStage, ...]
    features_missing: tuple[FeatureStage, ...]
    disagreements: tuple[str, ...] = ()
    next_required_feature: FeatureStage | None = None
    feature_contributions: dict[FeatureStage, float] = field(default_factory=dict)
    fusion_metadata: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
