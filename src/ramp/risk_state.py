from __future__ import annotations

from dataclasses import dataclass, field

from ramp.schemas.feature_result import FeatureResult, FeatureStage
from ramp.schemas.provenance import RuntimeProvenance
from ramp.schemas.risk_decision import RecommendedAction

REQUIRED_STAGES: tuple[FeatureStage, ...] = (
    FeatureStage.PROMPT_RISK,
    FeatureStage.EMBEDDING_RISK,
    FeatureStage.ACTIVATION_PROBE,
    FeatureStage.OUTPUT_RISK,
    FeatureStage.SESSION_DRIFT,
    FeatureStage.TOOL_ACTION_RISK,
)


@dataclass
class RiskState:
    request_id: str
    session_id: str
    provenance: RuntimeProvenance
    latency_budget_ms: int = 150
    action_stakes: str = "low"
    current_risk: float = 0.0
    confidence: float = 0.0
    recommended_action: RecommendedAction = RecommendedAction.CONTINUE_EVALUATION
    features: list[FeatureResult] = field(default_factory=list)
    disagreements: list[str] = field(default_factory=list)
    next_required_feature: FeatureStage | None = None

    @property
    def features_seen(self) -> tuple[FeatureStage, ...]:
        return tuple(feature.stage for feature in self.features)

    @property
    def features_missing(self) -> tuple[FeatureStage, ...]:
        seen = set(self.features_seen)
        return tuple(stage for stage in REQUIRED_STAGES if stage not in seen)

    def add_feature(self, feature: FeatureResult) -> None:
        self.features.append(feature)

    def has_feature(self, stage: FeatureStage) -> bool:
        return any(feature.stage == stage for feature in self.features)

    def feature(self, stage: FeatureStage) -> FeatureResult | None:
        for feature in reversed(self.features):
            if feature.stage == stage:
                return feature
        return None
