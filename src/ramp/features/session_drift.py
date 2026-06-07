from __future__ import annotations

from ramp.features.base import FeatureExtractor, FeatureInput
from ramp.risk_state import RiskState
from ramp.schemas.feature_result import FeatureResult, FeatureStage
from ramp.schemas.session_state import SessionState


class RollingSessionDriftFeature(FeatureExtractor):
    stage = FeatureStage.SESSION_DRIFT

    def extract(self, feature_input: FeatureInput, state: RiskState) -> FeatureResult:
        session = feature_input.context.get("session_state")
        if not isinstance(session, SessionState):
            rolling = 0.0
            turn_count = 0
        else:
            rolling = session.rolling_risk
            turn_count = len(session.turns)

        risk = min(0.95, 0.10 + rolling * 0.85)
        return FeatureResult(
            stage=self.stage,
            risk_score=risk,
            confidence=0.70 if turn_count else 0.55,
            label="increasing" if risk >= 0.60 else "stable",
            cost_tier=1,
            version="rolling_session_drift_v0",
            metadata={"turn_count": turn_count, "rolling_risk": rolling},
        )

