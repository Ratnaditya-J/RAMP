from __future__ import annotations

from ramp.features.base import FeatureExtractor, FeatureInput
from ramp.risk_state import RiskState
from ramp.schemas.feature_result import FeatureResult, FeatureStage


class StubActivationProbeFeature(FeatureExtractor):
    stage = FeatureStage.ACTIVATION_PROBE

    def extract(self, feature_input: FeatureInput, state: RiskState) -> FeatureResult:
        inherited_risk = state.current_risk
        probe_score = min(0.95, inherited_risk + 0.08)
        return FeatureResult(
            stage=self.stage,
            risk_score=probe_score,
            confidence=0.64,
            label="probe_placeholder",
            cost_tier=2,
            version="stub_activation_probe_v0",
            metadata={"selected_layer": "demo_layer", "probe_score": probe_score},
        )

