from __future__ import annotations

from ramp.features.base import FeatureExtractor, FeatureInput
from ramp.risk_state import RiskState
from ramp.schemas.feature_result import FeatureResult, FeatureStage


class KeywordOutputRiskFeature(FeatureExtractor):
    stage = FeatureStage.OUTPUT_RISK
    unsafe_terms = ("here are the steps", "run this command", "payload", "exfiltrate")
    refusal_terms = ("i can't help", "i cannot help", "safe alternative")

    def extract(self, feature_input: FeatureInput, state: RiskState) -> FeatureResult:
        text = (feature_input.output or "").lower()
        unsafe_hits = [term for term in self.unsafe_terms if term in text]
        refusal_hits = [term for term in self.refusal_terms if term in text]
        risk = min(0.95, 0.12 + 0.25 * len(unsafe_hits) - 0.08 * len(refusal_hits))
        risk = max(0.02, risk)
        return FeatureResult(
            stage=self.stage,
            risk_score=risk,
            confidence=0.78,
            label="unsafe_actionable_content" if unsafe_hits else "safe_or_refusal",
            cost_tier=3,
            version="keyword_output_risk_v0",
            metadata={
                "unsafe_hits": unsafe_hits,
                "refusal_hits": refusal_hits,
                "actionability_score": min(1.0, 0.2 * len(unsafe_hits)),
            },
        )

