from __future__ import annotations

from ramp.features.base import FeatureExtractor, FeatureInput
from ramp.risk_state import RiskState
from ramp.schemas.feature_result import FeatureResult, FeatureStage


class KeywordPromptRiskFeature(FeatureExtractor):
    """Deterministic placeholder for an off-the-shelf prompt classifier."""

    stage = FeatureStage.PROMPT_RISK
    risky_terms = ("bypass", "credential", "exploit", "weapon", "evade", "steal")

    def extract(self, feature_input: FeatureInput, state: RiskState) -> FeatureResult:
        text = feature_input.prompt.lower()
        hits = [term for term in self.risky_terms if term in text]
        risk = min(0.95, 0.08 + 0.22 * len(hits))
        confidence = 0.88 if hits else 0.82
        label = "elevated" if risk >= 0.50 else "low"
        return FeatureResult(
            stage=self.stage,
            risk_score=risk,
            confidence=confidence,
            label=label,
            harm_category="policy_relevant" if hits else None,
            cost_tier=1,
            version="keyword_prompt_risk_v0",
            metadata={"matched_terms": hits},
        )

