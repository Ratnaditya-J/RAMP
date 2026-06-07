from __future__ import annotations

from ramp.features.base import FeatureExtractor, FeatureInput
from ramp.risk_state import RiskState
from ramp.schemas.feature_result import FeatureResult, FeatureStage


class LexicalEmbeddingRiskFeature(FeatureExtractor):
    """Placeholder for span-to-cluster embedding proximity scoring."""

    stage = FeatureStage.EMBEDDING_RISK
    risky_spans = (
        "avoid detection",
        "bypass account",
        "access credentials",
        "make it stronger",
        "step by step",
    )
    benign_spans = ("defensive", "safety", "incident response", "patch", "legal")

    def extract(self, feature_input: FeatureInput, state: RiskState) -> FeatureResult:
        text = feature_input.prompt.lower()
        risky_hits = [span for span in self.risky_spans if span in text]
        benign_hits = [span for span in self.benign_spans if span in text]
        harm_similarity = min(1.0, 0.15 + 0.30 * len(risky_hits))
        benign_similarity = min(1.0, 0.15 + 0.25 * len(benign_hits))
        margin = harm_similarity - benign_similarity
        risk = min(0.95, max(0.04, 0.08 + 0.34 * len(risky_hits) - 0.16 * len(benign_hits)))
        return FeatureResult(
            stage=self.stage,
            risk_score=risk,
            confidence=0.72,
            label="semantic_proximity",
            cost_tier=1,
            version="lexical_embedding_risk_v0",
            metadata={
                "top_harm_cluster": "demo_harm_cluster" if risky_hits else None,
                "top_benign_cluster": "demo_benign_cluster" if benign_hits else None,
                "harm_similarity": harm_similarity,
                "benign_similarity": benign_similarity,
                "harm_minus_benign_margin": margin,
                "top_span": risky_hits[0] if risky_hits else None,
                "num_triggered_spans": len(risky_hits),
            },
        )
