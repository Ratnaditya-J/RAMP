from __future__ import annotations

from dataclasses import dataclass

from ramp.features.base import FeatureExtractor, FeatureInput
from ramp.fusion import WeightedRiskFusion
from ramp.risk_state import RiskState
from ramp.scheduler import AnytimeScheduler
from ramp.schemas.feature_result import FeatureStage
from ramp.schemas.risk_decision import RiskDecision


@dataclass
class RampPipeline:
    """Coordinates scheduled feature extraction and risk fusion."""

    features: dict[FeatureStage, FeatureExtractor]
    fusion: WeightedRiskFusion
    scheduler: AnytimeScheduler

    def evaluate(self, feature_input: FeatureInput, state: RiskState) -> RiskDecision:
        decision = self.fusion.update(state)
        next_stage = self.scheduler.choose_next(state, self.available_stages(feature_input))

        while next_stage is not None:
            extractor = self.features.get(next_stage)
            if extractor is None:
                break
            state.add_feature(extractor.extract(feature_input, state))
            decision = self.fusion.update(state)
            next_stage = self.scheduler.choose_next(state, self.available_stages(feature_input))

        return decision

    def add_feature(
        self,
        feature_input: FeatureInput,
        state: RiskState,
        stage: FeatureStage,
    ) -> RiskDecision:
        extractor = self.features[stage]
        state.add_feature(extractor.extract(feature_input, state))
        return self.fusion.update(state)

    def available_stages(self, feature_input: FeatureInput) -> set[FeatureStage]:
        stages = {
            FeatureStage.PROMPT_RISK,
            FeatureStage.EMBEDDING_RISK,
            FeatureStage.SESSION_DRIFT,
            FeatureStage.ACTIVATION_PROBE,
        }
        if feature_input.output is not None:
            stages.add(FeatureStage.OUTPUT_RISK)
        if feature_input.tool_name is not None:
            stages.add(FeatureStage.TOOL_ACTION_RISK)
        return stages


def default_pipeline() -> RampPipeline:
    from ramp.features import (
        KeywordOutputRiskFeature,
        KeywordPromptRiskFeature,
        LexicalEmbeddingRiskFeature,
        RollingSessionDriftFeature,
        SideEffectToolActionRiskFeature,
        StubActivationProbeFeature,
    )

    feature_list = [
        KeywordPromptRiskFeature(),
        LexicalEmbeddingRiskFeature(),
        RollingSessionDriftFeature(),
        StubActivationProbeFeature(),
        KeywordOutputRiskFeature(),
        SideEffectToolActionRiskFeature(),
    ]
    return RampPipeline(
        features={feature.stage: feature for feature in feature_list},
        fusion=WeightedRiskFusion(),
        scheduler=AnytimeScheduler(),
    )
