from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ramp.features.base import FeatureExtractor, FeatureInput
from ramp.fusion import CalibratedFusionConfig, WeightedRiskFusion
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


def prompt_risk_feature_from_backend(backend: str = "keyword") -> FeatureExtractor:
    from ramp.features import KeywordPromptRiskFeature, Qwen3GuardPromptRiskFeature

    normalized = backend.strip().lower()
    if normalized == "keyword":
        return KeywordPromptRiskFeature()
    if normalized in {"qwen", "qwen3guard"}:
        return Qwen3GuardPromptRiskFeature.from_env()
    raise ValueError(f"unknown prompt risk backend: {backend}")


def output_risk_feature_from_backend(backend: str = "keyword") -> FeatureExtractor:
    from ramp.features import KeywordOutputRiskFeature, Qwen3GuardOutputRiskFeature

    normalized = backend.strip().lower()
    if normalized == "keyword":
        return KeywordOutputRiskFeature()
    if normalized in {"qwen", "qwen3guard"}:
        return Qwen3GuardOutputRiskFeature.from_env()
    raise ValueError(f"unknown output risk backend: {backend}")


def default_pipeline(
    prompt_risk_backend: str = "keyword",
    *,
    output_risk_backend: str = "keyword",
    fusion_calibration_artifact: str | Path | None = None,
    fusion_policy_artifact: str | Path | None = None,
    fusion_calibration_objective: str = "selected_by_best_f1",
) -> RampPipeline:
    from ramp.features import (
        CompactSessionRiskFeature,
        LexicalEmbeddingRiskFeature,
        SideEffectToolActionRiskFeature,
        StubActivationProbeFeature,
    )
    if fusion_calibration_artifact is not None and fusion_policy_artifact is not None:
        raise ValueError(
            "provide only one of fusion_calibration_artifact or fusion_policy_artifact"
        )

    feature_list = [
        prompt_risk_feature_from_backend(prompt_risk_backend),
        LexicalEmbeddingRiskFeature(),
        CompactSessionRiskFeature(),
        StubActivationProbeFeature(),
        output_risk_feature_from_backend(output_risk_backend),
        SideEffectToolActionRiskFeature(),
    ]
    fusion_artifact = fusion_policy_artifact or fusion_calibration_artifact
    calibrated_config = (
        CalibratedFusionConfig.from_artifact(
            fusion_artifact,
            objective=fusion_calibration_objective,
        )
        if fusion_artifact is not None
        else None
    )
    return RampPipeline(
        features={feature.stage: feature for feature in feature_list},
        fusion=WeightedRiskFusion(calibrated_config=calibrated_config),
        scheduler=AnytimeScheduler(),
    )
