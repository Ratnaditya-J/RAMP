from __future__ import annotations

import json
from pathlib import Path

import pytest

from ramp.fusion import CalibratedFusionConfig, WeightedRiskFusion
from ramp.pipeline import default_pipeline
from ramp.risk_state import RiskState
from ramp.schemas.feature_result import FeatureResult, FeatureStage
from ramp.schemas.provenance import RuntimeProvenance
from ramp.schemas.risk_decision import RecommendedAction


def make_state() -> RiskState:
    return RiskState(
        request_id="req_calibrated_fusion",
        session_id="sess_calibrated_fusion",
        provenance=RuntimeProvenance(
            request_id="req_calibrated_fusion",
            session_id="sess_calibrated_fusion",
        ),
    )


def write_calibration_artifact(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact_id": "calibrated_fusion_test",
                "selected_by_best_f1": {
                    "prompt_weight": 0.5,
                    "embedding_weight": 0.1,
                    "activation_weight": 0.4,
                    "threshold": 0.47,
                    "metrics": {"f1": 0.9},
                },
            }
        ),
        encoding="utf-8",
    )


def write_policy_artifact(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact_id": "ramp_fusion_policy_test",
                "decision": "prompt_activation_runtime_policy",
                "selected_runtime_score": {
                    "method": "weighted_sum",
                    "threshold": 0.53,
                    "weights": {
                        "prompt_risk_score": 0.25,
                        "activation_probability": 0.75,
                        "embedding_prior_score": 0.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_calibrated_fusion_uses_artifact_weights_and_metadata(tmp_path: Path) -> None:
    artifact = tmp_path / "calibration.json"
    write_calibration_artifact(artifact)
    config = CalibratedFusionConfig.from_artifact(artifact)
    fusion = WeightedRiskFusion(calibrated_config=config)
    state = make_state()
    state.add_feature(
        FeatureResult(
            stage=FeatureStage.PROMPT_RISK,
            risk_score=0.92,
            confidence=0.90,
            label="Unsafe",
        )
    )
    state.add_feature(
        FeatureResult(
            stage=FeatureStage.EMBEDDING_RISK,
            risk_score=0.10,
            confidence=0.80,
            label="semantic_proximity",
        )
    )
    state.add_feature(
        FeatureResult(
            stage=FeatureStage.ACTIVATION_PROBE,
            risk_score=0.10,
            confidence=0.80,
            label="activation_probe_safe",
        )
    )

    decision = fusion.update(state)

    assert decision.current_risk == pytest.approx(0.51)
    assert decision.recommended_action == RecommendedAction.ESCALATE
    assert decision.feature_contributions[FeatureStage.PROMPT_RISK] == pytest.approx(0.46)
    assert decision.feature_contributions[FeatureStage.EMBEDDING_RISK] == pytest.approx(0.01)
    assert decision.feature_contributions[FeatureStage.ACTIVATION_PROBE] == pytest.approx(0.04)
    assert decision.fusion_metadata["fusion_mode"] == "calibrated_reviewed_fusion"
    assert decision.fusion_metadata["calibration_artifact_id"] == "calibrated_fusion_test"
    assert decision.fusion_metadata["threshold"] == 0.47
    assert decision.fusion_metadata["calibration_objective"] == "selected_by_best_f1"


def test_calibrated_fusion_waits_for_required_internal_signals(tmp_path: Path) -> None:
    artifact = tmp_path / "calibration.json"
    write_calibration_artifact(artifact)
    fusion = WeightedRiskFusion(
        calibrated_config=CalibratedFusionConfig.from_artifact(artifact)
    )
    state = make_state()
    state.add_feature(
        FeatureResult(
            stage=FeatureStage.PROMPT_RISK,
            risk_score=0.92,
            confidence=0.90,
            label="Unsafe",
        )
    )

    decision = fusion.update(state)

    assert decision.fusion_metadata["fusion_mode"] == "weighted_average"
    assert decision.current_risk == 0.92


def test_calibrated_fusion_applies_high_severity_prompt_floor(tmp_path: Path) -> None:
    artifact = tmp_path / "calibration.json"
    write_calibration_artifact(artifact)
    fusion = WeightedRiskFusion(
        calibrated_config=CalibratedFusionConfig.from_artifact(artifact)
    )
    state = make_state()
    state.add_feature(
        FeatureResult(
            stage=FeatureStage.PROMPT_RISK,
            risk_score=0.92,
            confidence=0.90,
            label="Unsafe",
            harm_category="PII",
            metadata={"categories": ["PII"]},
        )
    )
    state.add_feature(
        FeatureResult(
            stage=FeatureStage.EMBEDDING_RISK,
            risk_score=0.05,
            confidence=0.80,
            label="semantic_low",
        )
    )
    state.add_feature(
        FeatureResult(
            stage=FeatureStage.ACTIVATION_PROBE,
            risk_score=0.05,
            confidence=0.80,
            label="activation_probe_safe",
        )
    )

    decision = fusion.update(state)

    assert decision.current_risk == pytest.approx(0.72)
    assert decision.recommended_action == RecommendedAction.ESCALATE
    assert decision.fusion_metadata["raw_calibrated_score"] == pytest.approx(0.485)
    assert decision.fusion_metadata["severity_floor_applied"] is True
    assert decision.fusion_metadata["severity_floor_reason"] == (
        "prompt_unsafe_high_severity_category"
    )
    assert decision.fusion_metadata["severity_floor_categories"] == ["pii"]


def test_calibrated_fusion_does_not_floor_non_high_severity_prompt(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "calibration.json"
    write_calibration_artifact(artifact)
    fusion = WeightedRiskFusion(
        calibrated_config=CalibratedFusionConfig.from_artifact(artifact)
    )
    state = make_state()
    state.add_feature(
        FeatureResult(
            stage=FeatureStage.PROMPT_RISK,
            risk_score=0.92,
            confidence=0.90,
            label="Unsafe",
            harm_category="Copyright Violation",
            metadata={"categories": ["Copyright Violation"]},
        )
    )
    state.add_feature(
        FeatureResult(
            stage=FeatureStage.EMBEDDING_RISK,
            risk_score=0.05,
            confidence=0.80,
            label="semantic_low",
        )
    )
    state.add_feature(
        FeatureResult(
            stage=FeatureStage.ACTIVATION_PROBE,
            risk_score=0.05,
            confidence=0.80,
            label="activation_probe_safe",
        )
    )

    decision = fusion.update(state)

    assert decision.current_risk == pytest.approx(0.485)
    assert decision.recommended_action == RecommendedAction.ESCALATE
    assert decision.fusion_metadata["severity_floor_applied"] is False


def test_frozen_policy_uses_prompt_activation_and_ignores_embedding(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "policy.json"
    write_policy_artifact(artifact)
    fusion = WeightedRiskFusion(
        calibrated_config=CalibratedFusionConfig.from_artifact(artifact)
    )
    state = make_state()
    state.add_feature(
        FeatureResult(
            stage=FeatureStage.PROMPT_RISK,
            risk_score=0.20,
            confidence=0.80,
            label="Safe",
        )
    )
    state.add_feature(
        FeatureResult(
            stage=FeatureStage.ACTIVATION_PROBE,
            risk_score=0.80,
            confidence=0.90,
            label="activation_probe_unsafe",
        )
    )

    decision_without_embedding = fusion.update(state)

    assert decision_without_embedding.current_risk == pytest.approx(0.65)
    assert decision_without_embedding.recommended_action == RecommendedAction.ESCALATE
    assert decision_without_embedding.feature_contributions[
        FeatureStage.PROMPT_RISK
    ] == pytest.approx(0.05)
    assert decision_without_embedding.feature_contributions[
        FeatureStage.ACTIVATION_PROBE
    ] == pytest.approx(0.60)
    assert FeatureStage.EMBEDDING_RISK not in decision_without_embedding.feature_contributions
    assert decision_without_embedding.fusion_metadata["fusion_mode"] == (
        "prompt_activation_runtime_policy"
    )
    assert decision_without_embedding.fusion_metadata["required_stages"] == [
        FeatureStage.ACTIVATION_PROBE.value,
        FeatureStage.PROMPT_RISK.value,
    ]

    state.add_feature(
        FeatureResult(
            stage=FeatureStage.EMBEDDING_RISK,
            risk_score=1.00,
            confidence=1.00,
            label="semantic_high",
        )
    )
    decision_with_embedding = fusion.update(state)

    assert decision_with_embedding.current_risk == pytest.approx(
        decision_without_embedding.current_risk
    )
    assert decision_with_embedding.feature_contributions[
        FeatureStage.EMBEDDING_RISK
    ] == pytest.approx(0.0)
    assert decision_with_embedding.fusion_metadata["ignored_zero_weight_stages"] == [
        FeatureStage.EMBEDDING_RISK.value
    ]


def test_frozen_policy_waits_for_prompt_and_activation(tmp_path: Path) -> None:
    artifact = tmp_path / "policy.json"
    write_policy_artifact(artifact)
    fusion = WeightedRiskFusion(
        calibrated_config=CalibratedFusionConfig.from_artifact(artifact)
    )
    state = make_state()
    state.add_feature(
        FeatureResult(
            stage=FeatureStage.PROMPT_RISK,
            risk_score=0.90,
            confidence=0.90,
            label="Unsafe",
        )
    )

    decision = fusion.update(state)

    assert decision.fusion_metadata["fusion_mode"] == "weighted_average"
    assert decision.current_risk == pytest.approx(0.90)


def test_default_pipeline_loads_frozen_policy(tmp_path: Path) -> None:
    artifact = tmp_path / "policy.json"
    write_policy_artifact(artifact)

    pipeline = default_pipeline(fusion_policy_artifact=artifact)

    assert pipeline.fusion.calibrated_config is not None
    assert pipeline.fusion.calibrated_config.artifact_id == "ramp_fusion_policy_test"
    assert pipeline.fusion.calibrated_config.prompt_weight == pytest.approx(0.25)
    assert pipeline.fusion.calibrated_config.activation_weight == pytest.approx(0.75)
    assert pipeline.fusion.calibrated_config.embedding_weight == pytest.approx(0.0)
