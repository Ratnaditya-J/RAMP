from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ramp.risk_state import RiskState
from ramp.schemas.feature_result import FeatureResult, FeatureStage
from ramp.schemas.risk_decision import RecommendedAction, RiskDecision, RiskLevel

DEFAULT_WEIGHTS: dict[FeatureStage, float] = {
    FeatureStage.POLICY_HEURISTICS: 0.8,
    FeatureStage.PROMPT_RISK: 1.1,
    FeatureStage.EMBEDDING_RISK: 0.8,
    FeatureStage.ACTIVATION_PROBE: 1.3,
    FeatureStage.OUTPUT_RISK: 1.2,
    FeatureStage.SESSION_DRIFT: 1.0,
    FeatureStage.TOOL_ACTION_RISK: 1.4,
}

HIGH_SEVERITY_PROMPT_CATEGORIES: frozenset[str] = frozenset(
    {
        "pii",
        "personally identifiable information",
        "suicide & self-harm",
        "violent",
        "non-violent illegal acts",
        "jailbreak",
        "unethical acts",
    }
)


@dataclass(frozen=True)
class CalibratedFusionConfig:
    """Versioned prompt/embedding/activation fusion parameters."""

    artifact_id: str
    objective: str
    prompt_weight: float
    embedding_weight: float
    activation_weight: float
    threshold: float
    source_path: str | None = None
    version: str = "reviewed_fusion_calibration_v0.1"
    action: RecommendedAction = RecommendedAction.ESCALATE
    high_severity_floor: float = 0.72
    high_severity_categories: frozenset[str] = HIGH_SEVERITY_PROMPT_CATEGORIES
    runtime_decision: str = "calibrated_reviewed_fusion"

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        *,
        objective: str = "selected_by_best_f1",
        action: RecommendedAction = RecommendedAction.ESCALATE,
    ) -> CalibratedFusionConfig:
        artifact_path = Path(path)
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if isinstance(artifact.get("selected_runtime_score"), dict):
            return cls.from_policy_artifact(artifact_path, action=action)
        selected = artifact.get(objective)
        if not isinstance(selected, dict):
            raise ValueError(f"calibration artifact missing objective: {objective}")
        return cls(
            artifact_id=str(artifact.get("artifact_id", artifact_path.name)),
            objective=objective,
            prompt_weight=float(selected["prompt_weight"]),
            embedding_weight=float(selected["embedding_weight"]),
            activation_weight=float(selected["activation_weight"]),
            threshold=float(selected["threshold"]),
            source_path=str(artifact_path),
            action=action,
        )

    @classmethod
    def from_policy_artifact(
        cls,
        path: str | Path,
        *,
        action: RecommendedAction = RecommendedAction.ESCALATE,
    ) -> CalibratedFusionConfig:
        artifact_path = Path(path)
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        selected = artifact.get("selected_runtime_score")
        if not isinstance(selected, dict):
            raise ValueError("fusion policy artifact missing selected_runtime_score")
        weights = selected.get("weights")
        if not isinstance(weights, dict):
            raise ValueError("fusion policy artifact missing selected_runtime_score.weights")
        return cls(
            artifact_id=str(artifact.get("artifact_id", artifact_path.name)),
            objective=str(artifact.get("decision", "selected_runtime_score")),
            prompt_weight=float(weights.get("prompt_risk_score", 0.0)),
            embedding_weight=float(weights.get("embedding_prior_score", 0.0)),
            activation_weight=float(weights.get("activation_probability", 0.0)),
            threshold=float(selected["threshold"]),
            source_path=str(artifact_path),
            version=str(artifact.get("artifact_id", "ramp_fusion_policy_v0.1")),
            action=action,
            runtime_decision=str(artifact.get("decision", "prompt_activation_runtime_policy")),
        )

    def required_stages(self) -> frozenset[FeatureStage]:
        required = set()
        if self.prompt_weight > 0:
            required.add(FeatureStage.PROMPT_RISK)
        if self.embedding_weight > 0:
            required.add(FeatureStage.EMBEDDING_RISK)
        if self.activation_weight > 0:
            required.add(FeatureStage.ACTIVATION_PROBE)
        return frozenset(required)

    def metadata(self) -> dict[str, Any]:
        return {
            "fusion_mode": self.runtime_decision,
            "calibration_artifact_id": self.artifact_id,
            "calibration_artifact_path": self.source_path,
            "calibration_objective": self.objective,
            "calibration_version": self.version,
            "prompt_weight": self.prompt_weight,
            "embedding_weight": self.embedding_weight,
            "activation_weight": self.activation_weight,
            "threshold": self.threshold,
            "action": self.action.value,
            "high_severity_floor": self.high_severity_floor,
            "high_severity_categories": sorted(self.high_severity_categories),
        }


@dataclass
class WeightedRiskFusion:
    """Accumulates partial RAMP feature evidence into one risk estimate."""

    version: str = "risk_fusion_v0.1"
    weights: dict[FeatureStage, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    calibrated_config: CalibratedFusionConfig | None = None
    low_threshold: float = 0.20
    high_threshold: float = 0.78
    critical_threshold: float = 0.90
    high_confidence_threshold: float = 0.75

    def update(self, state: RiskState) -> RiskDecision:
        if not state.features:
            decision = self._decision_for_empty_state(state)
            self._apply(state, decision)
            return decision

        calibrated_decision = self._calibrated_decision(state)
        if calibrated_decision is not None:
            self._apply(state, calibrated_decision)
            return calibrated_decision

        weighted_sum = 0.0
        total_weight = 0.0
        contributions: dict[FeatureStage, float] = {}
        for feature in state.features:
            weight = self.weights.get(feature.stage, 1.0) * max(feature.confidence, 0.05)
            contribution = feature.risk_score * weight
            weighted_sum += contribution
            total_weight += weight
            contributions[feature.stage] = contribution

        risk = weighted_sum / total_weight if total_weight else 0.0
        confidence = self._combined_confidence(state.features)
        disagreements = self._find_disagreements(state.features)
        action = self._recommend_action(state, risk, confidence)
        risk_level = self._risk_level(risk)
        next_required_feature = (
            None if self._is_terminal_action(action) else state.next_required_feature
        )

        reason = self._reason(action, risk, confidence, disagreements)
        decision = RiskDecision(
            request_id=state.request_id,
            session_id=state.session_id,
            current_risk=risk,
            confidence=confidence,
            risk_level=risk_level,
            recommended_action=action,
            features_seen=state.features_seen,
            features_missing=state.features_missing,
            disagreements=tuple(disagreements),
            next_required_feature=next_required_feature,
            feature_contributions=contributions,
            fusion_metadata={"fusion_mode": "weighted_average", "fusion_version": self.version},
            reason=reason,
        )
        self._apply(state, decision)
        return decision

    def _decision_for_empty_state(self, state: RiskState) -> RiskDecision:
        return RiskDecision(
            request_id=state.request_id,
            session_id=state.session_id,
            current_risk=0.0,
            confidence=0.0,
            risk_level=RiskLevel.LOW,
            recommended_action=RecommendedAction.CONTINUE_EVALUATION,
            features_seen=(),
            features_missing=state.features_missing,
            next_required_feature=FeatureStage.PROMPT_RISK,
            fusion_metadata={"fusion_mode": "weighted_average", "fusion_version": self.version},
            reason="no evidence available yet",
        )

    def _calibrated_decision(self, state: RiskState) -> RiskDecision | None:
        if self.calibrated_config is None:
            return None
        by_stage = {feature.stage: feature for feature in state.features}
        prompt = by_stage.get(FeatureStage.PROMPT_RISK)
        embedding = by_stage.get(FeatureStage.EMBEDDING_RISK)
        activation = by_stage.get(FeatureStage.ACTIVATION_PROBE)
        required_stages = self.calibrated_config.required_stages()
        if any(stage not in by_stage for stage in required_stages):
            return None

        config = self.calibrated_config
        contributions: dict[FeatureStage, float] = {}
        if prompt is not None:
            contributions[FeatureStage.PROMPT_RISK] = prompt.risk_score * config.prompt_weight
        if embedding is not None:
            contributions[FeatureStage.EMBEDDING_RISK] = (
                embedding.risk_score * config.embedding_weight
            )
        if activation is not None:
            contributions[FeatureStage.ACTIVATION_PROBE] = (
                activation.risk_score * config.activation_weight
            )
        risk = sum(contributions.values())
        raw_calibrated_risk = risk
        confidence_numerator = sum(
            by_stage[stage].confidence * weight
            for stage, weight in (
                (FeatureStage.PROMPT_RISK, config.prompt_weight),
                (FeatureStage.EMBEDDING_RISK, config.embedding_weight),
                (FeatureStage.ACTIVATION_PROBE, config.activation_weight),
            )
            if stage in by_stage and weight > 0
        )
        confidence_denominator = sum(
            weight
            for stage, weight in (
                (FeatureStage.PROMPT_RISK, config.prompt_weight),
                (FeatureStage.EMBEDDING_RISK, config.embedding_weight),
                (FeatureStage.ACTIVATION_PROBE, config.activation_weight),
            )
            if stage in by_stage and weight > 0
        )
        confidence = (
            confidence_numerator / confidence_denominator
            if confidence_denominator > 0
            else 0.0
        )
        severity_floor = self._severity_floor(prompt, config) if prompt is not None else None
        severity_floor_applied = (
            severity_floor is not None and risk < severity_floor["floor"]
        )
        if severity_floor_applied:
            risk = float(severity_floor["floor"])
        disagreements = self._find_disagreements(state.features)
        risk_level = self._risk_level(risk)
        action = (
            config.action
            if risk >= config.threshold and confidence >= 0.50
            else RecommendedAction.CONTINUE_EVALUATION
        )
        if self._has_high_tool_risk(state):
            action = RecommendedAction.REQUIRE_HUMAN_APPROVAL
        elif risk >= self.critical_threshold and confidence >= 0.65:
            action = RecommendedAction.BLOCK
        next_required_feature = (
            None if self._is_terminal_action(action) else state.next_required_feature
        )
        metadata = config.metadata()
        metadata["calibrated_score"] = risk
        metadata["raw_calibrated_score"] = raw_calibrated_risk
        metadata["calibrated_confidence"] = confidence
        metadata["threshold_met"] = risk >= config.threshold
        metadata["required_stages"] = [stage.value for stage in sorted(required_stages)]
        metadata["available_calibrated_stages"] = [
            stage.value for stage in sorted(set(contributions))
        ]
        metadata["ignored_zero_weight_stages"] = [
            stage.value
            for stage, weight in (
                (FeatureStage.PROMPT_RISK, config.prompt_weight),
                (FeatureStage.EMBEDDING_RISK, config.embedding_weight),
                (FeatureStage.ACTIVATION_PROBE, config.activation_weight),
            )
            if weight == 0 and stage in by_stage
        ]
        metadata["severity_floor_applied"] = severity_floor_applied
        if severity_floor is not None:
            metadata["severity_floor_candidate"] = severity_floor["floor"]
            metadata["severity_floor_categories"] = severity_floor[
                "severity_floor_categories"
            ]
        if severity_floor_applied and severity_floor is not None:
            metadata["floor"] = severity_floor["floor"]
            metadata["severity_floor_reason"] = severity_floor[
                "severity_floor_reason"
            ]
        metadata["fusion_version"] = self.version
        reason = (
            f"{action.value}: calibrated risk={risk:.2f}, threshold={config.threshold:.2f}, "
            f"confidence={confidence:.2f}"
        )
        if severity_floor_applied:
            reason = f"{reason}; severity floor applied"
        if disagreements:
            reason = f"{reason}; feature disagreement requires attention"
        return RiskDecision(
            request_id=state.request_id,
            session_id=state.session_id,
            current_risk=risk,
            confidence=confidence,
            risk_level=risk_level,
            recommended_action=action,
            features_seen=state.features_seen,
            features_missing=state.features_missing,
            disagreements=tuple(disagreements),
            next_required_feature=next_required_feature,
            feature_contributions=contributions,
            fusion_metadata=metadata,
            reason=reason,
        )

    def _severity_floor(
        self,
        prompt: FeatureResult,
        config: CalibratedFusionConfig,
    ) -> dict[str, Any] | None:
        if str(prompt.label or "").strip().lower() != "unsafe":
            return None
        categories = self._prompt_categories(prompt)
        matched = sorted(categories & config.high_severity_categories)
        if not matched:
            return None
        return {
            "floor": config.high_severity_floor,
            "severity_floor_reason": "prompt_unsafe_high_severity_category",
            "severity_floor_categories": matched,
        }

    def _prompt_categories(self, prompt: FeatureResult) -> set[str]:
        categories = set()
        if prompt.harm_category:
            categories.add(str(prompt.harm_category).strip().lower())
        raw_categories = prompt.metadata.get("categories", [])
        if isinstance(raw_categories, str):
            categories.add(raw_categories.strip().lower())
        elif isinstance(raw_categories, list | tuple | set):
            categories.update(str(category).strip().lower() for category in raw_categories)
        return {category for category in categories if category and category != "none"}

    def _combined_confidence(self, features: list[FeatureResult]) -> float:
        total_weight = sum(self.weights.get(feature.stage, 1.0) for feature in features)
        if total_weight == 0:
            return 0.0
        return sum(
            feature.confidence * self.weights.get(feature.stage, 1.0) for feature in features
        ) / total_weight

    def _find_disagreements(self, features: list[FeatureResult]) -> list[str]:
        disagreements: list[str] = []
        by_stage = {feature.stage: feature for feature in features}

        prompt = by_stage.get(FeatureStage.PROMPT_RISK)
        embedding = by_stage.get(FeatureStage.EMBEDDING_RISK)
        activation = by_stage.get(FeatureStage.ACTIVATION_PROBE)
        output = by_stage.get(FeatureStage.OUTPUT_RISK)
        session = by_stage.get(FeatureStage.SESSION_DRIFT)
        tool = by_stage.get(FeatureStage.TOOL_ACTION_RISK)

        if prompt and embedding and prompt.risk_score < 0.30 and embedding.risk_score > 0.65:
            disagreements.append("prompt classifier low risk but embedding prior elevated")
        if activation and output and activation.risk_score > 0.65 and output.risk_score < 0.30:
            disagreements.append("activation probe high risk but output classifier safe")
        if output and tool and output.risk_score < 0.30 and tool.risk_score > 0.65:
            disagreements.append("output safe but proposed tool action high risk")
        if prompt and session and prompt.risk_score < 0.35 and session.risk_score > 0.65:
            disagreements.append("single-turn risk low but session drift increasing")

        return disagreements

    def _recommend_action(
        self, state: RiskState, risk: float, confidence: float
    ) -> RecommendedAction:
        if self._has_high_tool_risk(state):
            return RecommendedAction.REQUIRE_HUMAN_APPROVAL
        if risk >= self.critical_threshold and confidence >= 0.65:
            return RecommendedAction.BLOCK
        if risk >= self.high_threshold and confidence >= self.high_confidence_threshold:
            return RecommendedAction.ESCALATE
        if (
            risk <= self.low_threshold
            and confidence >= self.high_confidence_threshold
            and state.action_stakes == "low"
            and self._has_minimum_fast_path_evidence(state)
        ):
            return RecommendedAction.ALLOW_FAST_PATH
        return RecommendedAction.CONTINUE_EVALUATION

    def _has_high_tool_risk(self, state: RiskState) -> bool:
        tool = state.feature(FeatureStage.TOOL_ACTION_RISK)
        return tool is not None and tool.risk_score >= 0.65

    def _has_minimum_fast_path_evidence(self, state: RiskState) -> bool:
        early_stages = {
            FeatureStage.POLICY_HEURISTICS,
            FeatureStage.PROMPT_RISK,
            FeatureStage.EMBEDDING_RISK,
            FeatureStage.SESSION_DRIFT,
        }
        seen = {feature.stage for feature in state.features}
        return FeatureStage.PROMPT_RISK in seen and len(seen & early_stages) >= 2

    def _is_terminal_action(self, action: RecommendedAction) -> bool:
        return action in {
            RecommendedAction.ALLOW_FAST_PATH,
            RecommendedAction.BLOCK,
            RecommendedAction.ESCALATE,
            RecommendedAction.REQUIRE_HUMAN_APPROVAL,
        }

    def _risk_level(self, risk: float) -> RiskLevel:
        if risk >= self.critical_threshold:
            return RiskLevel.CRITICAL
        if risk >= self.high_threshold:
            return RiskLevel.HIGH
        if risk >= self.low_threshold:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _reason(
        self,
        action: RecommendedAction,
        risk: float,
        confidence: float,
        disagreements: list[str],
    ) -> str:
        if disagreements:
            return f"{action.value}: feature disagreement requires attention"
        return f"{action.value}: risk={risk:.2f}, confidence={confidence:.2f}"

    def _apply(self, state: RiskState, decision: RiskDecision) -> None:
        state.current_risk = decision.current_risk
        state.confidence = decision.confidence
        state.recommended_action = decision.recommended_action
        state.disagreements = list(decision.disagreements)
