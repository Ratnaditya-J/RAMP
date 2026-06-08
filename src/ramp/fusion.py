from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass
class WeightedRiskFusion:
    """Accumulates partial RAMP feature evidence into one risk estimate."""

    version: str = "risk_fusion_v0.1"
    weights: dict[FeatureStage, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    low_threshold: float = 0.20
    high_threshold: float = 0.78
    critical_threshold: float = 0.90
    high_confidence_threshold: float = 0.75

    def update(self, state: RiskState) -> RiskDecision:
        if not state.features:
            decision = self._decision_for_empty_state(state)
            self._apply(state, decision)
            return decision

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
            reason="no evidence available yet",
        )

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
