from __future__ import annotations

from dataclasses import dataclass

from ramp.risk_state import RiskState
from ramp.schemas.feature_result import FeatureStage
from ramp.schemas.risk_decision import RecommendedAction

SCHEDULE_ORDER: tuple[FeatureStage, ...] = (
    FeatureStage.PROMPT_RISK,
    FeatureStage.EMBEDDING_RISK,
    FeatureStage.SESSION_DRIFT,
    FeatureStage.ACTIVATION_PROBE,
    FeatureStage.OUTPUT_RISK,
)


@dataclass(frozen=True)
class AnytimeScheduler:
    """Selects early exit or the next useful feature from partial evidence."""

    low_risk_cutoff: float = 0.20
    high_risk_cutoff: float = 0.78
    high_confidence_cutoff: float = 0.75

    def choose_next(
        self,
        state: RiskState,
        available_stages: set[FeatureStage] | None = None,
    ) -> FeatureStage | None:
        if self.should_exit_early(state):
            state.next_required_feature = None
            return None

        available = available_stages or set(SCHEDULE_ORDER)

        for stage in SCHEDULE_ORDER:
            if stage not in available:
                continue
            if not state.has_feature(stage):
                state.next_required_feature = stage
                return stage

        state.next_required_feature = None
        return None

    def should_exit_early(self, state: RiskState) -> bool:
        if state.recommended_action in {
            RecommendedAction.ALLOW_FAST_PATH,
            RecommendedAction.BLOCK,
            RecommendedAction.ESCALATE,
            RecommendedAction.REQUIRE_HUMAN_APPROVAL,
        }:
            return True
        if (
            state.current_risk <= self.low_risk_cutoff
            and state.confidence >= self.high_confidence_cutoff
            and state.action_stakes == "low"
            and self._has_minimum_fast_path_evidence(state)
        ):
            return True
        return state.current_risk >= self.high_risk_cutoff and state.confidence >= 0.65

    def _has_minimum_fast_path_evidence(self, state: RiskState) -> bool:
        early_stages = {
            FeatureStage.POLICY_HEURISTICS,
            FeatureStage.PROMPT_RISK,
            FeatureStage.EMBEDDING_RISK,
            FeatureStage.SESSION_DRIFT,
        }
        seen = {feature.stage for feature in state.features}
        return FeatureStage.PROMPT_RISK in seen and len(seen & early_stages) >= 2
