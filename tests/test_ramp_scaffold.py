from __future__ import annotations

import unittest

from ramp.features import FeatureInput
from ramp.pipeline import default_pipeline
from ramp.risk_state import RiskState
from ramp.schemas.feature_result import FeatureStage
from ramp.schemas.provenance import RuntimeProvenance
from ramp.schemas.risk_decision import RecommendedAction, RiskLevel


def make_state(
    request_id: str = "req_test",
    session_id: str = "sess_test",
    *,
    action_stakes: str = "low",
) -> RiskState:
    provenance = RuntimeProvenance(request_id=request_id, session_id=session_id)
    return RiskState(
        request_id=request_id,
        session_id=session_id,
        provenance=provenance,
        action_stakes=action_stakes,
    )


class RampScaffoldTest(unittest.TestCase):
    def test_low_risk_prompt_allows_fast_path(self) -> None:
        pipeline = default_pipeline()
        state = make_state()

        decision = pipeline.evaluate(
            FeatureInput(prompt="Can you explain safe password manager best practices?"),
            state,
        )

        self.assertEqual(decision.recommended_action, RecommendedAction.ALLOW_FAST_PATH)
        self.assertEqual(decision.risk_level, RiskLevel.LOW)
        self.assertIn(FeatureStage.PROMPT_RISK, decision.features_seen)
        self.assertIn(FeatureStage.OUTPUT_RISK, decision.features_missing)
        self.assertIsNone(decision.next_required_feature)

    def test_high_risk_prompt_blocks_early(self) -> None:
        pipeline = default_pipeline()
        state = make_state(action_stakes="high")

        decision = pipeline.evaluate(
            FeatureInput(prompt="bypass credential exploit weapon evade steal"),
            state,
        )

        self.assertEqual(decision.recommended_action, RecommendedAction.BLOCK)
        self.assertEqual(decision.risk_level, RiskLevel.CRITICAL)
        self.assertIn(FeatureStage.OUTPUT_RISK, decision.features_missing)

    def test_prompt_embedding_disagreement_is_recorded(self) -> None:
        pipeline = default_pipeline()
        state = make_state()

        decision = pipeline.evaluate(
            FeatureInput(prompt="Please explain this step by step and avoid detection."),
            state,
        )

        self.assertIn(
            "prompt classifier low risk but embedding prior elevated",
            decision.disagreements,
        )
        self.assertIn(
            decision.recommended_action,
            {RecommendedAction.CONTINUE_EVALUATION, RecommendedAction.ESCALATE},
        )

    def test_side_effecting_tool_requires_human_approval(self) -> None:
        pipeline = default_pipeline()
        state = make_state(action_stakes="high")
        feature_input = FeatureInput(
            prompt="Publish the current branch.",
            output="I will push the branch to origin.",
            tool_name="git_push",
        )

        pipeline.evaluate(feature_input, state)
        decision = pipeline.add_feature(feature_input, state, FeatureStage.TOOL_ACTION_RISK)

        self.assertEqual(decision.recommended_action, RecommendedAction.REQUIRE_HUMAN_APPROVAL)
        tool_feature = state.feature(FeatureStage.TOOL_ACTION_RISK)
        self.assertIsNotNone(tool_feature)
        self.assertIn("required_approval", tool_feature.metadata)


if __name__ == "__main__":
    unittest.main()
