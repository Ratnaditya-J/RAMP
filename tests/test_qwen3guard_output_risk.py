from __future__ import annotations

import unittest

from ramp.features import FeatureInput, Qwen3GuardOutputRiskFeature
from ramp.risk_state import RiskState
from ramp.schemas.feature_result import FeatureStage
from ramp.schemas.provenance import RuntimeProvenance


def make_state() -> RiskState:
    request_id = "req_qwen_output_test"
    session_id = "sess_qwen_output_test"
    return RiskState(
        request_id=request_id,
        session_id=session_id,
        provenance=RuntimeProvenance(request_id=request_id, session_id=session_id),
    )


class Qwen3GuardOutputRiskTest(unittest.TestCase):
    def test_output_feature_scores_generated_text_without_model_load(self) -> None:
        feature = Qwen3GuardOutputRiskFeature(
            completion_fn=lambda _: "Safety: Unsafe\nCategories: Non-violent Illegal Acts"
        )

        result = feature.extract(
            FeatureInput(
                prompt="How would I do this?",
                output="Here are the concrete steps.",
            ),
            make_state(),
        )

        self.assertEqual(result.stage, FeatureStage.OUTPUT_RISK)
        self.assertEqual(result.label, "Unsafe")
        self.assertEqual(result.harm_category, "Non-violent Illegal Acts")
        self.assertGreater(result.risk_score, 0.90)
        self.assertEqual(result.cost_tier, 3)
        self.assertEqual(result.metadata["backend"], "qwen3guard")
        self.assertEqual(result.metadata["scored_text"], "output")


if __name__ == "__main__":
    unittest.main()
