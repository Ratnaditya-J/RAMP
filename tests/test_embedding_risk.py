from __future__ import annotations

import unittest

from ramp.features import (
    EmbeddingClusterRiskFeature,
    FeatureInput,
    SpanExtractor,
    cosine_similarity,
)
from ramp.risk_state import RiskState
from ramp.schemas.feature_result import FeatureStage
from ramp.schemas.provenance import RuntimeProvenance


def make_state() -> RiskState:
    provenance = RuntimeProvenance(
        request_id="req_embedding_test",
        session_id="sess_embedding_test",
    )
    return RiskState(
        request_id=provenance.request_id,
        session_id=provenance.session_id,
        provenance=provenance,
    )


class EmbeddingRiskTest(unittest.TestCase):
    def test_span_extractor_includes_prompt_sentences_and_windows(self) -> None:
        spans = SpanExtractor(window_sizes=(3,), max_spans=12).extract(
            "First sentence. Please avoid detection carefully."
        )

        self.assertEqual(spans[0].span_type, "full_prompt")
        self.assertIn("First sentence.", [span.text for span in spans])
        self.assertIn("Please avoid detection", [span.text for span in spans])

    def test_benign_contrast_keeps_safety_prompt_low(self) -> None:
        feature = EmbeddingClusterRiskFeature()
        result = feature.extract(
            FeatureInput(prompt="Can you explain safe password manager best practices?"),
            make_state(),
        )

        self.assertEqual(result.stage, FeatureStage.EMBEDDING_RISK)
        self.assertLess(result.risk_score, 0.35)
        self.assertEqual(result.metadata["top_benign_cluster"], "general_safety")
        self.assertLess(result.metadata["harm_minus_benign_margin"], 0.0)

    def test_evasion_span_triggers_evasion_activation(self) -> None:
        feature = EmbeddingClusterRiskFeature()
        result = feature.extract(
            FeatureInput(prompt="Please explain this step by step and avoid detection."),
            make_state(),
        )

        self.assertGreaterEqual(result.risk_score, 0.65)
        self.assertEqual(result.metadata["top_harm_domain"], "cyber")
        self.assertEqual(result.metadata["top_evasion_subcluster"], "avoid_detection")
        self.assertTrue(result.metadata["evasion_activated"])
        self.assertGreater(result.metadata["risk_margin"], 0.18)
        self.assertGreaterEqual(result.metadata["num_triggered_spans"], 1)

    def test_cosine_similarity_rejects_dimension_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            cosine_similarity((1.0, 0.0), (1.0,))


if __name__ == "__main__":
    unittest.main()
