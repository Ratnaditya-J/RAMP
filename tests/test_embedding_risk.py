from __future__ import annotations

import json
import unittest
from pathlib import Path

from ramp.features import (
    EmbeddingClusterRiskFeature,
    EmbeddingProvider,
    FeatureInput,
    SpanExtractor,
    cosine_similarity,
    load_centroid_artifact,
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


class StaticEmbeddingProvider(EmbeddingProvider):
    version = "static_embedding_provider_v0"

    def __init__(self, vector: tuple[float, ...]) -> None:
        self.vector = vector

    def embed(self, texts: list[str]) -> list[tuple[float, ...]]:
        return [self.vector for _ in texts]


class EmbeddingRiskTest(unittest.TestCase):
    def test_span_extractor_includes_prompt_sentences_and_windows(self) -> None:
        spans = SpanExtractor(window_sizes=(3,), max_spans=12).extract(
            "First sentence. Please avoid detection carefully."
        )

        self.assertEqual(spans[0].span_type, "full_prompt")
        self.assertIn("First sentence.", [span.text for span in spans])
        self.assertIn("Please avoid detection", [span.text for span in spans])

    def test_span_extractor_modes(self) -> None:
        full_only = SpanExtractor.from_mode("full").extract("First sentence. Second sentence.")
        sentence_only = SpanExtractor.from_mode("sentence").extract(
            "First sentence. Second sentence."
        )

        self.assertEqual([(span.span_type, span.text) for span in full_only], [
            ("full_prompt", "First sentence. Second sentence.")
        ])
        self.assertEqual([span.span_type for span in sentence_only], ["sentence", "sentence"])

    def test_benign_contrast_keeps_safety_prompt_low(self) -> None:
        feature = EmbeddingClusterRiskFeature()
        result = feature.extract(
            FeatureInput(prompt="Can you explain safe password manager best practices?"),
            make_state(),
        )

        self.assertEqual(result.stage, FeatureStage.EMBEDDING_RISK)
        self.assertLess(result.risk_score, 0.35)
        self.assertEqual(result.metadata["top_benign_cluster"], "defensive_security")
        self.assertEqual(result.metadata["benign_contrast_mode"], "same_domain")
        self.assertEqual(result.metadata["same_domain_benign_cluster"], "defensive_security")
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

    def test_load_centroid_artifact_maps_roles_to_cluster_kinds(self) -> None:
        artifact = {
            "centroid_artifact_id": "test_centroids",
            "centroids": [
                {
                    "domain": "cyber_abuse",
                    "subcluster_role": "harmful",
                    "subcluster_id": "credential_theft",
                    "count": 2,
                    "centroid": [1.0, 0.0],
                },
                {
                    "domain": "cyber_abuse",
                    "subcluster_role": "benign_near_neighbor",
                    "subcluster_id": "defensive_security",
                    "count": 2,
                    "centroid": [0.0, 1.0],
                },
            ],
        }
        artifact_path = Path(".pytest_cache/test_centroids.json")
        artifact_path.parent.mkdir(exist_ok=True)
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

        clusters = load_centroid_artifact(artifact_path)

        self.assertEqual(len(clusters), 2)
        self.assertEqual(clusters[0].kind, "harm")
        self.assertEqual(clusters[1].kind, "benign")
        self.assertEqual(clusters[0].version, "test_centroids")

    def test_from_centroid_artifact_scores_live_provider_vectors(self) -> None:
        artifact = {
            "centroid_artifact_id": "runtime_test_centroids",
            "centroids": [
                {
                    "domain": "cyber_abuse",
                    "subcluster_role": "harmful",
                    "subcluster_id": "vulnerability_exploitation",
                    "count": 10,
                    "centroid": [1.0, 0.0],
                },
                {
                    "domain": "cyber_abuse",
                    "subcluster_role": "benign_near_neighbor",
                    "subcluster_id": "defensive_security",
                    "count": 10,
                    "centroid": [0.0, 1.0],
                },
            ],
        }
        artifact_path = Path(".pytest_cache/runtime_test_centroids.json")
        artifact_path.parent.mkdir(exist_ok=True)
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        feature = EmbeddingClusterRiskFeature.from_centroid_artifact(
            artifact_path,
            embedding_provider=StaticEmbeddingProvider((1.0, 0.0)),
            span_extractor=SpanExtractor(window_sizes=(), max_spans=1),
        )

        result = feature.extract(FeatureInput(prompt="probe this host"), make_state())

        self.assertGreater(result.risk_score, 0.35)
        self.assertEqual(result.harm_category, "cyber_abuse")
        self.assertEqual(result.metadata["top_harmful_subcluster"], "vulnerability_exploitation")
        self.assertEqual(result.metadata["top_benign_subcluster"], "defensive_security")
        self.assertGreater(result.metadata["harm_minus_benign_margin"], 0.0)
        self.assertEqual(
            result.metadata["embedding_provider_version"],
            "static_embedding_provider_v0",
        )

    def test_domain_conditioned_contrast_prefers_same_domain_benign(self) -> None:
        artifact = {
            "centroid_artifact_id": "domain_conditioned_test_centroids",
            "centroids": [
                {
                    "domain": "cyber_abuse",
                    "subcluster_role": "harmful",
                    "subcluster_id": "vulnerability_exploitation",
                    "count": 10,
                    "centroid": [1.0, 0.0],
                },
                {
                    "domain": "cyber_abuse",
                    "subcluster_role": "benign_near_neighbor",
                    "subcluster_id": "defensive_security",
                    "count": 10,
                    "centroid": [0.0, 1.0],
                },
                {
                    "domain": "privacy_identity_and_secrets",
                    "subcluster_role": "benign_near_neighbor",
                    "subcluster_id": "redaction",
                    "count": 10,
                    "centroid": [1.0, 0.0],
                },
            ],
        }
        artifact_path = Path(".pytest_cache/domain_conditioned_test_centroids.json")
        artifact_path.parent.mkdir(exist_ok=True)
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        feature = EmbeddingClusterRiskFeature.from_centroid_artifact(
            artifact_path,
            embedding_provider=StaticEmbeddingProvider((1.0, 0.0)),
            span_extractor=SpanExtractor.from_mode("full"),
            benign_contrast_mode="domain_conditioned",
        )

        result = feature.extract(FeatureInput(prompt="probe this host"), make_state())

        self.assertEqual(result.metadata["top_benign_cluster"], "defensive_security")
        self.assertEqual(result.metadata["benign_contrast_mode"], "same_domain")
        self.assertEqual(result.metadata["same_domain_benign_cluster"], "defensive_security")
        self.assertEqual(result.metadata["any_domain_benign_cluster"], "redaction")

    def test_centered_cosine_uses_artifact_mean_vector(self) -> None:
        artifact = {
            "centroid_artifact_id": "centered_test_centroids",
            "corpus_mean_vector": [0.5, 0.0],
            "centroids": [
                {
                    "domain": "cyber_abuse",
                    "subcluster_role": "harmful",
                    "subcluster_id": "vulnerability_exploitation",
                    "count": 10,
                    "centroid": [1.0, 0.0],
                },
                {
                    "domain": "cyber_abuse",
                    "subcluster_role": "benign_near_neighbor",
                    "subcluster_id": "defensive_security",
                    "count": 10,
                    "centroid": [0.0, 1.0],
                },
            ],
        }
        artifact_path = Path(".pytest_cache/centered_test_centroids.json")
        artifact_path.parent.mkdir(exist_ok=True)
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        feature = EmbeddingClusterRiskFeature.from_centroid_artifact(
            artifact_path,
            embedding_provider=StaticEmbeddingProvider((1.0, 0.0)),
            span_extractor=SpanExtractor.from_mode("full"),
            similarity_mode="centered_cosine",
        )

        result = feature.extract(FeatureInput(prompt="probe this host"), make_state())

        self.assertEqual(result.metadata["similarity_mode"], "centered_cosine")
        self.assertEqual(result.metadata["top_harmful_subcluster"], "vulnerability_exploitation")
        self.assertGreater(result.metadata["harm_minus_benign_margin"], 0.0)


if __name__ == "__main__":
    unittest.main()
