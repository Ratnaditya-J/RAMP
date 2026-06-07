from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ramp.features.base import FeatureExtractor, FeatureInput
from ramp.risk_state import RiskState
from ramp.schemas.feature_result import FeatureResult, FeatureStage

Vector = tuple[float, ...]


@dataclass(frozen=True)
class TextSpan:
    text: str
    span_type: str
    start_token: int | None = None
    end_token: int | None = None


@dataclass(frozen=True)
class EmbeddingCluster:
    cluster_id: str
    category: str
    centroid: Vector
    kind: str
    version: str = "demo_clusters_v0"
    description: str = ""
    harm_domain: str = "unknown"
    subcluster_role: str = "harmful"


class EmbeddingProvider:
    version = "unknown_embedding_provider"

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        raise NotImplementedError


class KeywordVectorEmbeddingProvider(EmbeddingProvider):
    """Deterministic dev provider shaped like a real embedding adapter.

    This is not a semantic embedding model. It gives tests and examples stable vectors while
    preserving the same interface a target-model embedding provider will use later.
    """

    version = "keyword_vector_embedding_v0"

    dimensions = (
        "cyber_misuse",
        "evasion",
        "credential_access",
        "weapons",
        "benign_security",
        "safety_framing",
    )
    lexicon: dict[str, Vector] = {
        "bypass": (0.8, 0.9, 0.4, 0.0, 0.0, 0.0),
        "avoid detection": (0.5, 1.0, 0.2, 0.0, 0.0, 0.0),
        "evade": (0.4, 1.0, 0.1, 0.0, 0.0, 0.0),
        "credential": (0.7, 0.2, 1.0, 0.0, 0.0, 0.0),
        "access credentials": (0.8, 0.2, 1.0, 0.0, 0.0, 0.0),
        "exploit": (0.9, 0.4, 0.2, 0.0, 0.0, 0.0),
        "weapon": (0.0, 0.1, 0.0, 1.0, 0.0, 0.0),
        "step by step": (0.3, 0.4, 0.0, 0.0, 0.0, 0.0),
        "make it stronger": (0.3, 0.5, 0.0, 0.2, 0.0, 0.0),
        "make more effective": (0.3, 0.5, 0.0, 0.2, 0.0, 0.0),
        "increase success rate": (0.4, 0.6, 0.0, 0.0, 0.0, 0.0),
        "automate at scale": (0.5, 0.6, 0.0, 0.0, 0.0, 0.0),
        "defensive": (0.2, 0.0, 0.0, 0.0, 1.0, 0.6),
        "incident response": (0.2, 0.0, 0.0, 0.0, 1.0, 0.7),
        "password manager": (0.0, 0.0, 0.0, 0.0, 1.0, 0.8),
        "best practices": (0.0, 0.0, 0.0, 0.0, 0.6, 0.7),
        "safe": (0.0, 0.0, 0.0, 0.0, 0.3, 1.0),
        "safety": (0.0, 0.0, 0.0, 0.0, 0.4, 1.0),
        "patch": (0.2, 0.0, 0.0, 0.0, 0.9, 0.5),
        "legal": (0.0, 0.0, 0.0, 0.0, 0.5, 0.8),
    }

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> Vector:
        normalized = text.lower()
        score = [0.0 for _ in self.dimensions]
        for term, vector in self.lexicon.items():
            if term in normalized:
                for idx, value in enumerate(vector):
                    score[idx] += value
        if not any(score):
            return tuple(0.0 for _ in self.dimensions)
        return normalize(tuple(score))


class SpanExtractor:
    def __init__(self, *, window_sizes: tuple[int, ...] = (3, 5), max_spans: int = 32) -> None:
        self.window_sizes = window_sizes
        self.max_spans = max_spans

    def extract(self, text: str) -> list[TextSpan]:
        spans: list[TextSpan] = []
        stripped = text.strip()
        if not stripped:
            return spans

        spans.append(TextSpan(text=stripped, span_type="full_prompt"))
        for sentence in re.split(r"(?<=[.!?])\s+", stripped):
            sentence = sentence.strip()
            if sentence and sentence != stripped:
                spans.append(TextSpan(text=sentence, span_type="sentence"))

        tokens = re.findall(r"[A-Za-z0-9_'-]+", stripped)
        for window_size in self.window_sizes:
            if len(tokens) < window_size:
                continue
            for start in range(0, len(tokens) - window_size + 1):
                end = start + window_size
                spans.append(
                    TextSpan(
                        text=" ".join(tokens[start:end]),
                        span_type=f"sliding_{window_size}",
                        start_token=start,
                        end_token=end,
                    )
                )
                if len(spans) >= self.max_spans:
                    return dedupe_spans(spans)
        return dedupe_spans(spans)


class EmbeddingClusterRiskFeature(FeatureExtractor):
    stage = FeatureStage.EMBEDDING_RISK

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        harm_clusters: Sequence[EmbeddingCluster] | None = None,
        benign_clusters: Sequence[EmbeddingCluster] | None = None,
        span_extractor: SpanExtractor | None = None,
        trigger_margin: float = 0.18,
    ) -> None:
        self.embedding_provider = embedding_provider or KeywordVectorEmbeddingProvider()
        self.harm_clusters = tuple(harm_clusters or demo_harm_clusters())
        self.benign_clusters = tuple(benign_clusters or demo_benign_clusters())
        self.harmful_action_clusters = tuple(
            cluster for cluster in self.harm_clusters if cluster.subcluster_role == "harmful"
        )
        self.evasion_clusters = tuple(
            cluster for cluster in self.harm_clusters if cluster.subcluster_role == "evasion"
        )
        self.optimization_clusters = tuple(
            cluster for cluster in self.harm_clusters if cluster.subcluster_role == "optimization"
        )
        self.span_extractor = span_extractor or SpanExtractor()
        self.trigger_margin = trigger_margin

    @classmethod
    def from_centroid_artifact(
        cls,
        path: str | Path,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        span_extractor: SpanExtractor | None = None,
        trigger_margin: float = 0.18,
    ) -> EmbeddingClusterRiskFeature:
        clusters = load_centroid_artifact(path)
        harm_clusters = [cluster for cluster in clusters if cluster.kind == "harm"]
        benign_clusters = [cluster for cluster in clusters if cluster.kind == "benign"]
        return cls(
            embedding_provider=embedding_provider,
            harm_clusters=harm_clusters,
            benign_clusters=benign_clusters,
            span_extractor=span_extractor,
            trigger_margin=trigger_margin,
        )

    def extract(self, feature_input: FeatureInput, state: RiskState) -> FeatureResult:
        spans = self.span_extractor.extract(feature_input.prompt)
        if not spans:
            return self._empty_result()

        vectors = self.embedding_provider.embed([span.text for span in spans])
        scored = [
            self._score_span(span, vector)
            for span, vector in zip(spans, vectors, strict=True)
        ]
        positive_scored = [item for item in scored if item.risk_margin > 0.0]
        top = (
            max(positive_scored, key=lambda item: item.risk_margin)
            if positive_scored
            else min(scored, key=lambda item: item.margin)
        )
        triggered = [item for item in scored if item.risk_margin >= self.trigger_margin]
        risk_score = self._risk_score(
            top.risk_margin,
            len(triggered),
            evasion_similarity=top.evasion_similarity,
            optimization_similarity=top.optimization_similarity,
        )
        confidence = min(0.90, 0.55 + abs(top.risk_margin) * 0.35 + min(len(scored), 10) * 0.01)
        top_risk_cluster = top.top_risk_cluster()

        return FeatureResult(
            stage=self.stage,
            risk_score=risk_score,
            confidence=confidence,
            label="semantic_proximity",
            harm_category=top_risk_cluster.category if risk_score >= 0.35 else None,
            cost_tier=1,
            version=f"embedding_cluster_risk:{self.embedding_provider.version}",
            metadata={
                "embedding_provider_version": self.embedding_provider.version,
                "harm_cluster_version": top_risk_cluster.version,
                "benign_cluster_version": top.benign_cluster.version,
                "top_harm_domain": top_risk_cluster.harm_domain,
                "top_harmful_subcluster": top.harmful_cluster.cluster_id,
                "top_benign_subcluster": top.benign_cluster.cluster_id,
                "top_evasion_subcluster": (
                    top.evasion_cluster.cluster_id if top.evasion_cluster else None
                ),
                "top_optimization_subcluster": (
                    top.optimization_cluster.cluster_id if top.optimization_cluster else None
                ),
                "top_harm_cluster": top_risk_cluster.cluster_id,
                "top_benign_cluster": top.benign_cluster.cluster_id,
                "harm_similarity": top.harm_similarity,
                "benign_similarity": top.benign_similarity,
                "harm_minus_benign_margin": top.margin,
                "risk_margin": top.risk_margin,
                "evasion_similarity": top.evasion_similarity,
                "optimization_similarity": top.optimization_similarity,
                "evasion_activated": (
                    top.evasion_similarity - top.benign_similarity >= self.trigger_margin
                ),
                "optimization_activated": (
                    top.optimization_similarity - top.benign_similarity >= self.trigger_margin
                ),
                "top_span": top.span.text,
                "top_span_type": top.span.span_type,
                "num_spans_scored": len(scored),
                "num_triggered_spans": len(triggered),
                "trigger_margin": self.trigger_margin,
            },
        )

    def _empty_result(self) -> FeatureResult:
        return FeatureResult(
            stage=self.stage,
            risk_score=0.0,
            confidence=0.0,
            label="missing_prompt",
            cost_tier=1,
            version=f"embedding_cluster_risk:{self.embedding_provider.version}",
            metadata={"reason": "no prompt text"},
        )

    def _score_span(self, span: TextSpan, vector: Vector) -> SpanClusterScore:
        harm_cluster, harm_similarity = nearest_cluster(vector, self.harmful_action_clusters)
        benign_cluster, benign_similarity = nearest_cluster(vector, self.benign_clusters)
        evasion_cluster, evasion_similarity = nearest_cluster_or_none(vector, self.evasion_clusters)
        optimization_cluster, optimization_similarity = nearest_cluster_or_none(
            vector,
            self.optimization_clusters,
        )
        return SpanClusterScore(
            span=span,
            harmful_cluster=harm_cluster,
            benign_cluster=benign_cluster,
            evasion_cluster=evasion_cluster,
            optimization_cluster=optimization_cluster,
            harm_similarity=harm_similarity,
            benign_similarity=benign_similarity,
            evasion_similarity=evasion_similarity,
            optimization_similarity=optimization_similarity,
            margin=harm_similarity - benign_similarity,
        )

    def _risk_score(
        self,
        margin: float,
        triggered_count: int,
        *,
        evasion_similarity: float,
        optimization_similarity: float,
    ) -> float:
        if margin <= 0:
            return max(0.04, 0.12 + margin * 0.10)
        evasion_bonus = 0.06 if evasion_similarity >= 0.50 else 0.0
        optimization_bonus = 0.04 if optimization_similarity >= 0.50 else 0.0
        base_score = 0.12 + margin * 0.78 + min(triggered_count, 5) * 0.04
        return min(0.95, base_score + evasion_bonus + optimization_bonus)


@dataclass(frozen=True)
class SpanClusterScore:
    span: TextSpan
    harmful_cluster: EmbeddingCluster
    benign_cluster: EmbeddingCluster
    evasion_cluster: EmbeddingCluster | None
    optimization_cluster: EmbeddingCluster | None
    harm_similarity: float
    benign_similarity: float
    evasion_similarity: float
    optimization_similarity: float
    margin: float

    @property
    def risk_margin(self) -> float:
        return max(
            self.margin,
            self.evasion_similarity - self.benign_similarity,
            self.optimization_similarity - self.benign_similarity,
        )

    def top_risk_cluster(self) -> EmbeddingCluster:
        candidates = [
            (self.harmful_cluster, self.margin),
        ]
        if self.evasion_cluster is not None:
            candidates.append(
                (self.evasion_cluster, self.evasion_similarity - self.benign_similarity)
            )
        if self.optimization_cluster is not None:
            candidates.append(
                (
                    self.optimization_cluster,
                    self.optimization_similarity - self.benign_similarity,
                )
            )
        return max(candidates, key=lambda item: item[1])[0]


def demo_harm_clusters() -> list[EmbeddingCluster]:
    return [
        EmbeddingCluster(
            cluster_id="credential_access",
            category="credential_access_misuse",
            kind="harm",
            centroid=normalize((0.8, 0.2, 1.0, 0.0, 0.0, 0.0)),
            description="Credential, account, or access misuse.",
            harm_domain="cyber",
            subcluster_role="harmful",
        ),
        EmbeddingCluster(
            cluster_id="unauthorized_intrusion",
            category="cyber_misuse",
            kind="harm",
            centroid=normalize((0.7, 1.0, 0.3, 0.0, 0.0, 0.0)),
            description="Bypass, exploit chaining, or unauthorized access attempts.",
            harm_domain="cyber",
            subcluster_role="harmful",
        ),
        EmbeddingCluster(
            cluster_id="avoid_detection",
            category="cyber_misuse",
            kind="harm",
            centroid=normalize((0.4, 1.0, 0.1, 0.0, 0.0, 0.0)),
            description="Evasion, concealment, bypass, or avoidance of detection.",
            harm_domain="cyber",
            subcluster_role="evasion",
        ),
        EmbeddingCluster(
            cluster_id="make_more_effective",
            category="cyber_misuse",
            kind="harm",
            centroid=normalize((0.35, 0.55, 0.0, 0.2, 0.0, 0.0)),
            description="Optimization, effectiveness, success-rate, or scale-up requests.",
            harm_domain="cyber",
            subcluster_role="optimization",
        ),
        EmbeddingCluster(
            cluster_id="weapons_construction",
            category="weapons",
            kind="harm",
            centroid=normalize((0.0, 0.1, 0.0, 1.0, 0.0, 0.0)),
            description="Weapon construction or enhancement.",
            harm_domain="weapons",
            subcluster_role="harmful",
        ),
    ]


def demo_benign_clusters() -> list[EmbeddingCluster]:
    return [
        EmbeddingCluster(
            cluster_id="defensive_security",
            category="defensive_cybersecurity",
            kind="benign",
            centroid=normalize((0.2, 0.0, 0.0, 0.0, 1.0, 0.6)),
            description="Defensive security, patching, monitoring, and response.",
            harm_domain="cyber",
            subcluster_role="benign_near_neighbor",
        ),
        EmbeddingCluster(
            cluster_id="general_safety",
            category="safety_education",
            kind="benign",
            centroid=normalize((0.0, 0.0, 0.0, 0.0, 0.4, 1.0)),
            description="Safety, legality, and high-level educational framing.",
            harm_domain="general",
            subcluster_role="benign_near_neighbor",
        ),
    ]


def load_centroid_artifact(path: str | Path) -> list[EmbeddingCluster]:
    artifact_path = Path(path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact_id = artifact["centroid_artifact_id"]
    clusters: list[EmbeddingCluster] = []
    for centroid in artifact["centroids"]:
        role = centroid["subcluster_role"]
        kind = "benign" if role == "benign_near_neighbor" else "harm"
        clusters.append(
            EmbeddingCluster(
                cluster_id=centroid["subcluster_id"],
                category=centroid["domain"],
                kind=kind,
                centroid=tuple(float(value) for value in centroid["centroid"]),
                version=artifact_id,
                description=(
                    f"{centroid['domain']} / {role} / {centroid['subcluster_id']} "
                    f"from {centroid['count']} spans"
                ),
                harm_domain=centroid["domain"],
                subcluster_role=role,
            )
        )
    return clusters


def nearest_cluster(
    vector: Vector,
    clusters: Sequence[EmbeddingCluster],
) -> tuple[EmbeddingCluster, float]:
    if not clusters:
        raise ValueError("at least one cluster is required")
    return max(
        ((cluster, cosine_similarity(vector, cluster.centroid)) for cluster in clusters),
        key=lambda item: item[1],
    )


def nearest_cluster_or_none(
    vector: Vector,
    clusters: Sequence[EmbeddingCluster],
) -> tuple[EmbeddingCluster | None, float]:
    if not clusters:
        return None, 0.0
    return nearest_cluster(vector, clusters)


def cosine_similarity(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimensionality")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def normalize(vector: Iterable[float]) -> Vector:
    values = tuple(float(value) for value in vector)
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return tuple(value / norm for value in values)


def dedupe_spans(spans: Sequence[TextSpan]) -> list[TextSpan]:
    seen: set[str] = set()
    deduped: list[TextSpan] = []
    for span in spans:
        key = span.text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(span)
    return deduped


# Backwards-compatible export for existing examples and tests.
LexicalEmbeddingRiskFeature = EmbeddingClusterRiskFeature
