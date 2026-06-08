from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ramp.features.base import FeatureExtractor, FeatureInput
from ramp.risk_state import RiskState
from ramp.schemas.feature_result import FeatureResult, FeatureStage

Vector = tuple[float, ...]


class ActivationProvider:
    version = "unknown_activation_provider"

    def activation(self, feature_input: FeatureInput) -> Vector | None:
        raise NotImplementedError


class ContextActivationProvider(ActivationProvider):
    version = "context_activation_provider_v0"

    def __init__(self, context_key: str = "activation_vector") -> None:
        self.context_key = context_key

    def activation(self, feature_input: FeatureInput) -> Vector | None:
        vector = feature_input.context.get(self.context_key)
        if vector is None:
            return None
        return tuple(float(value) for value in vector)


@dataclass(frozen=True)
class LinearActivationProbe:
    artifact_id: str
    layer_id: str
    weights: Vector
    bias: float
    mean: Vector
    scale: Vector
    threshold: float
    model_version: str
    training_summary: dict[str, Any]

    @classmethod
    def from_artifact(cls, path: str | Path) -> LinearActivationProbe:
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            artifact_id=artifact["artifact_id"],
            layer_id=str(artifact["layer_id"]),
            weights=tuple(float(value) for value in artifact["weights"]),
            bias=float(artifact["bias"]),
            mean=tuple(float(value) for value in artifact["mean"]),
            scale=tuple(float(value) for value in artifact["scale"]),
            threshold=float(artifact["selected_threshold"]),
            model_version=artifact.get("model_version", "linear_activation_probe_v0"),
            training_summary=artifact.get("training_summary", {}),
        )

    def probability(self, vector: Sequence[float]) -> float:
        if len(vector) != len(self.weights):
            raise ValueError(
                f"activation dimension mismatch: got {len(vector)}, expected {len(self.weights)}"
            )
        logit = self.bias
        for value, mean, scale, weight in zip(
            vector,
            self.mean,
            self.scale,
            self.weights,
            strict=True,
        ):
            divisor = scale if scale > 0 else 1.0
            logit += ((float(value) - mean) / divisor) * weight
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))


class ActivationProbeFeature(FeatureExtractor):
    stage = FeatureStage.ACTIVATION_PROBE

    def __init__(
        self,
        probe: LinearActivationProbe,
        *,
        activation_provider: ActivationProvider | None = None,
    ) -> None:
        self.probe = probe
        self.activation_provider = activation_provider or ContextActivationProvider()

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        *,
        activation_provider: ActivationProvider | None = None,
    ) -> ActivationProbeFeature:
        return cls(
            LinearActivationProbe.from_artifact(path),
            activation_provider=activation_provider,
        )

    def extract(self, feature_input: FeatureInput, state: RiskState) -> FeatureResult:
        start = time.perf_counter()
        vector = self.activation_provider.activation(feature_input)
        if vector is None:
            return FeatureResult(
                stage=self.stage,
                risk_score=0.0,
                confidence=0.0,
                label="missing_activation",
                cost_tier=2,
                version=self.probe.model_version,
                metadata={
                    "reason": "activation vector unavailable",
                    "probe_artifact_id": self.probe.artifact_id,
                    "selected_layer": self.probe.layer_id,
                    "activation_provider_version": self.activation_provider.version,
                },
            )

        probability = self.probe.probability(vector)
        is_activated = probability >= self.probe.threshold
        latency_ms = round((time.perf_counter() - start) * 1000)
        return FeatureResult(
            stage=self.stage,
            risk_score=probability,
            confidence=0.72 if is_activated else 0.62,
            label="activation_probe_unsafe" if is_activated else "activation_probe_safe",
            harm_category="activation_probe" if is_activated else None,
            latency_ms=latency_ms,
            cost_tier=2,
            version=self.probe.model_version,
            metadata={
                "probe_artifact_id": self.probe.artifact_id,
                "selected_layer": self.probe.layer_id,
                "activation_provider_version": self.activation_provider.version,
                "unsafe_probability": probability,
                "selected_threshold": self.probe.threshold,
                "activated": is_activated,
                "training_summary": self.probe.training_summary,
            },
        )


class StubActivationProbeFeature(FeatureExtractor):
    stage = FeatureStage.ACTIVATION_PROBE

    def extract(self, feature_input: FeatureInput, state: RiskState) -> FeatureResult:
        inherited_risk = state.current_risk
        probe_score = min(0.95, inherited_risk + 0.08)
        return FeatureResult(
            stage=self.stage,
            risk_score=probe_score,
            confidence=0.64,
            label="probe_placeholder",
            cost_tier=2,
            version="stub_activation_probe_v0",
            metadata={"selected_layer": "demo_layer", "probe_score": probe_score},
        )
