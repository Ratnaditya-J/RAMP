from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ramp.risk_state import RiskState
from ramp.schemas.feature_result import FeatureResult, FeatureStage


@dataclass(frozen=True)
class FeatureInput:
    prompt: str = ""
    output: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)


class FeatureExtractor(ABC):
    stage: FeatureStage

    @abstractmethod
    def extract(self, feature_input: FeatureInput, state: RiskState) -> FeatureResult:
        """Return one feature result for the current RAMP state."""

