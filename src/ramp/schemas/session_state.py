from __future__ import annotations

from dataclasses import dataclass, field

from ramp.schemas.feature_result import FeatureResult


@dataclass
class SessionTurn:
    request_id: str
    features: list[FeatureResult] = field(default_factory=list)
    final_risk: float | None = None


@dataclass
class SessionState:
    session_id: str
    turns: list[SessionTurn] = field(default_factory=list)
    rolling_risk: float = 0.0
    top_accumulating_category: str | None = None

    def append_turn(self, turn: SessionTurn) -> None:
        self.turns.append(turn)
        risks = [
            candidate.final_risk
            for candidate in self.turns
            if candidate.final_risk is not None
        ]
        if risks:
            recent = risks[-5:]
            self.rolling_risk = sum(recent) / len(recent)
