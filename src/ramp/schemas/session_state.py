from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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


SEVERITY_RANK = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


@dataclass(frozen=True)
class SalientSessionTurn:
    turn_id: str
    turn_index: int
    role: str
    text: str
    risk_score: float
    classifier_label: str | None = None
    harm_domain: str | None = None
    subcluster_id: str | None = None
    harm_severity: str = "none"
    salience_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompactSessionState:
    session_id: str
    turn_count: int
    domains_seen: tuple[str, ...] = ()
    subclusters_seen: tuple[str, ...] = ()
    highest_severity: str = "none"
    highest_turn_risk: float = 0.0
    mean_top_turn_risk: float = 0.0
    risk_trend: str = "flat"
    intent_progression: bool = False
    evasion_attempts: int = 0
    operational_details_requested: bool = False
    benign_cover_story: bool = False
    cross_turn_composition: bool = False
    salient_turns: tuple[SalientSessionTurn, ...] = ()
    safety_summary: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def highest_severity_rank(self) -> int:
        return SEVERITY_RANK.get(self.highest_severity, 0)
