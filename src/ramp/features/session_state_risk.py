from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ramp.features.base import FeatureExtractor, FeatureInput
from ramp.risk_state import RiskState
from ramp.schemas.feature_result import FeatureResult, FeatureStage
from ramp.schemas.session_state import (
    SEVERITY_RANK,
    CompactSessionState,
    SalientSessionTurn,
)

EVASION_TERMS = (
    "bypass",
    "jailbreak",
    "ignore previous",
    "avoid detection",
    "evade",
    "filter",
    "rephrase",
    "encode",
    "roleplay",
    "hypothetical",
)
OPERATIONAL_TERMS = (
    "step by step",
    "specific",
    "detailed",
    "instructions",
    "procedure",
    "implement",
    "exploit",
    "automate",
    "deploy",
    "make",
    "build",
    "obtain",
    "hide",
)
COVER_STORY_TERMS = (
    "for a class",
    "for research",
    "for awareness",
    "for a story",
    "fictional",
    "hypothetical",
    "authorized",
    "professor",
    "dissertation",
    "training",
)
HARM_TERMS = (
    "weapon",
    "bomb",
    "explosive",
    "phishing",
    "credential",
    "malware",
    "exploit",
    "suicide",
    "self-harm",
    "poison",
    "toxin",
    "dox",
    "private patient",
    "illegal",
)
PROGRESSION_TERMS = (
    "now",
    "next",
    "more",
    "again",
    "continue",
    "make it",
    "turn this into",
    "add",
    "expand",
    "opposite meaning",
)


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def count_terms(text: str, terms: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term in lowered)


def compact_text(text: str, max_chars: int = 140) -> str:
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


def turn_text(turn: dict[str, Any]) -> str:
    return str(turn.get("text") or turn.get("span_text") or "").strip()


def turn_risk_score(turn: dict[str, Any]) -> float:
    try:
        return float(turn.get("risk_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def severity_rank(turn: dict[str, Any]) -> int:
    return SEVERITY_RANK.get(str(turn.get("harm_severity") or "none").lower(), 0)


@dataclass(frozen=True)
class SessionStateUpdater:
    """Builds compact, safety-specific session state from scored turns."""

    max_salient_turns: int = 6

    def update(self, session: dict[str, Any]) -> CompactSessionState:
        turns = list(session.get("turns") or [])
        risks = [turn_risk_score(turn) for turn in turns]
        top_risks = sorted(risks, reverse=True)[: min(3, len(risks))]
        domains = sorted(
            {str(turn.get("harm_domain")) for turn in turns if turn.get("harm_domain")}
        )
        subclusters = sorted(
            {str(turn.get("subcluster_id")) for turn in turns if turn.get("subcluster_id")}
        )
        highest_turn = max(turns, key=severity_rank, default={})
        highest_severity = str(highest_turn.get("harm_severity") or "none").lower()
        text_by_turn = [turn_text(turn) for turn in turns]
        full_text = "\n".join(text_by_turn)
        evasion_attempts = sum(count_terms(text, EVASION_TERMS) for text in text_by_turn)
        operational_hits = sum(1 for text in text_by_turn if contains_any(text, OPERATIONAL_TERMS))
        cover_hits = sum(1 for text in text_by_turn if contains_any(text, COVER_STORY_TERMS))
        harm_hits = sum(1 for text in text_by_turn if contains_any(text, HARM_TERMS))
        progression_hits = sum(
            1 for text in text_by_turn[1:] if contains_any(text, PROGRESSION_TERMS)
        )
        risk_trend = self._risk_trend(risks)
        salient_turns = self._salient_turns(turns)
        intent_progression = (
            progression_hits >= 2
            or (
                len(turns) >= 3
                and operational_hits >= 1
                and (harm_hits >= 1 or evasion_attempts >= 1)
            )
            or (risks and risks[-1] - risks[0] >= 0.20)
        )
        cross_turn_composition = (
            len(domains) >= 2
            or (cover_hits >= 1 and operational_hits >= 1 and harm_hits >= 1)
            or (progression_hits >= 1 and operational_hits >= 2)
        )
        return CompactSessionState(
            session_id=str(session.get("session_id") or ""),
            turn_count=len(turns),
            domains_seen=tuple(domains),
            subclusters_seen=tuple(subclusters),
            highest_severity=highest_severity,
            highest_turn_risk=max(risks) if risks else 0.0,
            mean_top_turn_risk=sum(top_risks) / len(top_risks) if top_risks else 0.0,
            risk_trend=risk_trend,
            intent_progression=intent_progression,
            evasion_attempts=evasion_attempts,
            operational_details_requested=operational_hits > 0,
            benign_cover_story=cover_hits > 0,
            cross_turn_composition=cross_turn_composition,
            salient_turns=tuple(salient_turns),
            safety_summary=self._summary(
                full_text,
                intent_progression=intent_progression,
                cross_turn_composition=cross_turn_composition,
                evasion_attempts=evasion_attempts,
                operational_hits=operational_hits,
            ),
            evidence={
                "harm_keyword_hits": harm_hits,
                "operational_hits": operational_hits,
                "cover_story_hits": cover_hits,
                "progression_hits": progression_hits,
                "risk_values": risks,
            },
        )

    def _risk_trend(self, risks: list[float]) -> str:
        if len(risks) < 2:
            return "flat"
        early = sum(risks[: min(2, len(risks))]) / min(2, len(risks))
        late = sum(risks[-min(2, len(risks)):]) / min(2, len(risks))
        if late - early >= 0.15:
            return "increasing"
        if early - late >= 0.15:
            return "decreasing"
        return "flat"

    def _salient_turns(self, turns: list[dict[str, Any]]) -> list[SalientSessionTurn]:
        candidates = []
        for turn in turns:
            text = turn_text(turn)
            reasons = []
            if turn_risk_score(turn) >= 0.35:
                reasons.append("risk_score")
            if contains_any(text, EVASION_TERMS):
                reasons.append("evasion")
            if contains_any(text, OPERATIONAL_TERMS):
                reasons.append("operational_detail")
            if contains_any(text, COVER_STORY_TERMS):
                reasons.append("cover_story")
            if severity_rank(turn) >= 3:
                reasons.append("high_severity")
            if not reasons:
                continue
            candidates.append(
                SalientSessionTurn(
                    turn_id=str(turn.get("turn_id") or ""),
                    turn_index=int(turn.get("turn_index") or 0),
                    role=str(turn.get("role") or "user"),
                    text=compact_text(text),
                    risk_score=turn_risk_score(turn),
                    classifier_label=turn.get("classifier_label"),
                    harm_domain=turn.get("harm_domain"),
                    subcluster_id=turn.get("subcluster_id"),
                    harm_severity=str(turn.get("harm_severity") or "none"),
                    salience_reasons=tuple(reasons),
                )
            )
        return sorted(
            candidates,
            key=lambda turn: (turn.risk_score, len(turn.salience_reasons), turn.turn_index),
            reverse=True,
        )[: self.max_salient_turns]

    def _summary(
        self,
        full_text: str,
        *,
        intent_progression: bool,
        cross_turn_composition: bool,
        evasion_attempts: int,
        operational_hits: int,
    ) -> str:
        parts = []
        if intent_progression:
            parts.append("intent appears to progress across turns")
        if cross_turn_composition:
            parts.append("multiple turns combine into stronger risk evidence")
        if evasion_attempts:
            parts.append(f"{evasion_attempts} evasion cues")
        if operational_hits:
            parts.append(f"{operational_hits} operational-detail cues")
        if not parts:
            parts.append("no compact session-risk pattern detected")
        return "; ".join(parts) + f"; transcript_chars={len(full_text)}"


@dataclass(frozen=True)
class CompactSessionRiskScorer:
    """Scores compact session state without sending the full transcript to a classifier."""

    def score(self, state: CompactSessionState) -> tuple[float, dict[str, float]]:
        severity = min(1.0, state.highest_severity_rank / 4)
        trend = (
            1.0
            if state.risk_trend == "increasing"
            else 0.35
            if state.risk_trend == "flat"
            else 0.0
        )
        evasion = min(1.0, state.evasion_attempts / 3)
        operational = 1.0 if state.operational_details_requested else 0.0
        progression = 1.0 if state.intent_progression else 0.0
        composition = 1.0 if state.cross_turn_composition else 0.0
        cover = 1.0 if state.benign_cover_story else 0.0
        contributions = {
            "highest_turn_risk": 0.22 * state.highest_turn_risk,
            "mean_top_turn_risk": 0.14 * state.mean_top_turn_risk,
            "severity": 0.12 * severity,
            "trend": 0.10 * trend,
            "progression": 0.17 * progression,
            "operational": 0.10 * operational,
            "evasion": 0.08 * evasion,
            "composition": 0.05 * composition,
            "cover_story": 0.03 * cover,
        }
        return min(0.99, sum(contributions.values())), contributions


def render_compact_session_evidence(state: CompactSessionState) -> str:
    salient = "\n".join(
        (
            f"- turn-{turn.turn_index + 1} role={turn.role} risk={turn.risk_score:.2f} "
            f"reasons={','.join(turn.salience_reasons)} text={turn.text}"
        )
        for turn in state.salient_turns
    )
    if not salient:
        salient = "- no salient turns selected"
    return "\n".join(
        [
            "Classify the safety risk of this multi-turn session from compact evidence.",
            f"turn_count: {state.turn_count}",
            f"domains_seen: {', '.join(state.domains_seen) or 'none'}",
            f"subclusters_seen: {', '.join(state.subclusters_seen) or 'none'}",
            f"highest_severity: {state.highest_severity}",
            f"highest_turn_risk: {state.highest_turn_risk:.2f}",
            f"mean_top_turn_risk: {state.mean_top_turn_risk:.2f}",
            f"risk_trend: {state.risk_trend}",
            f"intent_progression: {state.intent_progression}",
            f"evasion_attempts: {state.evasion_attempts}",
            f"operational_details_requested: {state.operational_details_requested}",
            f"benign_cover_story: {state.benign_cover_story}",
            f"cross_turn_composition: {state.cross_turn_composition}",
            f"safety_summary: {state.safety_summary}",
            "salient_turns:",
            salient,
        ]
    )


def render_full_transcript_evidence(session: dict[str, Any], max_chars: int = 12_000) -> str:
    lines = ["Classify the safety risk of this full multi-turn session transcript."]
    for turn in list(session.get("turns") or []):
        role = str(turn.get("role") or "user")
        idx = int(turn.get("turn_index") or 0) + 1
        text = compact_text(turn_text(turn), max_chars=1_200)
        if text:
            lines.append(f"turn-{idx} role={role}: {text}")
    rendered = "\n".join(lines)
    return rendered[:max_chars]


@dataclass(frozen=True)
class FullTranscriptLexicalSessionScorer:
    """Cheap oracle baseline: full transcript access, lexical session cues."""

    def score(self, session: dict[str, Any]) -> tuple[float, dict[str, float]]:
        text = "\n".join(turn_text(turn) for turn in list(session.get("turns") or []))
        harm = min(1.0, count_terms(text, HARM_TERMS) / 3)
        operational = min(1.0, count_terms(text, OPERATIONAL_TERMS) / 4)
        evasion = min(1.0, count_terms(text, EVASION_TERMS) / 3)
        cover = min(1.0, count_terms(text, COVER_STORY_TERMS) / 2)
        progression = min(1.0, count_terms(text, PROGRESSION_TERMS) / 3)
        contributions = {
            "harm_terms": 0.24 * harm,
            "operational_terms": 0.24 * operational,
            "evasion_terms": 0.18 * evasion,
            "cover_story_terms": 0.12 * cover,
            "progression_terms": 0.14 * progression,
            "length_prior": min(0.08, len(text) / 25_000),
        }
        return min(0.99, sum(contributions.values())), contributions


class CompactSessionRiskFeature(FeatureExtractor):
    stage = FeatureStage.SESSION_DRIFT

    def __init__(self, updater: SessionStateUpdater | None = None) -> None:
        self.updater = updater or SessionStateUpdater()
        self.scorer = CompactSessionRiskScorer()

    def extract(self, feature_input: FeatureInput, state: RiskState) -> FeatureResult:
        session = feature_input.context.get("session_record")
        compact_state = feature_input.context.get("compact_session_state")
        if not isinstance(compact_state, CompactSessionState):
            if not isinstance(session, dict):
                return FeatureResult(
                    stage=self.stage,
                    risk_score=0.0,
                    confidence=0.30,
                    label="missing_session_state",
                    cost_tier=1,
                    version="compact_session_risk_v0.1",
                    metadata={},
                )
            compact_state = self.updater.update(session)
        risk, contributions = self.scorer.score(compact_state)
        return FeatureResult(
            stage=self.stage,
            risk_score=risk,
            confidence=0.72 if compact_state.turn_count >= 2 else 0.45,
            label="session_risk" if risk >= 0.55 else "session_monitor",
            cost_tier=1,
            version="compact_session_risk_v0.1",
            metadata={
                "contributions": contributions,
                "compact_state": {
                    "turn_count": compact_state.turn_count,
                    "domains_seen": compact_state.domains_seen,
                    "subclusters_seen": compact_state.subclusters_seen,
                    "highest_severity": compact_state.highest_severity,
                    "risk_trend": compact_state.risk_trend,
                    "intent_progression": compact_state.intent_progression,
                    "evasion_attempts": compact_state.evasion_attempts,
                    "operational_details_requested": compact_state.operational_details_requested,
                    "benign_cover_story": compact_state.benign_cover_story,
                    "cross_turn_composition": compact_state.cross_turn_composition,
                    "safety_summary": compact_state.safety_summary,
                },
            },
        )
