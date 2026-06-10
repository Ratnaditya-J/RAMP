#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ramp.features.session_state_risk import (
    SessionStateUpdater,
    render_compact_session_evidence,
    render_full_transcript_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build session-level classifier inputs from compact state or full transcript."
    )
    parser.add_argument("--session-corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=["compact_state", "full_transcript"],
        default="compact_state",
    )
    parser.add_argument("--max-full-transcript-chars", type=int, default=12_000)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def row_for_session(
    session: dict[str, Any],
    *,
    mode: str,
    updater: SessionStateUpdater,
    max_full_transcript_chars: int,
) -> dict[str, Any]:
    session_id = str(session.get("session_id") or "")
    if mode == "compact_state":
        compact_state = updater.update(session)
        text = render_compact_session_evidence(compact_state)
        metadata = {
            "compact_turn_count": compact_state.turn_count,
            "compact_evidence_chars": len(text),
            "salient_turn_count": len(compact_state.salient_turns),
            "risk_trend": compact_state.risk_trend,
            "intent_progression": compact_state.intent_progression,
            "evasion_attempts": compact_state.evasion_attempts,
            "operational_details_requested": compact_state.operational_details_requested,
            "cross_turn_composition": compact_state.cross_turn_composition,
        }
    else:
        text = render_full_transcript_evidence(session, max_chars=max_full_transcript_chars)
        metadata = {
            "full_transcript_chars": len(text),
            "max_full_transcript_chars": max_full_transcript_chars,
        }
    return {
        "id": f"{session_id}.{mode}",
        "session_id": session_id,
        "source": session.get("source"),
        "source_record_id": session.get("source_record_id"),
        "session_label": session.get("session_label"),
        "session_risk_type": session.get("session_risk_type"),
        "label": session.get("session_label"),
        "domain": ",".join(session.get("harm_domains") or []),
        "subcluster_id": session.get("session_risk_type"),
        "span_text": text,
        "representation_mode": mode,
        **metadata,
    }


def main() -> None:
    args = parse_args()
    updater = SessionStateUpdater()
    rows = [
        row_for_session(
            session,
            mode=args.mode,
            updater=updater,
            max_full_transcript_chars=args.max_full_transcript_chars,
        )
        for session in load_jsonl(Path(args.session_corpus))
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"wrote {len(rows)} {args.mode} session classifier rows to {args.output}")


if __name__ == "__main__":
    main()
