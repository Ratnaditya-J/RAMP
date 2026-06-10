#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten session corpus turns into classifier-scoring rows."
    )
    parser.add_argument("--session-corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--roles",
        default="user,agent,assistant",
        help="Comma-separated roles to include.",
    )
    return parser.parse_args()


def load_sessions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def main() -> None:
    args = parse_args()
    roles = {role.strip().lower() for role in args.roles.split(",") if role.strip()}
    rows = []
    for session in load_sessions(Path(args.session_corpus)):
        for turn in session.get("turns", []):
            role = str(turn.get("role") or "user").lower()
            if role not in roles:
                continue
            turn_id = str(turn.get("turn_id"))
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            rows.append(
                {
                    "id": turn_id,
                    "session_id": session.get("session_id"),
                    "source": session.get("source"),
                    "source_record_id": session.get("source_record_id"),
                    "session_label": session.get("session_label"),
                    "session_risk_type": session.get("session_risk_type"),
                    "turn_id": turn_id,
                    "turn_index": turn.get("turn_index"),
                    "role": role,
                    "span_text": text,
                    "label": turn.get("turn_label"),
                    "domain": turn.get("harm_domain"),
                    "subcluster_id": turn.get("subcluster_id"),
                    "harm_severity": turn.get("harm_severity"),
                }
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"wrote {len(rows)} turn rows to {args.output}")


if __name__ == "__main__":
    main()
