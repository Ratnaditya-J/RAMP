#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join turn-level classifier scores back into a session corpus."
    )
    parser.add_argument("--session-corpus", required=True)
    parser.add_argument("--turn-scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-key", default="prompt_risk_score")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def main() -> None:
    args = parse_args()
    sessions = load_jsonl(Path(args.session_corpus))
    scores = {str(row.get("id")): row for row in load_jsonl(Path(args.turn_scores))}
    matched = 0
    for session in sessions:
        for turn in session.get("turns", []):
            turn_id = str(turn.get("turn_id"))
            score_row = scores.get(turn_id)
            if score_row is None:
                continue
            if args.score_key in score_row:
                turn["risk_score"] = float(score_row[args.score_key])
                turn["score_source"] = args.score_key
                turn["classifier_label"] = score_row.get("prompt_label") or score_row.get(
                    "output_label"
                )
                turn["classifier_version"] = score_row.get(
                    "prompt_classifier_version"
                ) or score_row.get("output_classifier_version")
                matched += 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as output_file:
        for session in sessions:
            output_file.write(json.dumps(session, separators=(",", ":")) + "\n")
    print(f"wrote scored sessions to {args.output}; matched_turn_scores={matched}")


if __name__ == "__main__":
    main()
