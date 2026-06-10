#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_session_risk import session_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine high-risk or missed sessions from a scored session corpus."
    )
    parser.add_argument("--session-corpus", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument(
        "--mode",
        choices=["top_session", "single_turn_misses", "all"],
        default="all",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def compact_turn(turn: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_id": turn.get("turn_id"),
        "turn_index": turn.get("turn_index"),
        "role": turn.get("role"),
        "risk_score": turn.get("risk_score"),
        "classifier_label": turn.get("classifier_label"),
        "text": str(turn.get("text") or "")[:500],
    }


def candidate_row(session: dict[str, Any], threshold: float) -> dict[str, Any]:
    scored = dict(session)
    scored.update(session_score(session))
    turns = list(scored.get("turns") or [])
    top_turns = sorted(turns, key=lambda turn: float(turn.get("risk_score") or 0.0), reverse=True)
    return {
        "session_id": scored.get("session_id"),
        "source": scored.get("source"),
        "session_label": scored.get("session_label"),
        "session_risk_type": scored.get("session_risk_type"),
        "turn_count": scored.get("turn_count"),
        "single_turn_max_score": scored["single_turn_max_score"],
        "session_risk_score": scored["session_risk_score"],
        "single_turn_predicted_unsafe": scored["single_turn_max_score"] >= threshold,
        "session_predicted_unsafe": scored["session_risk_score"] >= threshold,
        "harm_domains": scored.get("harm_domains"),
        "reviewer_notes": scored.get("reviewer_notes"),
        "top_turns": [compact_turn(turn) for turn in top_turns[:3]],
    }


def select_candidates(
    sessions: list[dict[str, Any]],
    mode: str,
    limit: int,
    threshold: float,
) -> list[dict[str, Any]]:
    rows = [candidate_row(session, threshold) for session in sessions]
    if mode == "single_turn_misses":
        rows = [
            row
            for row in rows
            if row["session_label"] == "unsafe" and not row["single_turn_predicted_unsafe"]
        ]
        return sorted(rows, key=lambda row: row["session_risk_score"], reverse=True)[:limit]
    if mode == "top_session":
        return sorted(rows, key=lambda row: row["session_risk_score"], reverse=True)[:limit]
    return sorted(
        rows,
        key=lambda row: (
            row["session_risk_score"],
            row["single_turn_max_score"],
            row.get("turn_count") or 0,
        ),
        reverse=True,
    )[:limit]


def write_markdown(path: Path, rows: list[dict[str, Any]], mode: str) -> None:
    lines = [
        f"# Session Risk Candidate Mining: {mode}",
        "",
        "| Rank | Session | Label | Single Max | Session Score | Top Turn |",
        "| ---: | --- | --- | ---: | ---: | --- |",
    ]
    for idx, row in enumerate(rows, start=1):
        top_turn = row["top_turns"][0] if row["top_turns"] else {}
        text = str(top_turn.get("text") or "").replace("|", "\\|")
        lines.append(
            "| {} | `{}` | {} | {:.4f} | {:.4f} | {} |".format(
                idx,
                row["session_id"],
                row["session_label"],
                row["single_turn_max_score"],
                row["session_risk_score"],
                text[:180],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = select_candidates(
        load_jsonl(Path(args.session_corpus)),
        args.mode,
        args.limit,
        args.threshold,
    )
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, separators=(",", ":")) + "\n")
    if args.output_md:
        write_markdown(Path(args.output_md), rows, args.mode)
    print(f"wrote {len(rows)} session candidates to {args.output_jsonl}")
    if args.output_md:
        print(f"wrote session candidate markdown to {args.output_md}")


if __name__ == "__main__":
    main()
