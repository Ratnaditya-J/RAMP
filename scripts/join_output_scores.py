#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

OUTPUT_FIELDS = (
    "output_risk_score",
    "output_confidence",
    "output_label",
    "output_harm_category",
    "output_classifier_version",
    "output_classifier_metadata",
    "output_reviewed_label",
    "output_text",
    "eval_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join output classifier scores into an existing feature table."
    )
    parser.add_argument("--feature-table", required=True)
    parser.add_argument("--output-scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument(
        "--require-output-score",
        action="store_true",
        help="Only write rows that have a matching output score.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_id(row: dict[str, Any]) -> str | None:
    value = row.get("source_id") or row.get("id")
    return str(value) if value is not None else None


def load_scores(path: Path) -> dict[str, dict[str, Any]]:
    scores = {}
    for row in load_jsonl(path):
        key = row_id(row)
        if key is not None:
            scores[key] = row
    return scores


def join_row(row: dict[str, Any], score_row: dict[str, Any]) -> dict[str, Any]:
    joined = dict(row)
    for field in OUTPUT_FIELDS:
        if field in score_row:
            output_field = "output_eval_id" if field == "eval_id" else field
            joined[output_field] = score_row[field]
    return joined


def main() -> None:
    args = parse_args()
    feature_rows = load_jsonl(Path(args.feature_table))
    score_rows = load_scores(Path(args.output_scores))

    joined_rows = []
    matched = 0
    for row in feature_rows:
        key = row_id(row)
        score_row = score_rows.get(key or "")
        if score_row is None:
            if not args.require_output_score:
                joined_rows.append(row)
            continue
        matched += 1
        joined_rows.append(join_row(row, score_row))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in joined_rows:
            output_file.write(json.dumps(row, separators=(",", ":")) + "\n")

    summary = {
        "artifact_id": "ramp_output_score_join_v0.1",
        "feature_table": args.feature_table,
        "output_scores": args.output_scores,
        "output": args.output,
        "feature_rows": len(feature_rows),
        "output_score_rows": len(score_rows),
        "matched_rows": matched,
        "written_rows": len(joined_rows),
        "require_output_score": args.require_output_score,
        "missing_output_score_rows": len(feature_rows) - matched,
    }
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote joined feature table to {args.output}; "
        f"matched={matched}; written={len(joined_rows)}"
    )


if __name__ == "__main__":
    main()
