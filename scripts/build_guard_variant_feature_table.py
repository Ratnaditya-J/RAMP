#!/usr/bin/env python
"""Build a feature-table variant whose prompt signal is an external guard's score.

Swaps `prompt_risk_score` for the external guard's score (keeping the original as
`prompt_risk_score_baseline`), leaving embedding_prior_score and activation_probability
unchanged. Running the survival ladder on the variant answers: once the front-door
classifier is a strong external guard, do the internal signals still earn weight?

Only rows that have a usable guard score are emitted (the evaluation set); the ladder uses
matched rows only, so this keeps the variant table small.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Splice external-guard scores into a feature table."
    )
    parser.add_argument("--feature-table", required=True)
    parser.add_argument(
        "--guard-scores", required=True, help="Guard score JSONL (source_id, guard_score)."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    return parser.parse_args()


def load_guard_scores(path: Path) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("guard_status") != "scored":
                continue
            if record.get("guard_score") is None:
                continue
            source_id = str(record.get("source_id", "")).strip()
            if source_id:
                scores[source_id] = record
    return scores


def build_variant(
    feature_rows: list[dict[str, Any]],
    guard_scores: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    matched = 0
    for row in feature_rows:
        row_id = str(row.get("id", ""))
        score_record = guard_scores.get(row_id)
        if score_record is None:
            continue
        if score_record.get("guard_status") != "scored" or score_record.get("guard_score") is None:
            continue
        if row.get("prompt_risk_score") is None:
            continue
        variant = dict(row)
        variant["prompt_risk_score_baseline"] = row.get("prompt_risk_score")
        variant["prompt_risk_score"] = float(score_record["guard_score"])
        variant["prompt_signal_source"] = score_record.get("guard_model")
        output.append(variant)
        matched += 1
    summary = {
        "feature_rows": len(feature_rows),
        "guard_scored_ids": len(guard_scores),
        "variant_rows": matched,
    }
    return output, summary


def main() -> None:
    args = parse_args()
    feature_rows = [
        json.loads(line)
        for line in Path(args.feature_table).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    guard_scores = load_guard_scores(Path(args.guard_scores))
    variant, summary = build_variant(feature_rows, guard_scores)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, sort_keys=True) for row in variant) + "\n")
    print(f"wrote {summary['variant_rows']} variant rows to {output}")
    if args.summary_output:
        Path(args.summary_output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
