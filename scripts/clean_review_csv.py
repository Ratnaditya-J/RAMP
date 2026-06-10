#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import UTC, date
from pathlib import Path
from typing import Any

VALID_REVIEW_STATUS = {"unreviewed", "reviewed", "skip"}
VALID_REVIEWED_LABELS = {
    "",
    "safe",
    "unsafe",
    "controversial",
    "ambiguous_or_context_needed",
    "bad_benchmark_label",
}
VALID_ISSUE_TYPES = {
    "",
    "none",
    "benchmark_label_wrong",
    "taxonomy_wrong",
    "prompt_classifier_wrong",
    "ambiguous_policy_boundary",
    "insufficient_context",
    "duplicate_or_low_quality",
    "other",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean mechanical review CSV edits.")
    parser.add_argument("--input", required=True, help="Input review CSV.")
    parser.add_argument("--output", required=True, help="Cleaned output CSV.")
    parser.add_argument("--summary-output", default=None, help="Optional summary JSON.")
    parser.add_argument("--reviewed-by", default="ratnaditya")
    parser.add_argument(
        "--reviewed-at",
        default=date.today().isoformat(),
        help="Date to fill for reviewed rows with blank reviewed_at.",
    )
    return parser.parse_args()


def normalized(value: Any) -> str:
    return str(value or "").strip().lower()


def normalize_label_issue(value: Any) -> str:
    value = normalized(value)
    aliases = {
        "na": "none",
        "n/a": "none",
        "no issue": "none",
        "no_issue": "none",
        "benchmark wrong": "benchmark_label_wrong",
        "benchmark_label": "benchmark_label_wrong",
        "label_wrong": "benchmark_label_wrong",
        "taxonomy wrong": "taxonomy_wrong",
        "classifier_wrong": "prompt_classifier_wrong",
        "prompt wrong": "prompt_classifier_wrong",
        "ambiguous": "ambiguous_policy_boundary",
        "needs_context": "insufficient_context",
        "duplicate": "duplicate_or_low_quality",
    }
    return aliases.get(value, value)


def clean_row(
    row: dict[str, str],
    *,
    reviewed_by: str,
    reviewed_at: str,
) -> tuple[dict[str, str], list[str]]:
    row = dict(row)
    fixes = []
    label = normalized(row.get("reviewed_label"))
    status = normalized(row.get("review_status"))
    issue = normalize_label_issue(row.get("label_issue_type"))

    if label not in VALID_REVIEWED_LABELS:
        fixes.append(f"invalid_reviewed_label:{label}")
    if status not in VALID_REVIEW_STATUS:
        fixes.append(f"invalid_review_status:{status}")
    if issue not in VALID_ISSUE_TYPES:
        fixes.append(f"invalid_label_issue_type:{issue}")

    if label and label != "bad_benchmark_label" and status == "unreviewed":
        status = "reviewed"
        fixes.append("promoted_status_to_reviewed")
    if label == "bad_benchmark_label" and status == "unreviewed":
        status = "reviewed"
        fixes.append("promoted_bad_benchmark_label_to_reviewed")
    if status == "reviewed" and not label:
        fixes.append("reviewed_status_missing_label")
    if status == "reviewed" and not issue:
        issue = "none"
        fixes.append("filled_label_issue_type_none")
    if status == "reviewed" and "reviewed_by" in row and not row.get("reviewed_by", "").strip():
        row["reviewed_by"] = reviewed_by
        fixes.append("filled_reviewed_by")
    if status == "reviewed" and "reviewed_at" in row and not row.get("reviewed_at", "").strip():
        row["reviewed_at"] = reviewed_at
        fixes.append("filled_reviewed_at")

    row["review_status"] = status
    row["reviewed_label"] = label
    row["label_issue_type"] = issue
    return row, fixes


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    with input_path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise ValueError("input CSV has no header")

    cleaned_rows = []
    fix_counts: Counter[str] = Counter()
    row_fix_counts = 0
    invalid_rows = []
    for index, row in enumerate(rows, start=2):
        cleaned, fixes = clean_row(
            row,
            reviewed_by=args.reviewed_by,
            reviewed_at=args.reviewed_at,
        )
        if fixes:
            row_fix_counts += 1
            fix_counts.update(fixes)
        if any(fix.startswith("invalid_") or fix.endswith("_missing_label") for fix in fixes):
            invalid_rows.append(
                {
                    "line": index,
                    "review_id": row.get("review_id"),
                    "source_id": row.get("source_id"),
                    "fixes": fixes,
                }
            )
        cleaned_rows.append(cleaned)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    status_counts = Counter(row.get("review_status", "") for row in cleaned_rows)
    label_counts = Counter(row.get("reviewed_label", "") for row in cleaned_rows)
    issue_counts = Counter(row.get("label_issue_type", "") for row in cleaned_rows)
    summary = {
        "artifact_id": "ramp_review_csv_cleaning_v0.1",
        "input": str(input_path),
        "output": str(output_path),
        "generated_at": date.today().isoformat(),
        "generated_timezone": "UTC" if date.today() == date.today() else str(UTC),
        "num_rows": len(cleaned_rows),
        "rows_with_fixes": row_fix_counts,
        "fix_counts": dict(fix_counts),
        "status_counts": dict(status_counts),
        "reviewed_label_counts": dict(label_counts),
        "label_issue_type_counts": dict(issue_counts),
        "invalid_rows": invalid_rows,
    }
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote cleaned review CSV to {output_path}")
    print(f"rows={len(cleaned_rows)} rows_with_fixes={row_fix_counts}")
    print(f"status_counts={dict(status_counts)}")
    print(f"reviewed_label_counts={dict(label_counts)}")
    if invalid_rows:
        print(f"invalid_rows={len(invalid_rows)}")


if __name__ == "__main__":
    main()
