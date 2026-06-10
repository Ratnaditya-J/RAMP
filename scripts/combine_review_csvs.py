#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine review CSVs by source_id.")
    parser.add_argument("--input", action="append", required=True, help="Input CSV. Repeatable.")
    parser.add_argument("--output", required=True, help="Combined output CSV.")
    parser.add_argument("--summary-output", default=None, help="Optional summary JSON.")
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        return list(reader.fieldnames or []), list(reader)


def row_precedence(row: dict[str, str], input_index: int, row_index: int) -> tuple[int, int, int]:
    status = str(row.get("review_status", "")).strip().lower()
    label = str(row.get("reviewed_label", "")).strip().lower()
    reviewed_rank = 1 if status == "reviewed" and label else 0
    return reviewed_rank, input_index, row_index


def main() -> None:
    args = parse_args()
    fieldnames: list[str] = []
    selected: dict[str, tuple[tuple[int, int, int], dict[str, str]]] = {}
    total_rows = 0
    for input_index, input_path in enumerate(args.input):
        path = Path(input_path)
        fields, rows = read_rows(path)
        if not fieldnames:
            fieldnames = fields
        else:
            for field in fields:
                if field not in fieldnames:
                    fieldnames.append(field)
        for row_index, row in enumerate(rows):
            total_rows += 1
            source_id = str(row.get("source_id", "")).strip()
            if not source_id:
                source_id = f"__missing_source_id__:{input_index}:{row_index}"
            precedence = row_precedence(row, input_index, row_index)
            previous = selected.get(source_id)
            if previous is None or precedence >= previous[0]:
                selected[source_id] = (precedence, dict(row))

    output_rows = [
        row
        for _, row in sorted(
            selected.values(),
            key=lambda item: item[1].get("review_id", ""),
        )
    ]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    status_counts = Counter(row.get("review_status", "") for row in output_rows)
    label_counts = Counter(row.get("reviewed_label", "") for row in output_rows)
    summary = {
        "artifact_id": "ramp_combined_review_csv_v0.1",
        "inputs": args.input,
        "output": str(output_path),
        "input_rows": total_rows,
        "output_rows": len(output_rows),
        "deduplicated_rows": total_rows - len(output_rows),
        "status_counts": dict(status_counts),
        "reviewed_label_counts": dict(label_counts),
    }
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote combined review CSV to {output_path}")
    print(f"input_rows={total_rows} output_rows={len(output_rows)}")
    print(f"status_counts={dict(status_counts)}")
    print(f"reviewed_label_counts={dict(label_counts)}")


if __name__ == "__main__":
    main()
