from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_clean_review_csv_promotes_labeled_rows(tmp_path: Path) -> None:
    input_csv = tmp_path / "review.csv"
    output_csv = tmp_path / "clean.csv"
    summary_json = tmp_path / "summary.json"
    with input_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "review_id",
                "source_id",
                "review_status",
                "reviewed_label",
                "label_issue_type",
                "reviewed_by",
                "reviewed_at",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "review_id": "r1",
                "source_id": "s1",
                "review_status": "unreviewed",
                "reviewed_label": "Unsafe",
                "label_issue_type": "",
                "reviewed_by": "",
                "reviewed_at": "",
            }
        )

    subprocess.run(
        [
            sys.executable,
            "scripts/clean_review_csv.py",
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
            "--summary-output",
            str(summary_json),
            "--reviewed-by",
            "tester",
            "--reviewed-at",
            "2026-06-08",
        ],
        check=True,
    )

    with output_csv.open(newline="", encoding="utf-8") as input_file:
        row = next(csv.DictReader(input_file))
    assert row["review_status"] == "reviewed"
    assert row["reviewed_label"] == "unsafe"
    assert row["label_issue_type"] == "none"
    assert row["reviewed_by"] == "tester"
    assert row["reviewed_at"] == "2026-06-08"
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["fix_counts"]["promoted_status_to_reviewed"] == 1


def test_combine_review_csvs_prefers_later_reviewed_rows(tmp_path: Path) -> None:
    first_csv = tmp_path / "first.csv"
    second_csv = tmp_path / "second.csv"
    output_csv = tmp_path / "combined.csv"
    for path, label, status in [
        (first_csv, "", "unreviewed"),
        (second_csv, "safe", "reviewed"),
    ]:
        with path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=["review_id", "source_id", "review_status", "reviewed_label"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "review_id": path.stem,
                    "source_id": "same-source",
                    "review_status": status,
                    "reviewed_label": label,
                }
            )

    subprocess.run(
        [
            sys.executable,
            "scripts/combine_review_csvs.py",
            "--input",
            str(first_csv),
            "--input",
            str(second_csv),
            "--output",
            str(output_csv),
        ],
        check=True,
    )

    with output_csv.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    assert len(rows) == 1
    assert rows[0]["review_id"] == "second"
    assert rows[0]["reviewed_label"] == "safe"
