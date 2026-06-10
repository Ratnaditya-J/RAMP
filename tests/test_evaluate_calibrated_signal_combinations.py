from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_evaluate_calibrated_signal_combinations_outputs_all_conditions(tmp_path: Path) -> None:
    review_csv = tmp_path / "review.csv"
    feature_table = tmp_path / "features.jsonl"
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"

    review_rows = []
    feature_rows = []
    for idx in range(20):
        unsafe = idx % 2 == 1
        label = "unsafe" if unsafe else "safe"
        row_id = f"row-{idx:02d}"
        review_rows.append(
            {
                "review_id": f"r{idx}",
                "source_id": row_id,
                "review_status": "reviewed",
                "reviewed_label": label,
                "prompt_text": row_id,
            }
        )
        feature_rows.append(
            {
                "id": row_id,
                "label": label,
                "source": "synthetic",
                "domain": "cyber_abuse",
                "subcluster_role": "harmful" if unsafe else "benign_near_neighbor",
                "subcluster_id": "vulnerability_exploitation",
                "span_text": row_id,
                "prompt_risk_score": 0.8 if unsafe else 0.2,
                "embedding_prior_score": 0.75 if unsafe else 0.25,
                "activation_probability": 0.7 if unsafe else 0.3,
            }
        )
    with review_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    feature_table.write_text(
        "\n".join(json.dumps(row) for row in feature_rows) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_calibrated_signal_combinations.py",
            "--review-csv",
            str(review_csv),
            "--feature-table",
            str(feature_table),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--num-splits",
            "3",
            "--weight-step",
            "0.10",
            "--threshold-step",
            "0.10",
        ],
        check=True,
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["num_splits"] == 3
    assert set(report["aggregate_holdout_metrics"]) == {
        "prompt_only_calibrated",
        "prompt_embedding_calibrated",
        "prompt_activation_calibrated",
        "prompt_embedding_activation_calibrated",
    }
    assert "Calibrated Signal Combination Stability" in output_md.read_text(
        encoding="utf-8"
    )


def test_evaluate_calibrated_signal_combinations_can_include_output(
    tmp_path: Path,
) -> None:
    review_csv = tmp_path / "review.csv"
    feature_table = tmp_path / "features.jsonl"
    output_json = tmp_path / "report.json"

    review_rows = []
    feature_rows = []
    for idx in range(20):
        unsafe = idx % 2 == 1
        label = "unsafe" if unsafe else "safe"
        row_id = f"row-output-{idx:02d}"
        review_rows.append(
            {
                "review_id": f"r{idx}",
                "source_id": row_id,
                "review_status": "reviewed",
                "reviewed_label": label,
                "prompt_text": row_id,
            }
        )
        feature_rows.append(
            {
                "id": row_id,
                "label": label,
                "source": "synthetic",
                "domain": "cyber_abuse",
                "subcluster_role": "harmful" if unsafe else "benign_near_neighbor",
                "subcluster_id": "vulnerability_exploitation",
                "span_text": row_id,
                "prompt_risk_score": 0.6 if unsafe else 0.4,
                "embedding_prior_score": 0.55 if unsafe else 0.45,
                "activation_probability": 0.7 if unsafe else 0.3,
                "output_risk_score": 0.9 if unsafe else 0.1,
            }
        )
    with review_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    feature_table.write_text(
        "\n".join(json.dumps(row) for row in feature_rows) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_calibrated_signal_combinations.py",
            "--review-csv",
            str(review_csv),
            "--feature-table",
            str(feature_table),
            "--output-json",
            str(output_json),
            "--num-splits",
            "3",
            "--weight-step",
            "0.10",
            "--threshold-step",
            "0.10",
            "--include-output",
        ],
        check=True,
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["include_output"] is True
    assert "prompt_output_calibrated" in report["aggregate_holdout_metrics"]
    assert "prompt_activation_output_calibrated" in report["aggregate_holdout_metrics"]
    assert "prompt_embedding_activation_output_calibrated" in report[
        "aggregate_holdout_metrics"
    ]
