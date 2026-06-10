from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_evaluate_severity_floors_reports_fn_and_fp_delta(tmp_path: Path) -> None:
    review_csv = tmp_path / "review.csv"
    feature_table = tmp_path / "features.jsonl"
    calibration_artifact = tmp_path / "calibration.json"
    output_json = tmp_path / "severity_floor.json"
    output_md = tmp_path / "severity_floor.md"

    review_rows = [
        {
            "review_id": "r1",
            "source_id": "unsafe-high",
            "review_status": "reviewed",
            "reviewed_label": "unsafe",
            "prompt_text": "unsafe high severity",
            "corpus_label": "unsafe",
            "qwen_label": "Unsafe",
        },
        {
            "review_id": "r2",
            "source_id": "safe-high",
            "review_status": "reviewed",
            "reviewed_label": "safe",
            "prompt_text": "safe but high severity classifier category",
            "corpus_label": "safe",
            "qwen_label": "Unsafe",
        },
        {
            "review_id": "r3",
            "source_id": "safe-low",
            "review_status": "reviewed",
            "reviewed_label": "safe",
            "prompt_text": "safe low risk",
            "corpus_label": "safe",
            "qwen_label": "Safe",
        },
    ]
    with review_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)

    feature_rows = [
        {
            "id": "unsafe-high",
            "prompt_risk_score": 0.92,
            "prompt_label": "Unsafe",
            "prompt_harm_category": "PII",
            "embedding_prior_score": 0.05,
            "activation_probability": 0.05,
        },
        {
            "id": "safe-high",
            "prompt_risk_score": 0.92,
            "prompt_label": "Unsafe",
            "prompt_harm_category": "PII",
            "embedding_prior_score": 0.05,
            "activation_probability": 0.05,
        },
        {
            "id": "safe-low",
            "prompt_risk_score": 0.08,
            "prompt_label": "Safe",
            "prompt_harm_category": None,
            "embedding_prior_score": 0.05,
            "activation_probability": 0.05,
        },
    ]
    feature_table.write_text(
        "\n".join(json.dumps(row) for row in feature_rows) + "\n",
        encoding="utf-8",
    )
    calibration_artifact.write_text(
        json.dumps(
            {
                "artifact_id": "test_calibration",
                "selected_by_best_f1": {
                    "prompt_weight": 0.5,
                    "embedding_weight": 0.1,
                    "activation_weight": 0.4,
                    "threshold": 0.72,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_severity_floors.py",
            "--review-csv",
            str(review_csv),
            "--feature-table",
            str(feature_table),
            "--calibration-artifact",
            str(calibration_artifact),
            "--calibration-objective",
            "selected_by_best_f1",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--floor",
            "0.72",
        ],
        check=True,
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))

    assert report["num_binary_eval_rows"] == 3
    assert report["severity_floor"]["candidate_rows"] == 2
    assert report["severity_floor"]["applied_rows"] == 2
    assert report["raw_calibrated"]["metrics"]["fn"] == 1
    assert report["raw_calibrated"]["metrics"]["fp"] == 0
    assert report["severity_floor_calibrated"]["metrics"]["fn"] == 0
    assert report["severity_floor_calibrated"]["metrics"]["fp"] == 1
    assert report["delta"]["false_negatives_fixed"] == ["unsafe-high"]
    assert report["delta"]["new_false_positives"] == ["safe-high"]
    assert "Severity Floor Evaluation" in output_md.read_text(encoding="utf-8")
