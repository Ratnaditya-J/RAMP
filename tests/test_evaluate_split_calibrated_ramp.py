from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_evaluate_split_calibrated_ramp_reports_holdout_metrics(tmp_path: Path) -> None:
    review_csv = tmp_path / "review.csv"
    feature_table = tmp_path / "features.jsonl"
    output_json = tmp_path / "split.json"
    output_md = tmp_path / "split.md"

    review_rows = []
    feature_rows = []
    for idx in range(12):
        label = "unsafe" if idx % 2 else "safe"
        row_id = f"row-{idx:02d}"
        review_rows.append(
            {
                "review_id": f"r{idx}",
                "source_id": row_id,
                "review_status": "reviewed",
                "reviewed_label": label,
                "prompt_text": row_id,
                "corpus_label": label,
                "qwen_label": "Unsafe" if label == "unsafe" else "Safe",
            }
        )
        feature_rows.append(
            {
                "id": row_id,
                "label": label,
                "source": "synthetic",
                "domain": "cyber_abuse",
                "subcluster_role": (
                    "harmful" if label == "unsafe" else "benign_near_neighbor"
                ),
                "subcluster_id": (
                    "vulnerability_exploitation"
                    if label == "unsafe"
                    else "defensive_security"
                ),
                "span_text": row_id,
                "prompt_risk_score": 0.85 if label == "unsafe" else 0.15,
                "prompt_label": "Unsafe" if label == "unsafe" else "Safe",
                "prompt_harm_category": None,
                "embedding_prior_score": 0.80 if label == "unsafe" else 0.10,
                "activation_probability": 0.80 if label == "unsafe" else 0.10,
                "prompt_embedding_score": 0.80 if label == "unsafe" else 0.10,
                "prompt_activation_score": 0.80 if label == "unsafe" else 0.10,
                "cumulative_full_score": 0.80 if label == "unsafe" else 0.10,
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
            "scripts/evaluate_split_calibrated_ramp.py",
            "--review-csv",
            str(review_csv),
            "--feature-table",
            str(feature_table),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--calibration-fraction",
            "0.5",
            "--weight-step",
            "0.10",
            "--threshold-step",
            "0.10",
            "--min-prompt-weight",
            "0.40",
            "--max-embedding-weight",
            "0.20",
            "--require-prompt-gte-activation",
            "--require-activation-gte-embedding",
        ],
        check=True,
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))

    assert report["num_binary_eval_rows"] == 12
    assert report["split"]["calibration_rows"] == 6
    assert report["split"]["holdout_rows"] == 6
    assert report["calibration_protocol"]["selected_config"]["threshold"] >= 0
    assert "ramp_fusion" in report["holdout_eval"]["ablations"]
    assert report["holdout_eval"]["ablations"]["ramp_fusion"]["metrics"]["recall"] == 1.0
    assert "Split-Calibrated RAMP Evaluation" in output_md.read_text(encoding="utf-8")
