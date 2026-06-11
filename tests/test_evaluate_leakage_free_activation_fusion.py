from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_evaluate_leakage_free_activation_fusion_trains_probe_per_split(
    tmp_path: Path,
) -> None:
    review_csv = tmp_path / "review.csv"
    feature_table = tmp_path / "features.jsonl"
    activation = tmp_path / "activation.jsonl"
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"

    review_rows = []
    feature_rows = []
    activation_rows = []
    for idx in range(24):
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
                "prompt_risk_score": 0.65 if unsafe else 0.35,
                "embedding_prior_score": 0.60 if unsafe else 0.40,
                "activation_probability": 0.99 if unsafe else 0.01,
            }
        )
        activation_rows.append(
            {
                "id": row_id,
                "span_text": row_id,
                "domain": "cyber_abuse",
                "subcluster_role": "harmful" if unsafe else "benign_near_neighbor",
                "subcluster_id": "vulnerability_exploitation",
                "label": label,
                "source": "synthetic",
                "embedding": [1.0, 0.0] if unsafe else [0.0, 1.0],
                "provenance": {"huggingface_model_id": "synthetic"},
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
    activation.write_text(
        "\n".join(json.dumps(row) for row in activation_rows) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_leakage_free_activation_fusion.py",
            "--review-csv",
            str(review_csv),
            "--feature-table",
            str(feature_table),
            "--activation",
            str(activation),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--num-splits",
            "2",
            "--weight-step",
            "0.50",
            "--threshold-step",
            "0.50",
            "--epochs",
            "3",
        ],
        check=True,
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["artifact_id"] == "ramp_leakage_free_activation_fusion_v0.1"
    assert report["num_splits"] == 2
    assert report["num_binary_eval_rows"] == 24
    assert "out-of-split predictions" in report["activation_probe_protocol"]
    assert "activation_probe" in report["split_reports"][0]
    assert "prompt_activation_calibrated" in report["aggregate_holdout_metrics"]
    assert "Calibrated Signal Combination Stability" in output_md.read_text(
        encoding="utf-8"
    )
