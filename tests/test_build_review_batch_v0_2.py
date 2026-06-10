from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_build_review_batch_v0_2_targets_harness_gaps(tmp_path: Path) -> None:
    feature_table = tmp_path / "features.jsonl"
    review_csv = tmp_path / "review.csv"
    calibration = tmp_path / "calibration.json"
    output_jsonl = tmp_path / "batch.jsonl"
    output_csv = tmp_path / "batch.csv"
    summary = tmp_path / "summary.json"

    review_csv.write_text(
        "\n".join(
            [
                "review_id,source_id,review_status,reviewed_label",
                "old-1,already-reviewed,reviewed,safe",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    calibration.write_text(
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
    rows = [
        {
            "id": "already-reviewed",
            "label": "unsafe",
            "source": "synthetic",
            "domain": "privacy_identity_and_secrets",
            "subcluster_role": "harmful",
            "subcluster_id": "pii_extraction",
            "span_text": "already reviewed",
            "prompt_risk_score": 0.92,
            "prompt_confidence": 0.9,
            "prompt_label": "Unsafe",
            "prompt_harm_category": "PII",
            "embedding_prior_score": 0.05,
            "activation_probability": 0.05,
        },
        {
            "id": "severe-miss",
            "label": "unsafe",
            "source": "synthetic",
            "domain": "cyber_abuse",
            "subcluster_role": "harmful",
            "subcluster_id": "vulnerability_exploitation",
            "span_text": "missed harmful request",
            "prompt_risk_score": 0.08,
            "prompt_confidence": 0.8,
            "prompt_label": "Safe",
            "prompt_harm_category": None,
            "embedding_prior_score": 0.05,
            "activation_probability": 0.05,
        },
        {
            "id": "hard-benign",
            "label": "safe",
            "source": "synthetic",
            "domain": "regulated_advice",
            "subcluster_role": "benign_near_neighbor",
            "subcluster_id": "general_information",
            "span_text": "benign but overflagged",
            "prompt_risk_score": 0.92,
            "prompt_confidence": 0.8,
            "prompt_label": "Unsafe",
            "prompt_harm_category": "Unethical Acts",
            "embedding_prior_score": 0.90,
            "activation_probability": 0.90,
        },
        {
            "id": "controversial-severe",
            "label": "unsafe",
            "source": "synthetic",
            "domain": "self_harm_and_wellbeing",
            "subcluster_role": "harmful",
            "subcluster_id": "suicide_methods",
            "span_text": "controversial self harm",
            "prompt_risk_score": 0.58,
            "prompt_confidence": 0.8,
            "prompt_label": "Controversial",
            "prompt_harm_category": "Suicide & Self-Harm",
            "embedding_prior_score": 0.10,
            "activation_probability": 0.10,
        },
    ]
    feature_table.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_review_batch_v0_2.py",
            "--feature-table",
            str(feature_table),
            "--review-csv",
            str(review_csv),
            "--calibration-artifact",
            str(calibration),
            "--output-jsonl",
            str(output_jsonl),
            "--output-csv",
            str(output_csv),
            "--summary-output",
            str(summary),
            "--max-rows",
            "3",
            "--max-per-bucket",
            "3",
            "--max-per-stratum",
            "3",
        ],
        check=True,
    )

    review_rows = [
        json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    csv_rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
    summary_obj = json.loads(summary.read_text(encoding="utf-8"))
    source_ids = {row["source_id"] for row in review_rows}
    buckets = {row["selection_bucket"] for row in review_rows}

    assert len(review_rows) == 3
    assert len(csv_rows) == 3
    assert "already-reviewed" not in source_ids
    assert "severe_miss_candidate" in buckets
    assert "hard_benign_fp_candidate" in buckets
    assert "controversial_high_severity" in buckets
    assert review_rows[0]["review_id"].startswith("prompt_review_v0_2_")
    assert review_rows[0]["ramp_fusion_score"] is not None
    assert summary_obj["num_rows"] == 3
