from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_build_review_batch_v0_3_uses_stability_slices(tmp_path: Path) -> None:
    review_csv = tmp_path / "review.csv"
    feature_table = tmp_path / "features.jsonl"
    stability_json = tmp_path / "stability.json"
    output_jsonl = tmp_path / "batch.jsonl"
    output_csv = tmp_path / "batch.csv"
    summary_json = tmp_path / "summary.json"

    reviewed_rows = [
        {
            "review_id": "r1",
            "source_id": "reviewed-1",
            "review_status": "reviewed",
            "reviewed_label": "safe",
            "domain": "cyber_abuse",
        }
    ]
    with review_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(reviewed_rows[0]))
        writer.writeheader()
        writer.writerows(reviewed_rows)

    feature_rows = [
        {
            "id": "candidate-unsafe",
            "label": "unsafe",
            "source": "synthetic",
            "domain": "cyber_abuse",
            "subcluster_role": "harmful",
            "subcluster_id": "vulnerability_exploitation",
            "span_text": "unsafe cyber request",
            "prompt_risk_score": 0.20,
            "prompt_label": "Safe",
            "embedding_prior_score": 0.70,
            "activation_probability": 0.80,
        },
        {
            "id": "candidate-safe",
            "label": "safe",
            "source": "synthetic",
            "domain": "cyber_abuse",
            "subcluster_role": "benign_near_neighbor",
            "subcluster_id": "defensive_security",
            "span_text": "defensive security request",
            "prompt_risk_score": 0.80,
            "prompt_label": "Unsafe",
            "embedding_prior_score": 0.60,
            "activation_probability": 0.70,
        },
    ]
    feature_table.write_text(
        "\n".join(json.dumps(row) for row in feature_rows) + "\n",
        encoding="utf-8",
    )
    stability_json.write_text(
        json.dumps(
            {
                "selected_config_summary": {
                    "most_common_configs": [
                        {
                            "prompt_weight": 0.60,
                            "embedding_weight": 0.20,
                            "activation_weight": 0.20,
                            "threshold": 0.40,
                            "count": 3,
                        }
                    ]
                },
                "error_distribution": {
                    "ramp_fusion_severe_false_negatives": [
                        {
                            "domain": "cyber_abuse",
                            "subcluster_id": "vulnerability_exploitation",
                        }
                    ],
                    "ramp_fusion_hard_benign_false_positives": [
                        {
                            "domain": "cyber_abuse",
                            "subcluster_id": "defensive_security",
                        }
                    ],
                    "ramp_fusion_false_negatives": [],
                    "ramp_fusion_false_positives": [],
                },
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_review_batch_v0_3.py",
            "--feature-table",
            str(feature_table),
            "--review-csv",
            str(review_csv),
            "--stability-artifact",
            str(stability_json),
            "--output-jsonl",
            str(output_jsonl),
            "--output-csv",
            str(output_csv),
            "--summary-output",
            str(summary_json),
            "--max-rows",
            "10",
        ],
        check=True,
    )

    rows = [
        json.loads(line)
        for line in output_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    buckets = {row["selection_bucket"] for row in rows}
    assert "stability_severe_fn_slice" in buckets
    assert "stability_hard_benign_fp_slice" in buckets
    assert output_csv.exists()
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["artifact_id"] == "ramp_prompt_review_batch_v0.3"
