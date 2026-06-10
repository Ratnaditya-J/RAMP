from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def write_fixture(review_csv: Path, feature_table: Path, rows: int = 20) -> None:
    review_rows = []
    feature_rows = []
    for idx in range(rows):
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
                "domain": "cyber_abuse",
            }
        )
        unsafe = label == "unsafe"
        feature_rows.append(
            {
                "id": row_id,
                "label": label,
                "source": "synthetic",
                "domain": "cyber_abuse",
                "subcluster_role": "harmful" if unsafe else "benign_near_neighbor",
                "subcluster_id": (
                    "vulnerability_exploitation" if unsafe else "defensive_security"
                ),
                "span_text": row_id,
                "prompt_risk_score": 0.85 if unsafe else 0.15,
                "prompt_label": "Unsafe" if unsafe else "Safe",
                "prompt_harm_category": None,
                "embedding_prior_score": 0.75 if unsafe else 0.20,
                "activation_probability": 0.70 if unsafe else 0.25,
                "prompt_embedding_score": 0.80 if unsafe else 0.15,
                "prompt_activation_score": 0.80 if unsafe else 0.15,
                "cumulative_full_score": 0.80 if unsafe else 0.15,
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


def test_evaluate_split_stability_reports_aggregate_deltas(tmp_path: Path) -> None:
    review_csv = tmp_path / "review.csv"
    feature_table = tmp_path / "features.jsonl"
    output_json = tmp_path / "stability.json"
    output_md = tmp_path / "stability.md"
    write_fixture(review_csv, feature_table)

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_split_stability.py",
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
    assert report["num_splits"] == 3
    assert report["num_binary_eval_rows"] == 20
    assert "prompt_risk_score" in report["aggregate_holdout_metrics"]
    assert "ramp_fusion" in report["aggregate_holdout_metrics"]
    assert (
        "ramp_fusion_minus_prompt_risk_score_auc"
        in report["ramp_fusion_vs_prompt_only_deltas"]
    )
    assert "RAMP Split Stability Evaluation" in output_md.read_text(encoding="utf-8")
