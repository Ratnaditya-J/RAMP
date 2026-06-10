from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_evaluate_reviewed_cumulative_signals_reports_ablation_deltas(
    tmp_path: Path,
) -> None:
    review_csv = tmp_path / "review.csv"
    feature_table = tmp_path / "features.jsonl"
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"
    reviewed_features = tmp_path / "reviewed_features.jsonl"

    review_rows = [
        {
            "review_id": "r1",
            "source_id": "safe-1",
            "review_status": "reviewed",
            "reviewed_label": "safe",
            "prompt_text": "safe but prompt flags",
            "corpus_label": "safe",
            "qwen_label": "Unsafe",
            "qwen_risk_score": "0.92",
            "audit_bucket": "corpus_safe_classifier_unsafe",
            "source": "synthetic",
            "domain": "cyber_abuse",
            "subcluster_id": "defensive_security",
        },
        {
            "review_id": "r2",
            "source_id": "unsafe-1",
            "review_status": "reviewed",
            "reviewed_label": "unsafe",
            "prompt_text": "unsafe prompt",
            "corpus_label": "unsafe",
            "qwen_label": "Unsafe",
            "qwen_risk_score": "0.92",
            "audit_bucket": "corpus_unsafe_classifier_agree",
            "source": "synthetic",
            "domain": "cyber_abuse",
            "subcluster_id": "vulnerability_exploitation",
        },
        {
            "review_id": "r3",
            "source_id": "boundary-1",
            "review_status": "reviewed",
            "reviewed_label": "controversial",
            "prompt_text": "boundary prompt",
            "corpus_label": "safe",
            "qwen_label": "Controversial",
            "qwen_risk_score": "0.58",
            "audit_bucket": "corpus_safe_classifier_controversial",
            "source": "synthetic",
            "domain": "regulated_advice",
            "subcluster_id": "general_information",
        },
    ]
    with review_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)

    feature_rows = [
        {
            "id": "safe-1",
            "label": "safe",
            "source": "synthetic",
            "domain": "cyber_abuse",
            "subcluster_role": "benign_near_neighbor",
            "subcluster_id": "defensive_security",
            "span_text": "safe but prompt flags",
            "prompt_risk_score": 0.92,
            "prompt_label": "Unsafe",
            "embedding_margin": -0.5,
            "embedding_prior_score": 0.1,
            "activation_probability": 0.1,
            "prompt_embedding_score": 0.3,
            "prompt_activation_score": 0.3,
            "cumulative_full_score": 0.3,
        },
        {
            "id": "unsafe-1",
            "label": "unsafe",
            "source": "synthetic",
            "domain": "cyber_abuse",
            "subcluster_role": "harmful",
            "subcluster_id": "vulnerability_exploitation",
            "span_text": "unsafe prompt",
            "prompt_risk_score": 0.92,
            "prompt_label": "Unsafe",
            "embedding_margin": 0.5,
            "embedding_prior_score": 0.9,
            "activation_probability": 0.9,
            "prompt_embedding_score": 0.9,
            "prompt_activation_score": 0.9,
            "cumulative_full_score": 0.9,
        },
        {
            "id": "boundary-1",
            "label": "safe",
            "source": "synthetic",
            "domain": "regulated_advice",
            "subcluster_role": "benign_near_neighbor",
            "subcluster_id": "general_information",
            "span_text": "boundary prompt",
            "prompt_risk_score": 0.58,
            "prompt_label": "Controversial",
            "embedding_margin": 0.0,
            "embedding_prior_score": 0.5,
            "activation_probability": 0.5,
            "prompt_embedding_score": 0.55,
            "prompt_activation_score": 0.55,
            "cumulative_full_score": 0.55,
        },
    ]
    feature_table.write_text(
        "\n".join(json.dumps(row) for row in feature_rows) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_reviewed_cumulative_signals.py",
            "--review-csv",
            str(review_csv),
            "--feature-table",
            str(feature_table),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--reviewed-feature-table",
            str(reviewed_features),
        ],
        check=True,
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    reviewed = [
        json.loads(line) for line in reviewed_features.read_text(encoding="utf-8").splitlines()
    ]

    assert report["num_reviewed_joined_rows"] == 3
    assert report["num_binary_eval_rows"] == 2
    assert report["ablations"]["prompt_risk_score"]["metrics"]["default_0_5"]["fp"] == 1
    assert report["ablations"]["cumulative_full_score"]["metrics"]["default_0_5"]["fp"] == 0
    assert report["false_positive_delta"]["prompt_fp_fixed_by_cumulative_full"] == ["safe-1"]
    assert len(reviewed) == 3
    assert "Reviewed Cumulative Signal Evaluation" in output_md.read_text(encoding="utf-8")
