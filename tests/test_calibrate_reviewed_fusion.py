from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_calibrate_reviewed_fusion_selects_weighted_candidate(tmp_path: Path) -> None:
    review_csv = tmp_path / "review.csv"
    feature_table = tmp_path / "features.jsonl"
    output_json = tmp_path / "calibration.json"
    output_md = tmp_path / "calibration.md"

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
            "source_id": "safe-2",
            "review_status": "reviewed",
            "reviewed_label": "safe",
            "prompt_text": "safe prompt",
            "corpus_label": "safe",
            "qwen_label": "Safe",
            "qwen_risk_score": "0.08",
            "audit_bucket": "corpus_unsafe_classifier_safe",
            "source": "synthetic",
            "domain": "cyber_abuse",
            "subcluster_id": "defensive_security",
        },
        {
            "review_id": "r3",
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
            "review_id": "r4",
            "source_id": "unsafe-2",
            "review_status": "reviewed",
            "reviewed_label": "unsafe",
            "prompt_text": "unsafe prompt two",
            "corpus_label": "unsafe",
            "qwen_label": "Controversial",
            "qwen_risk_score": "0.58",
            "audit_bucket": "corpus_unsafe_classifier_controversial",
            "source": "synthetic",
            "domain": "cyber_abuse",
            "subcluster_id": "vulnerability_exploitation",
        },
    ]
    with review_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)

    feature_rows = [
        {
            "id": "safe-1",
            "prompt_risk_score": 0.92,
            "embedding_prior_score": 0.1,
            "activation_probability": 0.1,
        },
        {
            "id": "safe-2",
            "prompt_risk_score": 0.08,
            "embedding_prior_score": 0.1,
            "activation_probability": 0.1,
        },
        {
            "id": "unsafe-1",
            "prompt_risk_score": 0.92,
            "embedding_prior_score": 0.9,
            "activation_probability": 0.9,
        },
        {
            "id": "unsafe-2",
            "prompt_risk_score": 0.58,
            "embedding_prior_score": 0.9,
            "activation_probability": 0.9,
        },
    ]
    feature_table.write_text(
        "\n".join(json.dumps(row) for row in feature_rows) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_reviewed_fusion.py",
            "--review-csv",
            str(review_csv),
            "--feature-table",
            str(feature_table),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--weight-step",
            "0.10",
            "--threshold-step",
            "0.10",
            "--target-fpr",
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
    selected = report["selected_by_target_fpr"]

    assert report["num_binary_eval_rows"] == 4
    assert selected["metrics"]["false_positive_rate"] <= 0.10
    assert selected["metrics"]["recall"] == 1.0
    assert selected["embedding_weight"] <= 0.20
    assert selected["prompt_weight"] >= selected["activation_weight"]
    assert selected["activation_weight"] >= selected["embedding_weight"]
    assert "Reviewed Fusion Calibration" in output_md.read_text(encoding="utf-8")
