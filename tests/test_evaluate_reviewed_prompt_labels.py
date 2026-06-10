from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_evaluate_reviewed_prompt_labels_outputs_metrics(tmp_path: Path) -> None:
    review_csv = tmp_path / "review.csv"
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"
    reviewed_jsonl = tmp_path / "reviewed.jsonl"

    rows = [
        {
            "review_id": "r1",
            "source_id": "s1",
            "review_status": "reviewed",
            "reviewed_label": "safe",
            "prompt_text": "safe prompt",
            "corpus_label": "unsafe",
            "qwen_label": "Safe",
            "qwen_risk_score": "0.08",
            "audit_bucket": "corpus_unsafe_classifier_safe",
            "source": "synthetic",
            "domain": "cyber_abuse",
            "subcluster_id": "defensive_security",
        },
        {
            "review_id": "r2",
            "source_id": "s2",
            "review_status": "reviewed",
            "reviewed_label": "unsafe",
            "prompt_text": "unsafe prompt",
            "corpus_label": "safe",
            "qwen_label": "Unsafe",
            "qwen_risk_score": "0.92",
            "audit_bucket": "corpus_safe_classifier_unsafe",
            "source": "synthetic",
            "domain": "cyber_abuse",
            "subcluster_id": "vulnerability_exploitation",
        },
        {
            "review_id": "r3",
            "source_id": "s3",
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
        {
            "review_id": "r4",
            "source_id": "s4",
            "review_status": "unreviewed",
            "reviewed_label": "",
            "prompt_text": "ignored prompt",
            "corpus_label": "safe",
            "qwen_label": "Unsafe",
            "qwen_risk_score": "0.92",
            "audit_bucket": "corpus_safe_classifier_unsafe",
            "source": "synthetic",
            "domain": "regulated_advice",
            "subcluster_id": "general_information",
        },
    ]
    with review_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_reviewed_prompt_labels.py",
            "--review-csv",
            str(review_csv),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--reviewed-jsonl",
            str(reviewed_jsonl),
        ],
        check=True,
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    reviewed_rows = [
        json.loads(line) for line in reviewed_jsonl.read_text(encoding="utf-8").splitlines()
    ]

    assert report["num_reviewed_rows"] == 3
    assert report["num_binary_eval_rows"] == 2
    assert report["binary_qwen_score_auc"] == 1.0
    assert report["binary_metrics_threshold_0_5"]["accuracy"] == 1.0
    assert report["reviewed_label_counts"]["controversial"] == 1
    assert len(reviewed_rows) == 3
    assert "Reviewed Prompt-Label Evaluation" in output_md.read_text(encoding="utf-8")
