from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_build_prompt_label_review_batch_outputs_reviewer_fields(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    output_jsonl = tmp_path / "review.jsonl"
    output_csv = tmp_path / "review.csv"
    summary = tmp_path / "summary.json"

    rows = []
    for idx in range(6):
        bucket = "corpus_unsafe_classifier_safe" if idx < 3 else "corpus_safe_classifier_unsafe"
        rows.append(
            {
                "id": f"row-{idx}",
                "label": "unsafe" if idx < 3 else "safe",
                "prompt_label": "Safe" if idx < 3 else "Unsafe",
                "prompt_risk_score": 0.08 if idx < 3 else 0.92,
                "prompt_confidence": 0.86,
                "audit_bucket": bucket,
                "audit_priority": 10.0 - idx,
                "source": "synthetic",
                "domain": "cyber_abuse" if idx % 2 == 0 else "regulated_advice",
                "subcluster_role": "harmful",
                "subcluster_id": "vulnerability_exploitation",
                "span_text": f"Prompt {idx}",
                "embedding_margin": 0.1,
                "activation_probability": 0.2,
                "prompt_classifier_version": "qwen3guard:test",
            }
        )
    candidates.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_prompt_label_review_batch.py",
            "--candidates",
            str(candidates),
            "--output-jsonl",
            str(output_jsonl),
            "--output-csv",
            str(output_csv),
            "--summary-output",
            str(summary),
            "--max-rows",
            "4",
            "--max-per-stratum",
            "2",
        ],
        check=True,
    )

    review_rows = [
        json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    csv_rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
    summary_obj = json.loads(summary.read_text(encoding="utf-8"))

    assert len(review_rows) == 4
    assert len(csv_rows) == 4
    assert review_rows[0]["review_status"] == "unreviewed"
    assert review_rows[0]["reviewed_label"] == ""
    assert "bad_benchmark_label" in review_rows[0]["review_label_options"]
    assert review_rows[0]["source_id"].startswith("row-")
    assert summary_obj["num_rows"] == 4
    assert set(summary_obj["by_audit_bucket"]) == {
        "corpus_safe_classifier_unsafe",
        "corpus_unsafe_classifier_safe",
    }
