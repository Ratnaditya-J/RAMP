from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_build_review_batch_v0_4_targets_activation_fns_and_embedding_fps(
    tmp_path: Path,
) -> None:
    feature_table = tmp_path / "features.jsonl"
    exclude_csv = tmp_path / "exclude.csv"
    output_jsonl = tmp_path / "batch.jsonl"
    output_csv = tmp_path / "batch.csv"
    summary_json = tmp_path / "summary.json"

    with exclude_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["source_id", "review_status"])
        writer.writeheader()
        writer.writerow({"source_id": "already-reviewed", "review_status": "reviewed"})

    rows = [
        {
            "id": "already-reviewed",
            "label": "unsafe",
            "source": "synthetic",
            "domain": "cyber_abuse",
            "subcluster_role": "harmful",
            "subcluster_id": "vulnerability_exploitation",
            "span_text": "excluded row",
            "prompt_risk_score": 0.9,
            "embedding_prior_score": 0.8,
            "activation_probability": 0.01,
        },
            {
                "id": "activation-miss",
                "label": "unsafe",
                "source": "synthetic",
                "domain": "regulated_advice",
                "subcluster_role": "harmful",
                "subcluster_id": "unsafe_professional_instruction",
                "span_text": "unsafe row with low activation",
                "prompt_risk_score": 0.7,
                "embedding_prior_score": 0.8,
            "embedding_margin": 0.3,
            "activation_probability": 0.02,
        },
        {
            "id": "severe-activation-miss",
            "label": "unsafe",
            "source": "synthetic",
            "domain": "weapons_and_physical_violence",
            "subcluster_role": "harmful",
            "subcluster_id": "weapon_construction",
            "span_text": "severe unsafe row with low activation",
            "prompt_risk_score": 0.6,
            "embedding_prior_score": 0.7,
            "embedding_margin": 0.2,
            "activation_probability": 0.03,
        },
        {
            "id": "embedding-fp",
            "label": "safe",
            "source": "synthetic",
            "domain": "cyber_abuse",
            "subcluster_role": "benign_near_neighbor",
            "subcluster_id": "defensive_security",
            "span_text": "defensive benign row with high embedding",
            "prompt_risk_score": 0.2,
            "embedding_prior_score": 0.9,
            "embedding_margin": 0.5,
            "activation_probability": 0.1,
        },
    ]
    feature_table.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_review_batch_v0_4.py",
            "--feature-table",
            str(feature_table),
            "--exclude-csv",
            str(exclude_csv),
            "--output-jsonl",
            str(output_jsonl),
            "--output-csv",
            str(output_csv),
            "--summary-output",
            str(summary_json),
            "--max-rows",
            "20",
            "--max-per-stratum",
            "1",
        ],
        check=True,
    )

    batch_rows = [
        json.loads(line)
        for line in output_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_ids = {row["source_id"] for row in batch_rows}
    buckets = {row["selection_bucket"] for row in batch_rows}
    assert "already-reviewed" not in source_ids
    assert "activation_false_negative_candidate" in buckets
    assert "severe_activation_false_negative_candidate" in buckets
    assert "embedding_false_positive_candidate" in buckets
    assert output_csv.exists()
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["artifact_id"] == "ramp_prompt_review_batch_v0.4"
