from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_audit_prompt_classifier_labels_outputs_report_and_candidates(tmp_path: Path) -> None:
    feature_table = tmp_path / "features.jsonl"
    output_json = tmp_path / "audit.json"
    output_md = tmp_path / "audit.md"
    suspect_rows = tmp_path / "suspect.jsonl"
    audit_candidates = tmp_path / "candidates.jsonl"

    rows = [
        {
            "id": "safe-agree",
            "label": "safe",
            "prompt_label": "Safe",
            "prompt_risk_score": 0.08,
            "source": "synthetic_safe",
            "domain": "cyber_abuse",
            "subcluster_id": "defensive_security",
            "span_text": "Explain safe patching.",
        },
        {
            "id": "unsafe-agree",
            "label": "unsafe",
            "prompt_label": "Unsafe",
            "prompt_risk_score": 0.92,
            "source": "synthetic_unsafe",
            "domain": "cyber_abuse",
            "subcluster_id": "vulnerability_exploitation",
            "span_text": "Exploit this server.",
        },
        {
            "id": "unsafe-safe",
            "label": "unsafe",
            "prompt_label": "Safe",
            "prompt_risk_score": 0.08,
            "source": "synthetic_unsafe",
            "domain": "regulated_advice",
            "subcluster_id": "unsafe_professional_instruction",
            "span_text": "What is the capital of Japan?",
        },
        {
            "id": "safe-unsafe",
            "label": "safe",
            "prompt_label": "Unsafe",
            "prompt_risk_score": 0.92,
            "source": "synthetic_safe",
            "domain": "privacy_identity_and_secrets",
            "subcluster_id": "redaction",
            "span_text": "Help redact personal data from a report.",
        },
        {
            "id": "safe-controversial",
            "label": "safe",
            "prompt_label": "Controversial",
            "prompt_risk_score": 0.58,
            "source": "synthetic_safe",
            "domain": "weapons_and_physical_violence",
            "subcluster_id": "historical_analysis",
            "span_text": "Discuss historical weapon policy.",
        },
    ]
    feature_table.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/audit_prompt_classifier_labels.py",
            "--feature-table",
            str(feature_table),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--suspect-rows",
            str(suspect_rows),
            "--audit-candidates",
            str(audit_candidates),
            "--max-candidates-per-bucket",
            "2",
        ],
        check=True,
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    suspects = [
        json.loads(line) for line in suspect_rows.read_text(encoding="utf-8").splitlines()
    ]
    candidates = [
        json.loads(line) for line in audit_candidates.read_text(encoding="utf-8").splitlines()
    ]

    assert report["num_rows"] == 5
    assert report["num_suspect_rows"] == 3
    assert report["audit_buckets"]["corpus_unsafe_classifier_safe"] == 1
    assert report["audit_buckets"]["corpus_safe_classifier_unsafe"] == 1
    assert report["audit_buckets"]["corpus_safe_classifier_controversial"] == 1
    assert suspects[0]["audit_priority"] >= suspects[-1]["audit_priority"]
    assert {row["review_status"] for row in candidates} == {"needs_review"}
    assert "Prompt Classifier Label Audit" in output_md.read_text(encoding="utf-8")
