from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_embedding_feature_calibration_report_outputs_recommendation(tmp_path: Path) -> None:
    scores = tmp_path / "scores.jsonl"
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"
    rows = [
        {
            "id": "safe-1",
            "label": "safe",
            "source": "synthetic",
            "domain": "cyber_abuse",
            "subcluster_role": "benign_near_neighbor",
            "subcluster_id": "defensive_security",
            "risk_margin": -0.4,
            "harm_similarity": 0.1,
            "benign_similarity": 0.5,
            "top_risk_cluster": "vulnerability_exploitation",
            "top_risk_domain": "cyber_abuse",
            "top_benign_cluster": "defensive_security",
        },
        {
            "id": "safe-2",
            "label": "safe",
            "source": "synthetic",
            "domain": "cyber_abuse",
            "subcluster_role": "benign_near_neighbor",
            "subcluster_id": "defensive_security",
            "risk_margin": 0.2,
            "harm_similarity": 0.7,
            "benign_similarity": 0.5,
            "top_risk_cluster": "vulnerability_exploitation",
            "top_risk_domain": "cyber_abuse",
            "top_benign_cluster": "defensive_security",
        },
        {
            "id": "unsafe-1",
            "label": "unsafe",
            "source": "synthetic",
            "domain": "cyber_abuse",
            "subcluster_role": "harmful",
            "subcluster_id": "vulnerability_exploitation",
            "risk_margin": 0.5,
            "harm_similarity": 0.8,
            "benign_similarity": 0.3,
            "top_risk_cluster": "vulnerability_exploitation",
            "top_risk_domain": "cyber_abuse",
            "top_benign_cluster": "defensive_security",
        },
        {
            "id": "unsafe-2",
            "label": "unsafe",
            "source": "synthetic",
            "domain": "cyber_abuse",
            "subcluster_role": "harmful",
            "subcluster_id": "vulnerability_exploitation",
            "risk_margin": 0.7,
            "harm_similarity": 0.9,
            "benign_similarity": 0.2,
            "top_risk_cluster": "vulnerability_exploitation",
            "top_risk_domain": "cyber_abuse",
            "top_benign_cluster": "defensive_security",
        },
    ]
    scores.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/report_embedding_feature_calibration.py",
            "--scores",
            str(scores),
            "--names",
            "synthetic",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    markdown = output_md.read_text(encoding="utf-8")

    run = report["runs"][0]
    assert run["name"] == "synthetic"
    assert run["labels"] == {"safe": 2, "unsafe": 2}
    assert run["thresholds"]["zero_margin"]["tp"] == 2
    assert run["thresholds"]["zero_margin"]["fp"] == 1
    assert run["hard_neighbors"]["num_safe_high_margin"] == 1
    assert run["recommendation"]["recommended_role"] in {
        "supporting_semantic_prior",
        "candidate_decision_feature_after_heldout_validation",
        "weak_routing_signal",
    }
    assert "Embedding Feature Calibration Report" in markdown
