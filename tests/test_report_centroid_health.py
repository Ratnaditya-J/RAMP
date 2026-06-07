from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_report_centroid_health_script_flags_expected_conditions(tmp_path: Path) -> None:
    artifact = {
        "centroid_artifact_id": "health_test_centroids",
        "embedding_source_id": "test_embedding_source",
        "taxonomy_id": "test_taxonomy",
        "total_embedding_rows": 42,
        "num_centroids": 3,
        "dimension": 2,
        "centroids": [
            {
                "domain": "cyber_abuse",
                "subcluster_role": "harmful",
                "subcluster_id": "vulnerability_exploitation",
                "count": 20,
                "centroid": [1.0, 0.0],
                "source_counts": {"source_a": 20},
            },
            {
                "domain": "regulated_advice",
                "subcluster_role": "benign_near_neighbor",
                "subcluster_id": "general_information",
                "count": 20,
                "centroid": [0.95, 0.05],
                "source_counts": {"source_b": 20},
            },
            {
                "domain": "privacy_identity_and_secrets",
                "subcluster_role": "benign_near_neighbor",
                "subcluster_id": "redaction",
                "count": 2,
                "centroid": [0.0, 1.0],
                "source_counts": {"source_c": 2},
            },
        ],
    }
    centroids = tmp_path / "centroids.json"
    output = tmp_path / "health.json"
    centroids.write_text(json.dumps(artifact), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/report_centroid_health.py",
            "--centroids",
            str(centroids),
            "--output",
            str(output),
            "--min-count",
            "5",
            "--source-concentration-threshold",
            "0.90",
            "--collision-threshold",
            "0.90",
        ],
        check=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["centroid_artifact_id"] == "health_test_centroids"
    assert report["role_counts"] == {"benign_near_neighbor": 2, "harmful": 1}
    assert report["summary"]["num_low_count_centroids"] == 1
    assert report["summary"]["num_source_concentration_warnings"] == 3
    assert report["summary"]["num_harm_benign_collisions"] == 1
    assert report["summary"]["num_missing_same_domain_benign"] == 1
