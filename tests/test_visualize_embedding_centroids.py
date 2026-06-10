from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_visualize_embedding_centroids_script(tmp_path: Path) -> None:
    artifact = {
        "centroid_artifact_id": "test_centroids_v0",
        "total_embedding_rows": 3,
        "num_centroids": 2,
        "dimension": 3,
        "warnings": [],
        "centroids": [
            {
                "domain": "cyber_abuse",
                "subcluster_role": "harmful",
                "subcluster_id": "credential_theft",
                "count": 2,
                "dimension": 3,
                "centroid": [1.0, 0.0, 0.0],
                "label_counts": {"unsafe": 2},
                "source_counts": {"synthetic": 2},
            },
            {
                "domain": "cyber_abuse",
                "subcluster_role": "benign_near_neighbor",
                "subcluster_id": "defensive_security",
                "count": 1,
                "dimension": 3,
                "centroid": [0.0, 1.0, 0.0],
                "label_counts": {"safe": 1},
                "source_counts": {"synthetic": 1},
            },
        ],
    }
    centroids = tmp_path / "centroids.json"
    output = tmp_path / "centroids.html"
    centroids.write_text(json.dumps(artifact), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/visualize_embedding_centroids.py",
            "--centroids",
            str(centroids),
            "--output",
            str(output),
            "--title",
            "Test Centroids",
        ],
        check=True,
    )

    html = output.read_text(encoding="utf-8")
    assert "Test Centroids" in html
    assert "test_centroids_v0" in html
    assert "<svg" in html
    assert "credential theft" in html
    assert "defensive security" in html
