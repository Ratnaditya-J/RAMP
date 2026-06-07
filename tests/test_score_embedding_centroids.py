from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_score_embedding_centroids_script_outputs_rows_and_summary(tmp_path: Path) -> None:
    embeddings = tmp_path / "embeddings.jsonl"
    centroids = tmp_path / "centroids.json"
    output = tmp_path / "scores.jsonl"
    summary = tmp_path / "summary.json"

    embeddings.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "id": "unsafe-1",
                    "label": "unsafe",
                    "source": "synthetic",
                    "domain": "cyber_abuse",
                    "subcluster_role": "harmful",
                    "subcluster_id": "vulnerability_exploitation",
                    "span_text": "exploit the service",
                    "embedding": [1.0, 0.0],
                },
                {
                    "id": "safe-1",
                    "label": "safe",
                    "source": "synthetic",
                    "domain": "cyber_abuse",
                    "subcluster_role": "benign_near_neighbor",
                    "subcluster_id": "defensive_security",
                    "span_text": "patch the service",
                    "embedding": [0.0, 1.0],
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    centroids.write_text(
        json.dumps(
            {
                "centroid_artifact_id": "score_test_centroids",
                "corpus_mean_vector": [0.5, 0.5],
                "centroids": [
                    {
                        "domain": "cyber_abuse",
                        "subcluster_role": "harmful",
                        "subcluster_id": "vulnerability_exploitation",
                        "count": 1,
                        "centroid": [1.0, 0.0],
                    },
                    {
                        "domain": "cyber_abuse",
                        "subcluster_role": "benign_near_neighbor",
                        "subcluster_id": "defensive_security",
                        "count": 1,
                        "centroid": [0.0, 1.0],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/score_embedding_centroids.py",
            "--embeddings",
            str(embeddings),
            "--centroids",
            str(centroids),
            "--output",
            str(output),
            "--summary-output",
            str(summary),
            "--similarity-mode",
            "centered_cosine",
        ],
        check=True,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    report = json.loads(summary.read_text(encoding="utf-8"))

    assert len(rows) == 2
    assert rows[0]["top_harm_cluster"] == "vulnerability_exploitation"
    assert rows[1]["top_benign_cluster"] == "defensive_security"
    assert report["num_records"] == 2
    assert report["labels"] == {"safe": 1, "unsafe": 1}
    assert report["similarity_mode"] == "centered_cosine"
