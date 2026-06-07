from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_build_embedding_centroids_script(tmp_path: Path) -> None:
    embeddings = tmp_path / "embeddings.jsonl"
    output = tmp_path / "centroids.json"
    records = [
        {
            "domain": "cyber_abuse",
            "subcluster_role": "harmful",
            "subcluster_id": "credential_theft",
            "label": "unsafe",
            "source": "synthetic",
            "embedding": [1.0, 0.0],
            "provenance": {"embedding_source_id": "test"},
        },
        {
            "domain": "cyber_abuse",
            "subcluster_role": "harmful",
            "subcluster_id": "credential_theft",
            "label": "unsafe",
            "source": "synthetic",
            "embedding": [0.0, 1.0],
            "provenance": {"embedding_source_id": "test"},
        },
        {
            "domain": "cyber_abuse",
            "subcluster_role": "benign_near_neighbor",
            "subcluster_id": "defensive_security",
            "label": "safe",
            "source": "synthetic",
            "embedding": [1.0, 0.0],
            "provenance": {"embedding_source_id": "test"},
        },
    ]
    embeddings.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_embedding_centroids.py",
            "--embeddings",
            str(embeddings),
            "--embedding-source",
            "data/embedding_source/gpt_oss_20b_hidden_state_v0_1.json",
            "--taxonomy",
            "data/taxonomy/ramp_taxonomy_v0_1.json",
            "--output",
            str(output),
            "--min-count-warning",
            "2",
        ],
        check=True,
    )

    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["total_embedding_rows"] == 3
    assert artifact["num_centroids"] == 2
    assert artifact["dimension"] == 2
    assert artifact["centroid_method"] == "mean_of_l2_normalized_vectors_then_l2_normalize"
    assert len(artifact["warnings"]) == 1

    harmful = next(
        centroid
        for centroid in artifact["centroids"]
        if centroid["subcluster_id"] == "credential_theft"
    )
    assert harmful["count"] == 2
    assert harmful["label_counts"] == {"unsafe": 2}
    assert round(harmful["centroid"][0], 6) == round(2**-0.5, 6)
    assert round(harmful["centroid"][1], 6) == round(2**-0.5, 6)
