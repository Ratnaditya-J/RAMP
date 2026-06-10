from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def test_train_reviewed_activation_probe_uses_reviewed_labels(tmp_path: Path) -> None:
    review_csv = tmp_path / "review.csv"
    activation = tmp_path / "activation.jsonl"
    output = tmp_path / "probe.json"
    report = tmp_path / "report.json"

    review_rows = []
    activation_rows = []
    for idx in range(12):
        label = "unsafe" if idx % 2 else "safe"
        row_id = f"row-{idx:02d}"
        review_rows.append(
            {
                "review_id": f"r{idx}",
                "source_id": row_id,
                "review_status": "reviewed",
                "reviewed_label": label,
                "prompt_text": row_id,
            }
        )
        activation_rows.append(
            {
                "id": row_id,
                "span_text": row_id,
                "domain": "cyber_abuse",
                "subcluster_role": "harmful" if label == "unsafe" else "benign_near_neighbor",
                "subcluster_id": "vulnerability_exploitation",
                "label": "safe",
                "source": "synthetic",
                "embedding": [1.0, 0.0] if label == "unsafe" else [0.0, 1.0],
                "provenance": {"huggingface_model_id": "synthetic"},
            }
        )
    with review_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    activation.write_text(
        "\n".join(json.dumps(row) for row in activation_rows) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/train_reviewed_activation_probe.py",
            "--review-csv",
            str(review_csv),
            "--activation",
            str(activation),
            "--layer",
            "1",
            "--output",
            str(output),
            "--report-output",
            str(report),
            "--epochs",
            "5",
        ],
        check=True,
    )

    artifact = json.loads(output.read_text(encoding="utf-8"))
    result = json.loads(report.read_text(encoding="utf-8"))
    assert artifact["artifact_id"] == "ramp_reviewed_activation_probe_layer_1_v0.1"
    assert artifact["review_csv"] == str(review_csv)
    assert result["reviewed_rows"] == 12
