from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def write_activation(path: Path, *, layer: str) -> None:
    rows = [
        ("safe-1", "safe", [0.0, 1.0]),
        ("safe-2", "safe", [0.1, 1.0]),
        ("safe-3", "safe", [0.0, 0.9]),
        ("unsafe-1", "unsafe", [1.0, 0.0]),
        ("unsafe-2", "unsafe", [1.0, 0.1]),
        ("unsafe-3", "unsafe", [0.9, 0.0]),
    ]
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": row_id,
                    "span_text": row_id,
                    "domain": "cyber_abuse",
                    "subcluster_role": "harmful" if label == "unsafe" else "benign_near_neighbor",
                    "subcluster_id": (
                        "vulnerability_exploitation"
                        if label == "unsafe"
                        else "defensive_security"
                    ),
                    "label": label,
                    "source": "synthetic",
                    "embedding": vector,
                    "provenance": {
                        "huggingface_model_id": "synthetic",
                        "representation": "hidden_state",
                        "layer_ids": [layer],
                    },
                }
            )
            for row_id, label, vector in rows
        )
        + "\n",
        encoding="utf-8",
    )


def test_train_activation_probes_outputs_layer_artifact_and_report(tmp_path: Path) -> None:
    activation = tmp_path / "layer_1.jsonl"
    output_dir = tmp_path / "probes"
    report = tmp_path / "report.json"
    write_activation(activation, layer="1")

    subprocess.run(
        [
            sys.executable,
            "scripts/train_activation_probes.py",
            "--activation",
            str(activation),
            "--layer",
            "1",
            "--output-dir",
            str(output_dir),
            "--report-output",
            str(report),
            "--epochs",
            "8",
            "--learning-rate",
            "0.2",
            "--seed",
            "3",
        ],
        check=True,
    )

    probe = json.loads((output_dir / "ramp_activation_probe_layer_1_v0_1.json").read_text())
    comparison = json.loads(report.read_text())

    assert probe["artifact_id"] == "ramp_activation_probe_layer_1_v0.1"
    assert probe["layer_id"] == "1"
    assert len(probe["weights"]) == 2
    assert comparison["selected_layer_id"] == "1"
    assert comparison["layers"][0]["test_rows"] > 0
