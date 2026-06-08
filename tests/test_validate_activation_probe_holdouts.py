from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_validate_activation_probe_holdouts_outputs_report(tmp_path: Path) -> None:
    activation = tmp_path / "activation.jsonl"
    output = tmp_path / "holdout.json"
    rows = []
    for source in ("source_a", "source_b"):
        for idx in range(4):
            rows.append(
                {
                    "id": f"{source}-safe-{idx}",
                    "span_text": "safe",
                    "domain": "cyber_abuse",
                    "subcluster_role": "benign_near_neighbor",
                    "subcluster_id": "defensive_security",
                    "label": "safe",
                    "source": source,
                    "embedding": [0.0, 1.0 + idx * 0.01],
                    "provenance": {"representation": "hidden_state", "layer_ids": ["1"]},
                }
            )
            rows.append(
                {
                    "id": f"{source}-unsafe-{idx}",
                    "span_text": "unsafe",
                    "domain": "cyber_abuse",
                    "subcluster_role": "harmful",
                    "subcluster_id": "vulnerability_exploitation",
                    "label": "unsafe",
                    "source": source,
                    "embedding": [1.0 + idx * 0.01, 0.0],
                    "provenance": {"representation": "hidden_state", "layer_ids": ["1"]},
                }
            )
    activation.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/validate_activation_probe_holdouts.py",
            "--activation",
            str(activation),
            "--layer",
            "1",
            "--holdout-key",
            "source",
            "--holdout-value",
            "source_a",
            "--output",
            str(output),
            "--epochs",
            "8",
            "--learning-rate",
            "0.2",
        ],
        check=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["layer_id"] == "1"
    assert report["num_valid_reports"] == 1
    assert report["reports"][0]["holdout_value"] == "source_a"
    assert report["reports"][0]["test_rows"] == 8
