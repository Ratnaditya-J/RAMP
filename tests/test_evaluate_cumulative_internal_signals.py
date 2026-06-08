from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_evaluate_cumulative_internal_signals_outputs_report_and_table(tmp_path: Path) -> None:
    embedding_scores = tmp_path / "embedding_scores.jsonl"
    prompt_scores = tmp_path / "prompt_scores.jsonl"
    activation = tmp_path / "activation.jsonl"
    probe = tmp_path / "probe.json"
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"
    feature_table = tmp_path / "features.jsonl"

    embedding_rows = [
        ("safe-1", "safe", -0.4),
        ("safe-2", "safe", -0.2),
        ("unsafe-1", "unsafe", 0.4),
        ("unsafe-2", "unsafe", 0.6),
    ]
    activation_vectors = {
        "safe-1": [0.0, 1.0],
        "safe-2": [0.1, 1.0],
        "unsafe-1": [1.0, 0.0],
        "unsafe-2": [1.0, 0.1],
    }
    embedding_scores.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": row_id,
                    "label": label,
                    "source": "synthetic",
                    "domain": "cyber_abuse",
                    "subcluster_role": (
                        "harmful" if label == "unsafe" else "benign_near_neighbor"
                    ),
                    "subcluster_id": (
                        "vulnerability_exploitation"
                        if label == "unsafe"
                        else "defensive_security"
                    ),
                    "span_text": row_id,
                    "risk_margin": margin,
                    "top_risk_cluster": "vulnerability_exploitation",
                    "top_benign_cluster": "defensive_security",
                }
            )
            for row_id, label, margin in embedding_rows
        )
        + "\n",
        encoding="utf-8",
    )
    activation.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": row_id,
                    "label": label,
                    "source": "synthetic",
                    "domain": "cyber_abuse",
                    "subcluster_role": (
                        "harmful" if label == "unsafe" else "benign_near_neighbor"
                    ),
                    "subcluster_id": (
                        "vulnerability_exploitation"
                        if label == "unsafe"
                        else "defensive_security"
                    ),
                    "span_text": row_id,
                    "embedding": activation_vectors[row_id],
                }
            )
            for row_id, label, _ in embedding_rows
        )
        + "\n",
        encoding="utf-8",
    )
    prompt_scores.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": row_id,
                    "label": label,
                    "source": "synthetic",
                    "domain": "cyber_abuse",
                    "subcluster_role": (
                        "harmful" if label == "unsafe" else "benign_near_neighbor"
                    ),
                    "subcluster_id": (
                        "vulnerability_exploitation"
                        if label == "unsafe"
                        else "defensive_security"
                    ),
                    "span_text": row_id,
                    "prompt_risk_score": 0.9 if label == "unsafe" else 0.1,
                    "prompt_confidence": 0.8,
                    "prompt_label": "Unsafe" if label == "unsafe" else "Safe",
                    "prompt_harm_category": None,
                    "prompt_classifier_version": "synthetic_prompt_classifier",
                }
            )
            for row_id, label, _ in embedding_rows
        )
        + "\n",
        encoding="utf-8",
    )
    probe.write_text(
        json.dumps(
            {
                "artifact_id": "probe",
                "layer_id": "1",
                "weights": [2.0, -2.0],
                "bias": 0.0,
                "mean": [0.0, 0.0],
                "scale": [1.0, 1.0],
                "selected_threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_cumulative_internal_signals.py",
            "--embedding-scores",
            str(embedding_scores),
            "--activation",
            str(activation),
            "--activation-probe",
            str(probe),
            "--prompt-scores",
            str(prompt_scores),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--feature-table",
            str(feature_table),
        ],
        check=True,
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in feature_table.read_text(encoding="utf-8").splitlines()]

    assert report["num_rows"] == 4
    assert set(report["ablations"]) == {
        "embedding_only",
        "activation_only",
        "cumulative_fixed",
        "prompt_only",
        "prompt_embedding",
        "prompt_activation",
        "prompt_embedding_activation",
    }
    assert rows[0]["prompt_risk_score"] == 0.1
    assert rows[0]["cumulative_full_score"] is not None
    assert rows[0]["embedding_prior_score"] > 0.0
    assert rows[0]["activation_probability"] > 0.0
    assert "Cumulative Internal-Signal Evaluation" in output_md.read_text(encoding="utf-8")
