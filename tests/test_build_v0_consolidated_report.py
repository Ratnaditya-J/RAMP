from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_build_v0_consolidated_report(tmp_path: Path) -> None:
    policy = tmp_path / "policy.json"
    input_policy = tmp_path / "input_policy.json"
    output = tmp_path / "output.json"
    session = tmp_path / "session.json"
    activation = tmp_path / "activation.json"
    embedding = tmp_path / "embedding.json"
    report = tmp_path / "report.md"

    write_json(
        policy,
        {
            "artifact_id": "policy",
            "decision": "prompt_activation_primary_with_audit_and_escalation_signals",
        },
    )
    write_json(
        input_policy,
        {
            "decision": "prompt_activation_runtime_policy",
            "selected_runtime_score": {
                "threshold": 0.53,
                "weights": {
                    "prompt_risk_score": 0.25,
                    "activation_probability": 0.75,
                    "embedding_prior_score": 0.0,
                },
            },
            "reviewed_split_stability": {
                "prompt_activation_calibrated": {
                    "auc_mean": 0.99,
                    "recall_mean": 0.95,
                    "false_positive_rate_mean": 0.04,
                    "false_positive_count_mean": 2,
                    "false_negative_count_mean": 3,
                },
                "prompt_embedding_activation_calibrated": {
                    "auc_mean": 0.98,
                    "recall_mean": 0.94,
                    "false_positive_rate_mean": 0.05,
                    "false_positive_count_mean": 4,
                    "false_negative_count_mean": 5,
                },
            },
        },
    )
    write_json(
        output,
        {
            "aggregate_holdout_metrics": {
                "prompt_embedding_activation_calibrated": {
                    "auc": {"mean": 0.90},
                    "recall": {"mean": 0.91},
                    "false_positive_rate": {"mean": 0.10},
                }
            }
        },
    )
    session_report = {
        "metrics": {
            "single_turn_max": {
                "recall": 0.8,
                "false_positive_rate": 0.2,
                "f1": 0.7,
            }
        },
        "threshold_sweeps": {"single_turn_max": {"auc": 0.75}},
        "single_turn_false_negatives_caught": {"single_turn_max": []},
    }
    write_json(session, session_report)
    write_json(activation, {"selected_layer_id": "19"})
    write_json(embedding, {"runs": []})

    subprocess.run(
        [
            sys.executable,
            "scripts/build_v0_consolidated_report.py",
            "--policy",
            str(policy),
            "--input-policy",
            str(input_policy),
            "--output-calibration",
            str(output),
            "--session-rjudge",
            str(session),
            "--session-mhj",
            str(session),
            "--activation-comparison",
            str(activation),
            "--embedding-calibration",
            str(embedding),
            "--output-md",
            str(report),
        ],
        check=True,
    )

    text = report.read_text(encoding="utf-8")
    assert "RAMP v0 Consolidated Research Report" in text
    assert "Prompt weight: `0.25`" in text
    assert "`single_turn_max`" in text
