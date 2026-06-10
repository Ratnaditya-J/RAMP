from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_join_output_scores_adds_output_signal_to_feature_table(tmp_path: Path) -> None:
    feature_table = tmp_path / "features.jsonl"
    output_scores = tmp_path / "output_scores.jsonl"
    joined = tmp_path / "joined.jsonl"
    summary = tmp_path / "summary.json"

    feature_table.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"id": "row-1", "prompt_risk_score": 0.2},
                {"id": "row-2", "prompt_risk_score": 0.8},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_scores.write_text(
        json.dumps(
            {
                "source_id": "row-2",
                "eval_id": "e2",
                "output_text": "Here are the steps.",
                "output_risk_score": 0.92,
                "output_label": "Unsafe",
                "output_classifier_version": "qwen3guard_output:test",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/join_output_scores.py",
            "--feature-table",
            str(feature_table),
            "--output-scores",
            str(output_scores),
            "--output",
            str(joined),
            "--summary-output",
            str(summary),
        ],
        check=True,
    )

    rows = [json.loads(line) for line in joined.read_text().splitlines()]
    by_id = {row["id"]: row for row in rows}
    assert "output_risk_score" not in by_id["row-1"]
    assert by_id["row-2"]["output_risk_score"] == 0.92
    assert by_id["row-2"]["output_eval_id"] == "e2"
    assert json.loads(summary.read_text())["matched_rows"] == 1

