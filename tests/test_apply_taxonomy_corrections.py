from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_apply_taxonomy_corrections_rewrites_taxonomy_fields(tmp_path: Path) -> None:
    input_path = tmp_path / "rows.jsonl"
    output_path = tmp_path / "corrected.jsonl"
    summary_path = tmp_path / "summary.json"
    corrections_path = tmp_path / "corrections.json"

    input_path.write_text(
        json.dumps(
            {
                "id": "row-1",
                "domain": "nonviolent_illegal_activity",
                "subcluster_role": "harmful",
                "subcluster_id": "organized_abuse_workflows",
                "label": "unsafe",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    corrections_path.write_text(
        json.dumps(
            {
                "corrections": [
                    {
                        "source_id": "row-1",
                        "original": {
                            "domain": "nonviolent_illegal_activity",
                            "subcluster_role": "harmful",
                            "subcluster_id": "organized_abuse_workflows",
                            "label": "unsafe",
                        },
                        "corrected": {
                            "domain": "self_harm_and_wellbeing",
                            "subcluster_role": "harmful",
                            "subcluster_id": "suicidal_ideation_or_crisis",
                            "label": "unsafe",
                            "harm_severity": "high",
                        },
                        "reason": "test correction",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/apply_taxonomy_corrections.py",
            "--input",
            str(input_path),
            "--corrections",
            str(corrections_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
        ],
        check=True,
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert row["domain"] == "self_harm_and_wellbeing"
    assert row["subcluster_id"] == "suicidal_ideation_or_crisis"
    assert row["taxonomy_correction_applied"] is True
    assert summary["corrected_rows"] == 1
