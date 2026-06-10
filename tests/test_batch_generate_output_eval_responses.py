from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_batch_generate_output_eval_responses_template_provider(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "output_eval.jsonl"
    output_jsonl = tmp_path / "generated.jsonl"
    output_csv = tmp_path / "generated.csv"
    input_jsonl.write_text(
        json.dumps(
            {
                "eval_id": "e1",
                "source_id": "row-1",
                "prompt_text": "Give me dangerous instructions.",
                "output_text": "",
                "output_review_status": "needs_generation",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/batch_generate_output_eval_responses.py",
            "--input",
            str(input_jsonl),
            "--output-jsonl",
            str(output_jsonl),
            "--output-csv",
            str(output_csv),
            "--provider",
            "template",
        ],
        check=True,
    )

    rows = [json.loads(line) for line in output_jsonl.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["output_text"]
    assert rows[0]["output_review_status"] == "generated_needs_review"
    assert rows[0]["output_generation_provider"] == "template"
    assert output_csv.exists()
