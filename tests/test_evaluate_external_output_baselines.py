from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_evaluate_external_output_baselines_reports_named_scores(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "output_corpus.jsonl"
    scores = tmp_path / "scores.jsonl"
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"

    corpus_rows = []
    score_rows = []
    for idx in range(20):
        unsafe = idx % 2 == 1
        label = "unsafe" if unsafe else "safe"
        row_id = f"eval-{idx:02d}"
        corpus_rows.append(
            {
                "id": row_id,
                "eval_id": row_id,
                "source": "wildguardmix" if idx < 10 else "beavertails",
                "prompt_text": f"prompt {idx}",
                "output_text": f"output {idx}",
                "output_reviewed_label": label,
                "label": label,
            }
        )
        score_rows.append(
            {
                "id": row_id,
                "eval_id": row_id,
                "output_risk_score": 0.8 if unsafe else 0.2,
                "output_label": "Unsafe" if unsafe else "Safe",
            }
        )
    corpus.write_text(
        "\n".join(json.dumps(row) for row in corpus_rows) + "\n",
        encoding="utf-8",
    )
    scores.write_text(
        "\n".join(json.dumps(row) for row in score_rows) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_external_output_baselines.py",
            "--corpus",
            str(corpus),
            "--score",
            f"qwen={scores}",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["artifact_id"] == "ramp_external_output_baselines_v0.1"
    assert report["baselines"]["qwen"]["overall"]["rows"] == 20
    assert "target_fpr_0_05" in report["baselines"]["qwen"]["overall"]["thresholds"]
    assert "External Output Baseline Evaluation" in output_md.read_text(
        encoding="utf-8"
    )
