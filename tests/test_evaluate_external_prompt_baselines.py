from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_evaluate_external_prompt_baselines_reports_named_scores(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    qwen_scores = tmp_path / "qwen.jsonl"
    keyword_scores = tmp_path / "keyword.jsonl"
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"

    corpus_rows = []
    qwen_rows = []
    keyword_rows = []
    for idx in range(20):
        unsafe = idx % 2 == 1
        label = "unsafe" if unsafe else "safe"
        row_id = f"row-{idx:02d}"
        corpus_rows.append(
            {
                "id": row_id,
                "label": label,
                "source": "source_a" if idx < 10 else "source_b",
                "domain": "cyber_abuse",
                "subcluster_role": "harmful" if unsafe else "benign_near_neighbor",
                "subcluster_id": "vulnerability_exploitation",
                "span_text": row_id,
            }
        )
        qwen_rows.append(
            {
                "id": row_id,
                "prompt_risk_score": 0.8 if unsafe else 0.2,
                "prompt_label": "Unsafe" if unsafe else "Safe",
            }
        )
        keyword_rows.append(
            {
                "id": row_id,
                "prompt_risk_score": 0.55 if unsafe else 0.45,
                "prompt_label": "Unsafe" if unsafe else "Safe",
            }
        )
    corpus.write_text(
        "\n".join(json.dumps(row) for row in corpus_rows) + "\n",
        encoding="utf-8",
    )
    qwen_scores.write_text(
        "\n".join(json.dumps(row) for row in qwen_rows) + "\n",
        encoding="utf-8",
    )
    keyword_scores.write_text(
        "\n".join(json.dumps(row) for row in keyword_rows) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_external_prompt_baselines.py",
            "--corpus",
            str(corpus),
            "--score",
            f"qwen={qwen_scores}",
            "--score",
            f"keyword={keyword_scores}",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["artifact_id"] == "ramp_external_prompt_baselines_v0.1"
    assert set(report["baselines"]) == {"qwen", "keyword"}
    assert report["baselines"]["qwen"]["overall"]["rows"] == 20
    assert "target_fpr_0_05" in report["baselines"]["qwen"]["overall"]["thresholds"]
    assert "External Prompt Baseline Evaluation" in output_md.read_text(
        encoding="utf-8"
    )
