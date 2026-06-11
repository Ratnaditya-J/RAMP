from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def write_corpus(
    tmp_path: Path,
    *,
    count: int = 32,
) -> tuple[Path, Path, Path]:
    review_csv = tmp_path / "review.csv"
    feature_table = tmp_path / "features.jsonl"
    activation = tmp_path / "activation.jsonl"

    review_rows = []
    feature_rows = []
    activation_rows = []
    for idx in range(count):
        unsafe = idx % 2 == 1
        label = "unsafe" if unsafe else "safe"
        source = "source_a" if idx % 4 < 2 else "source_b"
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
        feature_rows.append(
            {
                "id": row_id,
                "label": label,
                "source": source,
                "domain": "cyber_abuse",
                "subcluster_role": "harmful" if unsafe else "benign_near_neighbor",
                "subcluster_id": "vulnerability_exploitation",
                "span_text": row_id,
                "prompt_risk_score": 0.65 if unsafe else 0.35,
                "embedding_prior_score": 0.60 if unsafe else 0.40,
                "activation_probability": 0.99 if unsafe else 0.01,
            }
        )
        activation_rows.append(
            {
                "id": row_id,
                "span_text": row_id,
                "domain": "cyber_abuse",
                "subcluster_role": "harmful" if unsafe else "benign_near_neighbor",
                "subcluster_id": "vulnerability_exploitation",
                "label": label,
                "source": source,
                "embedding": [1.0, 0.0] if unsafe else [0.0, 1.0],
                "provenance": {"huggingface_model_id": "synthetic"},
            }
        )
    with review_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    feature_table.write_text(
        "\n".join(json.dumps(row) for row in feature_rows) + "\n",
        encoding="utf-8",
    )
    activation.write_text(
        "\n".join(json.dumps(row) for row in activation_rows) + "\n",
        encoding="utf-8",
    )
    return review_csv, feature_table, activation


def test_signal_survival_ladder_runs_all_local_rungs(tmp_path: Path) -> None:
    review_csv, feature_table, activation = write_corpus(tmp_path)
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_signal_survival_ladder.py",
            "--review-csv",
            str(review_csv),
            "--feature-table",
            str(feature_table),
            "--activation",
            str(activation),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--num-splits",
            "2",
            "--calibration-folds",
            "2",
            "--min-shifted-class-rows",
            "2",
            "--weight-step",
            "0.50",
            "--threshold-step",
            "0.50",
            "--epochs",
            "3",
        ],
        check=True,
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["artifact_id"] == "ramp_signal_survival_ladder_v0.1"
    assert report["num_binary_eval_rows"] == 32

    rungs = report["rungs"]
    assert rungs["naive"]["status"] == "completed"
    assert rungs["split"]["status"] == "completed"
    assert rungs["crossfit"]["status"] == "completed"
    assert rungs["blind"]["status"] == "pending"
    assert rungs["shifted"]["status"] == "completed"
    assert sorted(rungs["shifted"]["holdout_sources"]) == ["source_a", "source_b"]

    # The naive rung's probe is in-sample; the crossfit rung must use out-of-fold
    # calibration scores instead.
    assert rungs["naive"]["activation_probe"]["mode"] == "in_sample"
    crossfit_probe = rungs["crossfit"]["aggregate_holdout_metrics"]
    assert "prompt_activation_calibrated" in crossfit_probe

    table = report["survival_table"]
    assert set(table) == {"prompt", "embedding", "activation", "full_fusion"}
    assert table["prompt"]["naive"]["verdict"] == "baseline"
    assert table["activation"]["blind"]["verdict"] == "pending"
    for signal in ("embedding", "activation", "full_fusion"):
        for rung_name in ("naive", "split", "crossfit", "shifted"):
            assert table[signal][rung_name]["verdict"] in {
                "survives",
                "mixed",
                "fails",
            }

    markdown = output_md.read_text(encoding="utf-8")
    assert "# Signal Survival Ladder" in markdown
    assert "## Survival Table" in markdown


def test_signal_survival_ladder_blind_rung_with_blind_csv(tmp_path: Path) -> None:
    review_csv, feature_table, activation = write_corpus(tmp_path)

    # Blind set: a disjoint slice of rows present in the feature table but absent
    # from the adaptive review CSV.
    blind_csv = tmp_path / "blind.csv"
    feature_rows = [
        json.loads(line)
        for line in feature_table.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    activation_rows = [
        json.loads(line)
        for line in activation.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    blind_feature_rows = []
    blind_activation_rows = []
    blind_review_rows = []
    for idx in range(8):
        unsafe = idx % 2 == 1
        label = "unsafe" if unsafe else "safe"
        row_id = f"blind-{idx:02d}"
        blind_review_rows.append(
            {
                "review_id": f"b{idx}",
                "source_id": row_id,
                "review_status": "reviewed",
                "reviewed_label": label,
                "prompt_text": row_id,
            }
        )
        blind_feature_rows.append(
            {
                **feature_rows[idx],
                "id": row_id,
                "span_text": row_id,
                "label": label,
            }
        )
        blind_activation_rows.append(
            {
                **activation_rows[idx],
                "id": row_id,
                "span_text": row_id,
                "label": label,
                "embedding": [1.0, 0.0] if unsafe else [0.0, 1.0],
            }
        )
    feature_table.write_text(
        "\n".join(json.dumps(row) for row in feature_rows + blind_feature_rows) + "\n",
        encoding="utf-8",
    )
    activation.write_text(
        "\n".join(json.dumps(row) for row in activation_rows + blind_activation_rows) + "\n",
        encoding="utf-8",
    )
    with blind_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(blind_review_rows[0]))
        writer.writeheader()
        writer.writerows(blind_review_rows)

    output_json = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_signal_survival_ladder.py",
            "--review-csv",
            str(review_csv),
            "--feature-table",
            str(feature_table),
            "--activation",
            str(activation),
            "--blind-review-csv",
            str(blind_csv),
            "--output-json",
            str(output_json),
            "--rungs",
            "blind",
            "--calibration-folds",
            "2",
            "--weight-step",
            "0.50",
            "--threshold-step",
            "0.50",
            "--epochs",
            "3",
        ],
        check=True,
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    blind = report["rungs"]["blind"]
    assert blind["status"] == "completed"
    assert blind["blind_rows"] == 8
    assert report["survival_table"]["activation"]["blind"]["verdict"] in {
        "survives",
        "mixed",
        "fails",
    }
