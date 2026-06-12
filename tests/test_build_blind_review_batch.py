from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    feature_table = tmp_path / "features.jsonl"
    prior_review = tmp_path / "prior_review.csv"

    feature_rows = []
    for idx in range(40):
        source = "source_a" if idx % 2 == 0 else "source_b"
        domain = "cyber_abuse" if idx % 4 < 2 else "regulated_advice"
        feature_rows.append(
            {
                "id": f"row-{idx:02d}",
                "span_text": f"prompt text {idx}",
                "source": source,
                "domain": domain,
                "label": "unsafe" if idx % 3 == 0 else "safe",
                "prompt_risk_score": 0.9,
                "embedding_prior_score": 0.8,
                "activation_probability": 0.7,
            }
        )
    feature_table.write_text(
        "\n".join(json.dumps(row) for row in feature_rows) + "\n",
        encoding="utf-8",
    )

    prior_rows = [
        {"review_id": f"r{idx}", "source_id": f"row-{idx:02d}", "reviewed_label": "safe"}
        for idx in range(10)
    ]
    with prior_review.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(prior_rows[0]))
        writer.writeheader()
        writer.writerows(prior_rows)
    return feature_table, prior_review


def run_sampler(
    feature_table: Path,
    prior_review: Path,
    output_csv: Path,
    output_manifest: Path,
    *,
    seed: str = "test_seed",
) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/build_blind_review_batch.py",
            "--feature-table",
            str(feature_table),
            "--exclude-review-csv",
            str(prior_review),
            "--max-rows",
            "12",
            "--seed",
            seed,
            "--output-csv",
            str(output_csv),
            "--output-manifest",
            str(output_manifest),
        ],
        check=True,
    )


def test_blind_batch_excludes_reviewed_rows_and_hides_scores(tmp_path: Path) -> None:
    feature_table, prior_review = write_inputs(tmp_path)
    output_csv = tmp_path / "blind.csv"
    output_manifest = tmp_path / "blind.manifest.json"
    run_sampler(feature_table, prior_review, output_csv, output_manifest)

    with output_csv.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    assert len(rows) == 12
    excluded = {f"row-{idx:02d}" for idx in range(10)}
    assert not excluded & {row["source_id"] for row in rows}

    # Blinding: no score, bucket, domain, label, or severity columns.
    forbidden = {
        "prompt_risk_score",
        "embedding_prior_score",
        "activation_probability",
        "label",
        "domain",
        "bucket",
        "severity",
        "ramp_fusion_score",
    }
    assert not forbidden & set(fieldnames)
    assert {"source_id", "prompt_text", "reviewed_label"} <= set(fieldnames)

    manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert manifest["summary"]["sampled_rows"] == 12
    assert manifest["summary"]["excluded_ids"] == 10
    assert len(manifest["sampled"]) == 12
    assert manifest["sampled_ids_sha256"]


def test_blind_batch_is_deterministic_and_stratified(tmp_path: Path) -> None:
    feature_table, prior_review = write_inputs(tmp_path)
    first_csv = tmp_path / "blind1.csv"
    second_csv = tmp_path / "blind2.csv"
    run_sampler(feature_table, prior_review, first_csv, tmp_path / "m1.json")
    run_sampler(feature_table, prior_review, second_csv, tmp_path / "m2.json")
    assert first_csv.read_text(encoding="utf-8") == second_csv.read_text(encoding="utf-8")

    manifest = json.loads((tmp_path / "m1.json").read_text(encoding="utf-8"))
    allocation = manifest["summary"]["allocation"]
    # 30 eligible rows over 4 equal-ish cells sampled down to 12: every cell
    # keeps a proportional share (no cell starved, no cell dominant).
    assert len(allocation) == 4
    assert all(2 <= count <= 4 for count in allocation.values())
