#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_reviewed_cumulative_signals import (
    binary_label,
    load_reviewed_rows,
)
from train_activation_probes import (
    auc,
    compact_artifact,
    layer_report,
    load_activation_rows,
    make_indices,
    require_numpy,
    train_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a linear activation probe using human-reviewed labels."
    )
    parser.add_argument("--review-csv", required=True, help="Reviewed prompt-label CSV.")
    parser.add_argument("--activation", required=True, help="Activation JSONL.")
    parser.add_argument("--layer", required=True, help="Layer id for the activation JSONL.")
    parser.add_argument("--output", required=True, help="Output probe artifact JSON.")
    parser.add_argument("--report-output", required=True, help="Output report JSON.")
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.10)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def reviewed_activation_rows(
    *,
    review_csv: Path,
    activation_path: Path,
) -> tuple[list[dict[str, Any]], list[list[float]], list[int]]:
    reviewed = load_reviewed_rows(review_csv)
    rows, vectors, _ = load_activation_rows(activation_path, max_records=None)
    output_rows: list[dict[str, Any]] = []
    output_vectors: list[list[float]] = []
    output_labels: list[int] = []
    for row, vector in zip(rows, vectors, strict=True):
        row_id = str(row.get("id"))
        review = reviewed.get(row_id)
        if not review:
            continue
        label = binary_label(str(review.get("reviewed_label")))
        if label is None:
            continue
        corrected = dict(row)
        corrected["label"] = "unsafe" if label == 1 else "safe"
        corrected["reviewed_label"] = review.get("reviewed_label")
        corrected["review_id"] = review.get("review_id")
        corrected["prompt_text"] = review.get("prompt_text", row.get("span_text"))
        output_rows.append(corrected)
        output_vectors.append(vector)
        output_labels.append(label)
    return output_rows, output_vectors, output_labels


def main() -> None:
    args = parse_args()
    np = require_numpy()
    activation_path = Path(args.activation)
    rows, vectors, labels = reviewed_activation_rows(
        review_csv=Path(args.review_csv),
        activation_path=activation_path,
    )
    if not rows:
        raise ValueError("no reviewed activation rows found")
    train_indices, test_indices = make_indices(
        labels,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    probe = train_probe(
        np,
        vectors,
        labels,
        train_indices=train_indices,
        test_indices=test_indices,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    artifact = compact_artifact(
        layer_id=args.layer,
        activation_path=activation_path,
        probe=probe,
        rows=rows,
        train_indices=train_indices,
        test_indices=test_indices,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )
    artifact["artifact_id"] = f"ramp_reviewed_activation_probe_layer_{args.layer}_v0.1"
    artifact["model_version"] = "linear_reviewed_activation_probe_v0.1"
    artifact["review_csv"] = args.review_csv

    report = layer_report(
        layer_id=args.layer,
        rows=rows,
        labels=labels,
        train_indices=train_indices,
        test_indices=test_indices,
        probe=probe,
    )
    report["artifact_path"] = args.output
    report["review_csv"] = args.review_csv
    report["activation_path"] = args.activation
    report["reviewed_rows"] = len(rows)
    report["holdout_auc"] = auc(
        [int(value) for value in probe["test_labels"].tolist()],
        probe["test_probabilities"].tolist(),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote reviewed activation probe to {output_path}")
    print(f"wrote reviewed activation probe report to {report_path}")
    print(
        f"reviewed activation probe auc={report['holdout_auc']:.4f} "
        f"recall@5fpr={report['test_metrics']['target_fpr_0_05']['recall']:.4f}"
    )


if __name__ == "__main__":
    main()
