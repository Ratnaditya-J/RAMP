#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from train_activation_probes import (
    auc,
    label_to_binary,
    load_activation_rows,
    metrics,
    require_numpy,
    threshold_sweep,
    train_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate activation probes with source-held-out or domain-held-out splits."
    )
    parser.add_argument("--activation", required=True, help="Activation JSONL for one layer.")
    parser.add_argument("--layer", required=True, help="Layer id, e.g. 19.")
    parser.add_argument(
        "--holdout-key",
        choices=["source", "domain"],
        required=True,
        help="Metadata field used for held-out validation.",
    )
    parser.add_argument(
        "--holdout-value",
        action="append",
        default=None,
        help="Specific value to hold out. Repeatable. Defaults to all values with both labels.",
    )
    parser.add_argument("--output", required=True, help="Output JSON report.")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.10)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def candidate_holdouts(
    rows: list[dict[str, Any]],
    *,
    holdout_key: str,
    requested: list[str] | None,
) -> list[str]:
    if requested:
        return requested
    labels_by_value: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        value = str(row.get(holdout_key, "unknown"))
        labels_by_value[value][str(row.get("label", "unknown"))] += 1
    return [value for value, labels in sorted(labels_by_value.items()) if sum(labels.values()) > 0]


def make_holdout_indices(
    rows: list[dict[str, Any]],
    *,
    holdout_key: str,
    holdout_value: str,
) -> tuple[list[int], list[int]]:
    train_indices = []
    test_indices = []
    for idx, row in enumerate(rows):
        if str(row.get(holdout_key, "unknown")) == holdout_value:
            test_indices.append(idx)
        else:
            train_indices.append(idx)
    return train_indices, test_indices


def labels_for_indices(labels: list[int], indices: list[int]) -> Counter[str]:
    return Counter("unsafe" if labels[idx] == 1 else "safe" for idx in indices)


def validate_split(
    np: Any,
    *,
    rows: list[dict[str, Any]],
    vectors: list[list[float]],
    labels: list[int],
    train_indices: list[int],
    test_indices: list[int],
    holdout_key: str,
    holdout_value: str,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> dict[str, Any]:
    train_labels = labels_for_indices(labels, train_indices)
    test_labels = labels_for_indices(labels, test_indices)
    if len(train_labels) < 2 or not test_labels:
        return {
            "holdout_key": holdout_key,
            "holdout_value": holdout_value,
            "skipped": True,
            "reason": "train split must contain both labels and test split must be non-empty",
            "train_rows": len(train_indices),
            "test_rows": len(test_indices),
            "train_labels": dict(train_labels),
            "test_labels": dict(test_labels),
        }

    probe = train_probe(
        np,
        vectors,
        labels,
        train_indices=train_indices,
        test_indices=test_indices,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    test_labels_array = probe["test_labels"]
    test_probabilities = probe["test_probabilities"]
    selected = probe["train_metrics"]["target_fpr_0_05"]
    selected_threshold = float(selected["threshold"])
    selected_test_metrics = metrics(
        test_labels_array,
        test_probabilities,
        threshold=selected_threshold,
    )

    domain_metrics: dict[str, Any] = {}
    for domain in sorted({str(rows[idx].get("domain", "unknown")) for idx in test_indices}):
        indices = [
            local_idx
            for local_idx, row_idx in enumerate(test_indices)
            if str(rows[row_idx].get("domain", "unknown")) == domain
        ]
        if not indices:
            continue
        domain_metrics[domain] = metrics(
            test_labels_array[indices],
            test_probabilities[indices],
            threshold=selected_threshold,
        )

    hard_benign = [
        {
            "id": rows[row_idx].get("id"),
            "source": rows[row_idx].get("source"),
            "domain": rows[row_idx].get("domain"),
            "subcluster_id": rows[row_idx].get("subcluster_id"),
            "unsafe_probability": float(probability),
            "span_text": rows[row_idx].get("span_text"),
        }
        for row_idx, label, probability in sorted(
            zip(test_indices, test_labels_array.tolist(), test_probabilities.tolist(), strict=True),
            key=lambda item: item[2],
            reverse=True,
        )
        if int(label) == 0
    ][:20]

    unsafe_low = [
        {
            "id": rows[row_idx].get("id"),
            "source": rows[row_idx].get("source"),
            "domain": rows[row_idx].get("domain"),
            "subcluster_id": rows[row_idx].get("subcluster_id"),
            "unsafe_probability": float(probability),
            "span_text": rows[row_idx].get("span_text"),
        }
        for row_idx, label, probability in sorted(
            zip(test_indices, test_labels_array.tolist(), test_probabilities.tolist(), strict=True),
            key=lambda item: item[2],
        )
        if int(label) == 1
    ][:20]

    return {
        "holdout_key": holdout_key,
        "holdout_value": holdout_value,
        "skipped": False,
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
        "train_labels": dict(train_labels),
        "test_labels": dict(test_labels),
        "auc": auc(
            [int(value) for value in test_labels_array.tolist()],
            test_probabilities.tolist(),
        ),
        "selected_threshold": selected_threshold,
        "threshold_source": "train_target_fpr_0_05",
        "train_metrics": probe["train_metrics"],
        "test_sweep_metrics": probe["test_metrics"],
        "test_metrics": {
            "selected_threshold": selected_test_metrics,
            "best_f1": probe["test_metrics"]["best_f1"],
            "best_balanced_accuracy": probe["test_metrics"]["best_balanced_accuracy"],
            "target_fpr_0_05": probe["test_metrics"]["target_fpr_0_05"],
        },
        "domain_metrics": domain_metrics,
        "hard_benign_near_neighbors": hard_benign,
        "unsafe_low_probability": unsafe_low,
    }


def main() -> None:
    args = parse_args()
    np = require_numpy()
    activation_path = Path(args.activation)
    rows, vectors, labels = load_activation_rows(activation_path, max_records=args.max_records)
    if not rows:
        raise ValueError(f"no labeled rows found in {activation_path}")

    # Keep this import-used behavior explicit for scripts run from arbitrary cwd.
    _ = label_to_binary, threshold_sweep

    reports = []
    for holdout_value in candidate_holdouts(
        rows,
        holdout_key=args.holdout_key,
        requested=args.holdout_value,
    ):
        train_indices, test_indices = make_holdout_indices(
            rows,
            holdout_key=args.holdout_key,
            holdout_value=holdout_value,
        )
        report = validate_split(
            np,
            rows=rows,
            vectors=vectors,
            labels=labels,
            train_indices=train_indices,
            test_indices=test_indices,
            holdout_key=args.holdout_key,
            holdout_value=holdout_value,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )
        reports.append(report)
        if report["skipped"]:
            print(f"{args.holdout_key}={holdout_value}: skipped {report['reason']}")
        else:
            metric = report["test_metrics"]["selected_threshold"]
            auc_text = f"{report['auc']:.4f}" if report["auc"] is not None else "n/a"
            print(
                f"{args.holdout_key}={holdout_value}: auc={auc_text} "
                f"recall={metric['recall']:.4f} fpr={metric['false_positive_rate']:.4f}"
            )

    valid_reports = [report for report in reports if not report["skipped"]]
    output = {
        "layer_id": args.layer,
        "activation_path": str(activation_path),
        "holdout_key": args.holdout_key,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "seed": args.seed,
        "num_reports": len(reports),
        "num_valid_reports": len(valid_reports),
        "mean_auc": (
            sum(float(report["auc"]) for report in valid_reports if report["auc"] is not None)
            / len([report for report in valid_reports if report["auc"] is not None])
            if [report for report in valid_reports if report["auc"] is not None]
            else None
        ),
        "mean_recall_at_train_selected_threshold": (
            sum(
                float(report["test_metrics"]["selected_threshold"]["recall"])
                for report in valid_reports
            )
            / len(valid_reports)
            if valid_reports
            else None
        ),
        "reports": reports,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote holdout validation report to {output_path}")


if __name__ == "__main__":
    main()
