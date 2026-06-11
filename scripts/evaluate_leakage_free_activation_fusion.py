#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from evaluate_calibrated_signal_combinations import (
    aggregate,
    calibrate_combination,
    combinations,
    evaluate_combination,
    markdown_report,
    stratified_split,
)
from evaluate_reviewed_cumulative_signals import (
    binary_label,
    join_rows,
    load_feature_table,
    load_reviewed_rows,
)
from evaluate_severity_floors import DEFAULT_HIGH_SEVERITY_CATEGORIES, normalize_category
from train_activation_probes import auc, metrics, require_numpy, threshold_sweep, train_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate signal fusion with leakage-free activation probabilities. "
            "For each split, the activation probe is trained only on calibration rows "
            "and scored only against that split's holdout rows."
        )
    )
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--feature-table", required=True)
    parser.add_argument("--activation", required=True)
    parser.add_argument("--layer", default="19")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--num-splits", type=int, default=30)
    parser.add_argument("--seed-prefix", default="ramp_leakage_free_activation_v0.1")
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--min-prompt-weight", type=float, default=0.40)
    parser.add_argument("--max-embedding-weight", type=float, default=0.20)
    parser.add_argument("--max-output-weight", type=float, default=1.0)
    parser.add_argument("--require-prompt-gte-activation", action="store_true")
    parser.add_argument("--include-output", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.10)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument(
        "--calibration-folds",
        type=int,
        default=5,
        help=(
            "Number of folds for out-of-fold activation predictions inside each "
            "calibration split. Use 1 to keep in-sample calibration predictions."
        ),
    )
    parser.add_argument("--top-k-errors", type=int, default=50)
    return parser.parse_args()


def load_activation_vectors(path: Path) -> dict[str, list[float]]:
    vectors: dict[str, list[float]] = {}
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row.get("id")
            if row_id is None:
                continue
            vectors[str(row_id)] = [float(value) for value in row["embedding"]]
    return vectors


def binary_feature_rows(
    rows: list[dict[str, Any]],
    *,
    activation_vectors: dict[str, list[float]],
    include_output: bool,
) -> list[dict[str, Any]]:
    required = {"prompt_risk_score", "embedding_prior_score"}
    if include_output:
        required.add("output_risk_score")
    output = []
    for row in rows:
        row_id = str(row.get("source_id") or row.get("id"))
        if binary_label(str(row.get("reviewed_label"))) is None:
            continue
        if row_id not in activation_vectors:
            continue
        if any(row.get(key) is None for key in required):
            continue
        copied = dict(row)
        copied["source_id"] = row_id
        copied.pop("activation_probability", None)
        output.append(copied)
    return output


def attach_split_activation_scores(
    np: Any,
    calibration_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    activation_vectors: dict[str, list[float]],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    calibration_folds: int,
    seed: str,
) -> dict[str, Any]:
    calibration_labels = [
        int(binary_label(str(row["reviewed_label"]))) for row in calibration_rows
    ]
    calibration_vectors = [
        activation_vectors[str(row["source_id"])] for row in calibration_rows
    ]
    crossfit_report = attach_crossfit_calibration_scores(
        np,
        calibration_rows,
        calibration_vectors,
        calibration_labels,
        folds=calibration_folds,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed,
    )

    rows = calibration_rows + holdout_rows
    labels = [int(binary_label(str(row["reviewed_label"]))) for row in rows]
    vectors = [activation_vectors[str(row["source_id"])] for row in rows]
    train_indices = list(range(len(calibration_rows)))
    test_indices = list(range(len(calibration_rows), len(rows)))
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
    test_probs = [float(value) for value in probe["test_probabilities"].tolist()]
    for row, probability in zip(holdout_rows, test_probs, strict=True):
        row["activation_probability"] = probability
    return {
        "train_rows": len(train_indices),
        "holdout_rows": len(test_indices),
        "train_metrics": probe["train_metrics"],
        "holdout_metrics": probe["test_metrics"],
        "calibration_crossfit": crossfit_report,
        "selected_threshold": float(probe["selected_threshold"]),
    }


def stable_fold_key(row: dict[str, Any], seed: str) -> str:
    row_id = str(row.get("source_id") or row.get("id") or row.get("review_id"))
    return hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest()


def stratified_fold_indices(
    rows: list[dict[str, Any]],
    labels: list[int],
    *,
    folds: int,
    seed: str,
) -> list[list[int]]:
    output = [[] for _ in range(folds)]
    for label in sorted(set(labels)):
        indices = [idx for idx, value in enumerate(labels) if value == label]
        indices = sorted(indices, key=lambda idx: stable_fold_key(rows[idx], seed))
        for offset, idx in enumerate(indices):
            output[offset % folds].append(idx)
    return [sorted(fold) for fold in output if fold]


def attach_crossfit_calibration_scores(
    np: Any,
    calibration_rows: list[dict[str, Any]],
    calibration_vectors: list[list[float]],
    calibration_labels: list[int],
    *,
    folds: int,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: str,
) -> dict[str, Any]:
    if folds <= 1:
        probe = train_probe(
            np,
            calibration_vectors,
            calibration_labels,
            train_indices=list(range(len(calibration_rows))),
            test_indices=list(range(len(calibration_rows))),
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        probabilities = [float(value) for value in probe["test_probabilities"].tolist()]
        for row, probability in zip(calibration_rows, probabilities, strict=True):
            row["activation_probability"] = probability
        return {
            "mode": "in_sample",
            "folds": 1,
            "auc": auc(calibration_labels, probabilities),
            "thresholds": threshold_sweep(
                np.asarray(calibration_labels, dtype=np.float32),
                np.asarray(probabilities, dtype=np.float32),
            ),
        }

    smallest_class_count = min(
        calibration_labels.count(0),
        calibration_labels.count(1),
    )
    effective_folds = max(2, min(folds, smallest_class_count))
    fold_indices = stratified_fold_indices(
        calibration_rows,
        calibration_labels,
        folds=effective_folds,
        seed=seed,
    )
    probabilities: list[float | None] = [None] * len(calibration_rows)
    fold_reports = []
    all_indices = set(range(len(calibration_rows)))
    for fold_index, test_indices in enumerate(fold_indices):
        train_indices = sorted(all_indices - set(test_indices))
        probe = train_probe(
            np,
            calibration_vectors,
            calibration_labels,
            train_indices=train_indices,
            test_indices=test_indices,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        test_probabilities = [float(value) for value in probe["test_probabilities"].tolist()]
        for idx, probability in zip(test_indices, test_probabilities, strict=True):
            probabilities[idx] = probability
        fold_reports.append(
            {
                "fold_index": fold_index,
                "train_rows": len(train_indices),
                "calibration_rows": len(test_indices),
                "metrics": probe["test_metrics"],
            }
        )
    if any(value is None for value in probabilities):
        raise RuntimeError("cross-fit activation scoring failed to fill all calibration rows")
    resolved = [float(value) for value in probabilities if value is not None]
    for row, probability in zip(calibration_rows, resolved, strict=True):
        row["activation_probability"] = probability
    label_array = np.asarray(calibration_labels, dtype=np.float32)
    probability_array = np.asarray(resolved, dtype=np.float32)
    return {
        "mode": "out_of_fold",
        "folds": effective_folds,
        "auc": auc(calibration_labels, resolved),
        "thresholds": threshold_sweep(label_array, probability_array),
        "default_0_5": metrics(label_array, probability_array),
        "fold_reports": fold_reports,
    }


def build_report(
    rows: list[dict[str, Any]],
    activation_vectors: dict[str, list[float]],
    *,
    layer: str,
    num_splits: int,
    seed_prefix: str,
    calibration_fraction: float,
    weight_step: float,
    threshold_step: float,
    min_prompt_weight: float,
    max_embedding_weight: float,
    max_output_weight: float,
    require_prompt_gte_activation: bool,
    include_output: bool,
    epochs: int,
    learning_rate: float,
    l2: float,
    calibration_folds: int,
    top_k_errors: int,
) -> dict[str, Any]:
    np = require_numpy()
    combo_map = combinations(include_output)
    eval_rows = binary_feature_rows(
        rows,
        activation_vectors=activation_vectors,
        include_output=include_output,
    )
    high_severity_categories = {
        normalize_category(category) for category in DEFAULT_HIGH_SEVERITY_CATEGORIES
    }
    split_reports = []
    for split_index in range(num_splits):
        calibration_rows, holdout_rows = stratified_split(
            eval_rows,
            calibration_fraction=calibration_fraction,
            seed=f"{seed_prefix}:{split_index:03d}",
        )
        calibration_rows = [dict(row) for row in calibration_rows]
        holdout_rows = [dict(row) for row in holdout_rows]
        probe_report = attach_split_activation_scores(
            np,
            calibration_rows,
            holdout_rows,
            activation_vectors,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            calibration_folds=calibration_folds,
            seed=f"{seed_prefix}:{split_index:03d}:activation_crossfit",
        )
        calibrations = {
            name: calibrate_combination(
                calibration_rows,
                keys,
                weight_step=weight_step,
                threshold_step=threshold_step,
                min_prompt_weight=min_prompt_weight,
                max_embedding_weight=max_embedding_weight,
                max_output_weight=max_output_weight,
                require_prompt_gte_activation=require_prompt_gte_activation,
            )
            for name, keys in combo_map.items()
        }
        holdout = {
            name: evaluate_combination(
                holdout_rows,
                name,
                calibration,
                high_severity_categories=high_severity_categories,
                top_k_errors=top_k_errors,
            )
            for name, calibration in calibrations.items()
        }
        split_reports.append(
            {
                "split_index": split_index,
                "calibration_rows": len(calibration_rows),
                "holdout_rows": len(holdout_rows),
                "activation_probe": probe_report,
                "calibrations": calibrations,
                "holdout": holdout,
            }
        )
    return {
        "artifact_id": "ramp_leakage_free_activation_fusion_v0.1",
        "layer": layer,
        "num_binary_eval_rows": len(eval_rows),
        "num_splits": num_splits,
        "include_output": include_output,
        "activation_probe_protocol": (
            "Each split trains a fresh activation probe on calibration rows only. "
            "Calibration activation_probability values are out-of-fold predictions "
            "inside the calibration split, and holdout values are out-of-split "
            "predictions from a probe trained on the full calibration split."
        ),
        "activation_probe_training": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "calibration_folds": calibration_folds,
        },
        "combinations": {name: list(keys) for name, keys in combo_map.items()},
        "aggregate_holdout_metrics": aggregate(split_reports, combo_map),
        "split_reports": split_reports,
    }


def main() -> None:
    args = parse_args()
    reviewed = load_reviewed_rows(Path(args.review_csv))
    features = load_feature_table(Path(args.feature_table))
    rows = join_rows(reviewed, features)
    activation_vectors = load_activation_vectors(Path(args.activation))
    report = build_report(
        rows,
        activation_vectors,
        layer=args.layer,
        num_splits=args.num_splits,
        seed_prefix=args.seed_prefix,
        calibration_fraction=args.calibration_fraction,
        weight_step=args.weight_step,
        threshold_step=args.threshold_step,
        min_prompt_weight=args.min_prompt_weight,
        max_embedding_weight=args.max_embedding_weight,
        max_output_weight=args.max_output_weight,
        require_prompt_gte_activation=args.require_prompt_gte_activation,
        include_output=args.include_output,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        calibration_folds=args.calibration_folds,
        top_k_errors=args.top_k_errors,
    )
    report["review_csv"] = args.review_csv
    report["feature_table"] = args.feature_table
    report["activation"] = args.activation
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote leakage-free activation fusion report to {output_json}")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown_report(report) + "\n")
        print(f"wrote leakage-free activation fusion markdown to {output_md}")
    for name, metric_summary in report["aggregate_holdout_metrics"].items():
        print(
            f"{name}: auc={metric_summary['auc']['mean']:.4f} "
            f"recall={metric_summary['recall']['mean']:.4f} "
            f"fpr={metric_summary['false_positive_rate']['mean']:.4f}"
        )


if __name__ == "__main__":
    main()
