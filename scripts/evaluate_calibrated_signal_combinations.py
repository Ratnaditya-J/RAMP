#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluate_ramp_harness import evaluate_named_scores
from evaluate_reviewed_cumulative_signals import (
    binary_label,
    join_rows,
    load_feature_table,
    load_reviewed_rows,
)
from evaluate_severity_floors import DEFAULT_HIGH_SEVERITY_CATEGORIES, normalize_category

COMBINATIONS = {
    "prompt_only_calibrated": ("prompt_risk_score",),
    "prompt_embedding_calibrated": ("prompt_risk_score", "embedding_prior_score"),
    "prompt_activation_calibrated": ("prompt_risk_score", "activation_probability"),
    "prompt_embedding_activation_calibrated": (
        "prompt_risk_score",
        "embedding_prior_score",
        "activation_probability",
    ),
}

OUTPUT_COMBINATIONS = {
    "prompt_output_calibrated": ("prompt_risk_score", "output_risk_score"),
    "prompt_activation_output_calibrated": (
        "prompt_risk_score",
        "activation_probability",
        "output_risk_score",
    ),
    "prompt_embedding_activation_output_calibrated": (
        "prompt_risk_score",
        "embedding_prior_score",
        "activation_probability",
        "output_risk_score",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate signal combinations fairly across repeated reviewed splits."
    )
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--feature-table", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--num-splits", type=int, default=30)
    parser.add_argument("--seed-prefix", default="ramp_combination_stability_v0.1")
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--min-prompt-weight", type=float, default=0.40)
    parser.add_argument("--max-embedding-weight", type=float, default=0.20)
    parser.add_argument("--max-output-weight", type=float, default=1.0)
    parser.add_argument("--require-prompt-gte-activation", action="store_true")
    parser.add_argument(
        "--include-output",
        action="store_true",
        help=(
            "Include output_risk_score combinations. Requires output_risk_score "
            "in the feature table."
        ),
    )
    parser.add_argument("--top-k-errors", type=int, default=50)
    return parser.parse_args()


def combinations(include_output: bool) -> dict[str, tuple[str, ...]]:
    output = dict(COMBINATIONS)
    if include_output:
        output.update(OUTPUT_COMBINATIONS)
    return output


def binary_feature_rows(
    rows: list[dict[str, Any]],
    *,
    include_output: bool,
) -> list[dict[str, Any]]:
    required = {"prompt_risk_score", "embedding_prior_score", "activation_probability"}
    if include_output:
        required.add("output_risk_score")
    return [
        row
        for row in rows
        if binary_label(str(row.get("reviewed_label"))) is not None
        and all(row.get(key) is not None for key in required)
    ]


def split_sort_key(row: dict[str, Any], seed: str) -> str:
    row_id = str(row.get("source_id") or row.get("id") or row.get("review_id"))
    return hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest()


def stratified_split(
    rows: list[dict[str, Any]],
    *,
    calibration_fraction: float,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("reviewed_label"))].append(row)
    calibration_rows = []
    holdout_rows = []
    for group_rows in grouped.values():
        ordered = sorted(group_rows, key=lambda row: split_sort_key(row, seed))
        calibration_count = round(len(ordered) * calibration_fraction)
        calibration_count = min(max(calibration_count, 1), len(ordered) - 1)
        calibration_rows.extend(ordered[:calibration_count])
        holdout_rows.extend(ordered[calibration_count:])
    return calibration_rows, holdout_rows


def values_from_step(step: float) -> list[float]:
    count = round(1.0 / step)
    return [round(idx * step, 10) for idx in range(count + 1)]


def weight_candidates(
    keys: tuple[str, ...],
    *,
    weight_step: float,
    min_prompt_weight: float,
    max_embedding_weight: float,
    max_output_weight: float,
    require_prompt_gte_activation: bool,
) -> list[dict[str, float]]:
    if len(keys) == 1:
        return [{keys[0]: 1.0}]
    candidates = []
    values = values_from_step(weight_step)
    if len(keys) == 2:
        first, second = keys
        for first_weight in values:
            second_weight = round(1.0 - first_weight, 10)
            weights = {first: first_weight, second: second_weight}
            if weights.get("prompt_risk_score", 0.0) < min_prompt_weight:
                continue
            if weights.get("embedding_prior_score", 0.0) > max_embedding_weight:
                continue
            if weights.get("output_risk_score", 0.0) > max_output_weight:
                continue
            if (
                require_prompt_gte_activation
                and weights.get("prompt_risk_score", 0.0)
                < weights.get("activation_probability", 0.0)
            ):
                continue
            candidates.append(weights)
        return candidates
    non_prompt_keys = [key for key in keys if key != "prompt_risk_score"]

    def expand_weights(
        remaining_keys: list[str],
        remaining_weight: float,
        current: dict[str, float],
    ) -> None:
        if not remaining_keys:
            if abs(remaining_weight) > 1e-9:
                return
            weights = dict(current)
            if weights.get("prompt_risk_score", 0.0) < min_prompt_weight:
                return
            if weights.get("embedding_prior_score", 0.0) > max_embedding_weight:
                return
            if weights.get("output_risk_score", 0.0) > max_output_weight:
                return
            if (
                require_prompt_gte_activation
                and weights.get("prompt_risk_score", 0.0)
                < weights.get("activation_probability", 0.0)
            ):
                return
            candidates.append(weights)
            return
        key = remaining_keys[0]
        if len(remaining_keys) == 1:
            value_options = [round(remaining_weight, 10)]
        else:
            value_options = [value for value in values if value <= remaining_weight + 1e-9]
        for value in value_options:
            current[key] = round(value, 10)
            expand_weights(
                remaining_keys[1:],
                round(remaining_weight - value, 10),
                current,
            )
            current.pop(key, None)

    for prompt_weight in values:
        if prompt_weight > 1.0:
            continue
        current = {"prompt_risk_score": prompt_weight} if "prompt_risk_score" in keys else {}
        expand_weights(non_prompt_keys, round(1.0 - prompt_weight, 10), current)
    return candidates


def weighted_score(row: dict[str, Any], weights: dict[str, float]) -> float:
    return sum(float(row[key]) * weight for key, weight in weights.items())


def calibrate_combination(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
    *,
    weight_step: float,
    threshold_step: float,
    min_prompt_weight: float,
    max_embedding_weight: float,
    max_output_weight: float,
    require_prompt_gte_activation: bool,
) -> dict[str, Any]:
    labels = [int(binary_label(str(row["reviewed_label"]))) for row in rows]
    thresholds = values_from_step(threshold_step)
    best: dict[str, Any] | None = None
    for weights in weight_candidates(
        keys,
        weight_step=weight_step,
        min_prompt_weight=min_prompt_weight,
        max_embedding_weight=max_embedding_weight,
        max_output_weight=max_output_weight,
        require_prompt_gte_activation=require_prompt_gte_activation,
    ):
        scores = [weighted_score(row, weights) for row in rows]
        for threshold in thresholds:
            report = evaluate_named_scores(
                rows,
                "calibration",
                scores,
                threshold=threshold,
                high_severity_categories=set(DEFAULT_HIGH_SEVERITY_CATEGORIES),
                top_k_errors=0,
            )
            metric = report["metrics"]
            candidate = {
                "weights": weights,
                "threshold": threshold,
                "auc": report["auc"],
                "metrics": metric,
                "objective": (
                    metric["f1"],
                    metric["accuracy"],
                    metric["recall"],
                    -metric["false_positive_rate"],
                    report["auc"] or 0.0,
                ),
            }
            if best is None or candidate["objective"] > best["objective"]:
                best = candidate
    if best is None:
        raise ValueError(f"no candidates for {keys}")
    best["num_rows"] = len(rows)
    best["labels"] = {"safe": labels.count(0), "unsafe": labels.count(1)}
    return best


def evaluate_combination(
    rows: list[dict[str, Any]],
    name: str,
    calibration: dict[str, Any],
    *,
    high_severity_categories: set[str],
    top_k_errors: int,
) -> dict[str, Any]:
    scores = [weighted_score(row, calibration["weights"]) for row in rows]
    report = evaluate_named_scores(
        rows,
        name,
        scores,
        threshold=float(calibration["threshold"]),
        high_severity_categories=high_severity_categories,
        top_k_errors=top_k_errors,
    )
    report["weights"] = calibration["weights"]
    return report


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "stdev": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def metric_value(report: dict[str, Any], metric_name: str) -> float | None:
    if metric_name == "auc":
        return None if report.get("auc") is None else float(report["auc"])
    if metric_name in report["metrics"]:
        return float(report["metrics"][metric_name])
    if metric_name in report["error_counts"]:
        return float(report["error_counts"][metric_name])
    return None


def aggregate(
    splits: list[dict[str, Any]],
    combo_map: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    metric_names = (
        "auc",
        "accuracy",
        "precision",
        "recall",
        "false_positive_rate",
        "f1",
        "fp",
        "fn",
        "severe_false_negatives",
        "hard_benign_false_positives",
    )
    output: dict[str, Any] = {}
    for name in combo_map:
        output[name] = {}
        for metric_name in metric_names:
            values = [
                value
                for split in splits
                if (value := metric_value(split["holdout"][name], metric_name)) is not None
            ]
            output[name][metric_name] = numeric_summary(values)
    return output


def build_report(
    rows: list[dict[str, Any]],
    *,
    num_splits: int,
    seed_prefix: str,
    calibration_fraction: float,
    weight_step: float,
    threshold_step: float,
    min_prompt_weight: float,
    max_embedding_weight: float,
    max_output_weight: float,
    require_prompt_gte_activation: bool,
    top_k_errors: int,
    include_output: bool,
) -> dict[str, Any]:
    combo_map = combinations(include_output)
    eval_rows = binary_feature_rows(rows, include_output=include_output)
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
                "calibrations": calibrations,
                "holdout": holdout,
            }
        )
    return {
        "artifact_id": "ramp_calibrated_signal_combination_stability_v0.1",
        "num_binary_eval_rows": len(eval_rows),
        "num_splits": num_splits,
        "include_output": include_output,
        "combinations": {name: list(keys) for name, keys in combo_map.items()},
        "aggregate_holdout_metrics": aggregate(split_reports, combo_map),
        "split_reports": split_reports,
    }


def summary_cell(summary: dict[str, Any], precision: int = 4) -> str:
    mean = summary.get("mean")
    stdev = summary.get("stdev")
    if mean is None:
        return "n/a"
    return f"{mean:.{precision}f} +/- {stdev:.{precision}f}"


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Calibrated Signal Combination Stability",
        "",
        f"Splits: {report['num_splits']}",
        f"Binary rows: {report['num_binary_eval_rows']}",
        "",
        "| Condition | AUC | Accuracy | Recall | FPR | FP | FN | Severe FN | Hard benign FP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in report["aggregate_holdout_metrics"].items():
        lines.append(
            f"| `{name}` | {summary_cell(metrics['auc'])} | "
            f"{summary_cell(metrics['accuracy'])} | {summary_cell(metrics['recall'])} | "
            f"{summary_cell(metrics['false_positive_rate'])} | "
            f"{summary_cell(metrics['fp'], 2)} | {summary_cell(metrics['fn'], 2)} | "
            f"{summary_cell(metrics['severe_false_negatives'], 2)} | "
            f"{summary_cell(metrics['hard_benign_false_positives'], 2)} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    reviewed = load_reviewed_rows(Path(args.review_csv))
    features = load_feature_table(Path(args.feature_table))
    rows = join_rows(reviewed, features)
    report = build_report(
        rows,
        num_splits=args.num_splits,
        seed_prefix=args.seed_prefix,
        calibration_fraction=args.calibration_fraction,
        weight_step=args.weight_step,
        threshold_step=args.threshold_step,
        min_prompt_weight=args.min_prompt_weight,
        max_embedding_weight=args.max_embedding_weight,
        max_output_weight=args.max_output_weight,
        require_prompt_gte_activation=args.require_prompt_gte_activation,
        top_k_errors=args.top_k_errors,
        include_output=args.include_output,
    )
    report["review_csv"] = args.review_csv
    report["feature_table"] = args.feature_table
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote calibrated combination stability report to {output_json}")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown_report(report) + "\n")
        print(f"wrote calibrated combination stability markdown to {output_md}")
    for name, metrics in report["aggregate_holdout_metrics"].items():
        print(
            f"{name}: auc={metrics['auc']['mean']:.4f} "
            f"recall={metrics['recall']['mean']:.4f} "
            f"fpr={metrics['false_positive_rate']['mean']:.4f}"
        )


if __name__ == "__main__":
    main()
