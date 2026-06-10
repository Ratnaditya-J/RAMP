#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluate_reviewed_cumulative_signals import (
    join_rows,
    load_feature_table,
    load_reviewed_rows,
)
from evaluate_severity_floors import DEFAULT_HIGH_SEVERITY_CATEGORIES, normalize_category
from evaluate_split_calibrated_ramp import build_split_report

DEFAULT_COMPARE_KEYS = ("prompt_risk_score", "ramp_fusion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated split-calibrated RAMP evaluations for stability analysis."
    )
    parser.add_argument("--review-csv", required=True, help="Reviewed prompt-label CSV.")
    parser.add_argument(
        "--feature-table",
        required=True,
        help="Joined prompt/embedding/activation feature table JSONL.",
    )
    parser.add_argument("--output-json", required=True, help="Output stability JSON.")
    parser.add_argument("--output-md", default=None, help="Optional Markdown summary.")
    parser.add_argument("--num-splits", type=int, default=30)
    parser.add_argument("--seed-prefix", default="ramp_stability_v0.1")
    parser.add_argument("--calibration-fraction", type=float, default=0.50)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--target-fpr", type=float, default=0.10)
    parser.add_argument("--min-prompt-weight", type=float, default=0.40)
    parser.add_argument("--max-embedding-weight", type=float, default=0.20)
    parser.add_argument("--require-prompt-gte-activation", action="store_true")
    parser.add_argument("--require-activation-gte-embedding", action="store_true")
    parser.add_argument(
        "--calibration-objective",
        default="selected_by_best_f1",
        choices=(
            "selected_by_target_fpr",
            "selected_by_best_f1",
            "selected_by_no_false_negatives",
        ),
    )
    parser.add_argument("--floor", type=float, default=0.72)
    parser.add_argument(
        "--high-severity-category",
        action="append",
        default=None,
        help="High-severity prompt category. Repeat to override defaults.",
    )
    parser.add_argument("--top-k-errors", type=int, default=50)
    return parser.parse_args()


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
        auc = report.get("auc")
        return None if auc is None else float(auc)
    metrics = report.get("metrics", {})
    if metric_name in metrics:
        return float(metrics[metric_name])
    error_counts = report.get("error_counts", {})
    if metric_name in error_counts:
        return float(error_counts[metric_name])
    return None


def collect_error_counts(
    split_reports: list[dict[str, Any]],
    *,
    condition: str,
    error_key: str,
) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    rows_by_id: dict[str, dict[str, Any]] = {}
    split_ids: dict[str, set[str]] = defaultdict(set)
    for split in split_reports:
        split_id = str(split["split_index"])
        report = split["holdout_eval"]["ablations"][condition]
        for row in report.get(error_key, []):
            row_id = str(row.get("source_id") or row.get("review_id"))
            if not row_id or row_id == "None":
                continue
            counts[row_id] += 1
            rows_by_id[row_id] = row
            split_ids[row_id].add(split_id)

    ranked = []
    for row_id, count in counts.most_common():
        row = rows_by_id[row_id]
        ranked.append(
            {
                "source_id": row_id,
                "count": count,
                "split_frequency": count / max(len(split_reports), 1),
                "splits": sorted(split_ids[row_id]),
                "reviewed_label": row.get("reviewed_label"),
                "domain": row.get("domain"),
                "subcluster_role": row.get("subcluster_role"),
                "subcluster_id": row.get("subcluster_id"),
                "qwen_label": row.get("qwen_label"),
                "prompt_text": row.get("prompt_text"),
                "score": row.get("score"),
            }
        )
    return ranked


def aggregate_condition(
    split_reports: list[dict[str, Any]],
    condition: str,
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
    output = {}
    for metric_name in metric_names:
        values = [
            value
            for split in split_reports
            if (
                value := metric_value(
                    split["holdout_eval"]["ablations"][condition],
                    metric_name,
                )
            )
            is not None
        ]
        output[metric_name] = numeric_summary(values)
    return output


def aggregate_deltas(
    split_reports: list[dict[str, Any]],
    *,
    baseline: str,
    candidate: str,
) -> dict[str, Any]:
    metric_names = (
        "auc",
        "accuracy",
        "recall",
        "false_positive_rate",
        "fp",
        "fn",
        "severe_false_negatives",
        "hard_benign_false_positives",
    )
    output = {}
    for metric_name in metric_names:
        values = []
        for split in split_reports:
            base = metric_value(split["holdout_eval"]["ablations"][baseline], metric_name)
            cand = metric_value(split["holdout_eval"]["ablations"][candidate], metric_name)
            if base is not None and cand is not None:
                values.append(cand - base)
        output[f"{candidate}_minus_{baseline}_{metric_name}"] = numeric_summary(values)
    return output


def selected_config_summary(split_reports: list[dict[str, Any]]) -> dict[str, Any]:
    configs = [
        split["calibration_protocol"]["selected_config"] for split in split_reports
    ]
    weights = {
        "prompt_weight": [float(config["prompt_weight"]) for config in configs],
        "embedding_weight": [float(config["embedding_weight"]) for config in configs],
        "activation_weight": [float(config["activation_weight"]) for config in configs],
        "threshold": [float(config["threshold"]) for config in configs],
    }
    return {
        key: numeric_summary(values)
        for key, values in weights.items()
    } | {
        "most_common_configs": [
            {
                "prompt_weight": key[0],
                "embedding_weight": key[1],
                "activation_weight": key[2],
                "threshold": key[3],
                "count": count,
            }
            for key, count in Counter(
                (
                    float(config["prompt_weight"]),
                    float(config["embedding_weight"]),
                    float(config["activation_weight"]),
                    float(config["threshold"]),
                )
                for config in configs
            ).most_common()
        ]
    }


def build_stability_report(
    rows: list[dict[str, Any]],
    *,
    num_splits: int,
    seed_prefix: str,
    calibration_fraction: float,
    weight_step: float,
    threshold_step: float,
    target_fpr: float,
    min_prompt_weight: float,
    max_embedding_weight: float,
    require_prompt_gte_activation: bool,
    require_activation_gte_embedding: bool,
    calibration_objective: str,
    floor: float,
    high_severity_categories: set[str],
    top_k_errors: int,
) -> dict[str, Any]:
    if num_splits <= 0:
        raise ValueError("num_splits must be positive")
    split_reports = []
    for split_index in range(num_splits):
        seed = f"{seed_prefix}:{split_index:03d}"
        split_report = build_split_report(
            rows,
            calibration_fraction=calibration_fraction,
            split_seed=seed,
            weight_step=weight_step,
            threshold_step=threshold_step,
            target_fpr=target_fpr,
            min_prompt_weight=min_prompt_weight,
            max_embedding_weight=max_embedding_weight,
            require_prompt_gte_activation=require_prompt_gte_activation,
            require_activation_gte_embedding=require_activation_gte_embedding,
            calibration_objective=calibration_objective,
            floor=floor,
            high_severity_categories=high_severity_categories,
            top_k_errors=top_k_errors,
        )
        split_report["split_index"] = split_index
        split_report["split"]["seed"] = seed
        split_reports.append(split_report)

    conditions = {
        condition
        for split in split_reports
        for condition in split["holdout_eval"]["ablations"]
    }
    aggregate = {
        condition: aggregate_condition(split_reports, condition)
        for condition in sorted(conditions)
    }
    deltas = aggregate_deltas(
        split_reports,
        baseline="prompt_risk_score",
        candidate="ramp_fusion",
    )
    error_distribution = {
        "ramp_fusion_false_positives": collect_error_counts(
            split_reports,
            condition="ramp_fusion",
            error_key="false_positives",
        ),
        "ramp_fusion_false_negatives": collect_error_counts(
            split_reports,
            condition="ramp_fusion",
            error_key="false_negatives",
        ),
        "ramp_fusion_severe_false_negatives": collect_error_counts(
            split_reports,
            condition="ramp_fusion",
            error_key="severe_false_negatives",
        ),
        "ramp_fusion_hard_benign_false_positives": collect_error_counts(
            split_reports,
            condition="ramp_fusion",
            error_key="hard_benign_false_positives",
        ),
        "prompt_only_false_positives": collect_error_counts(
            split_reports,
            condition="prompt_risk_score",
            error_key="false_positives",
        ),
        "prompt_only_false_negatives": collect_error_counts(
            split_reports,
            condition="prompt_risk_score",
            error_key="false_negatives",
        ),
    }
    return {
        "artifact_id": "ramp_split_stability_evaluation_v0.1",
        "num_splits": num_splits,
        "seed_prefix": seed_prefix,
        "num_binary_eval_rows": split_reports[0]["num_binary_eval_rows"],
        "calibration_fraction": calibration_fraction,
        "calibration_objective": calibration_objective,
        "aggregate_holdout_metrics": aggregate,
        "ramp_fusion_vs_prompt_only_deltas": deltas,
        "selected_config_summary": selected_config_summary(split_reports),
        "error_distribution": error_distribution,
        "split_reports": split_reports,
        "interpretation": (
            "Repeated deterministic splits estimate whether RAMP fusion improvements over the "
            "prompt classifier are stable under reviewed-label calibration/holdout variation."
        ),
    }


def summary_cell(summary: dict[str, Any], precision: int = 4) -> str:
    mean = summary.get("mean")
    stdev = summary.get("stdev")
    min_value = summary.get("min")
    max_value = summary.get("max")
    if mean is None:
        return "n/a"
    return (
        f"{mean:.{precision}f} +/- {stdev:.{precision}f} "
        f"[{min_value:.{precision}f}, {max_value:.{precision}f}]"
    )


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# RAMP Split Stability Evaluation",
        "",
        report["interpretation"],
        "",
        f"Splits: {report['num_splits']}",
        f"Binary rows: {report['num_binary_eval_rows']}",
        f"Calibration fraction: {report['calibration_fraction']:.2f}",
        f"Objective: `{report['calibration_objective']}`",
        "",
        "## Holdout Metrics",
        "",
        "| Condition | AUC | Accuracy | Recall | FPR | FP | FN | Severe FN | Hard benign FP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition, metrics in report["aggregate_holdout_metrics"].items():
        lines.append(
            f"| `{condition}` | {summary_cell(metrics['auc'])} | "
            f"{summary_cell(metrics['accuracy'])} | {summary_cell(metrics['recall'])} | "
            f"{summary_cell(metrics['false_positive_rate'])} | "
            f"{summary_cell(metrics['fp'], precision=2)} | "
            f"{summary_cell(metrics['fn'], precision=2)} | "
            f"{summary_cell(metrics['severe_false_negatives'], precision=2)} | "
            f"{summary_cell(metrics['hard_benign_false_positives'], precision=2)} |"
        )
    lines.extend(
        [
            "",
            "## RAMP Minus Prompt-Only",
            "",
            "| Delta | Mean +/- stdev [min, max] |",
            "| --- | ---: |",
        ]
    )
    for name, summary in report["ramp_fusion_vs_prompt_only_deltas"].items():
        lines.append(f"| `{name}` | {summary_cell(summary)} |")
    config = report["selected_config_summary"]
    lines.extend(
        [
            "",
            "## Selected Config Stability",
            "",
            f"Prompt weight: {summary_cell(config['prompt_weight'])}",
            f"Embedding weight: {summary_cell(config['embedding_weight'])}",
            f"Activation weight: {summary_cell(config['activation_weight'])}",
            f"Threshold: {summary_cell(config['threshold'])}",
            "",
            "## Top Repeated RAMP Errors",
            "",
        ]
    )
    for key in (
        "ramp_fusion_severe_false_negatives",
        "ramp_fusion_hard_benign_false_positives",
        "ramp_fusion_false_negatives",
        "ramp_fusion_false_positives",
    ):
        lines.append(f"### `{key}`")
        rows = report["error_distribution"][key][:10]
        if not rows:
            lines.append("")
            lines.append("None.")
            lines.append("")
            continue
        lines.extend(
            [
                "",
                "| Count | Frequency | Domain | Subcluster | Source ID | Prompt |",
                "| ---: | ---: | --- | --- | --- | --- |",
            ]
        )
        for row in rows:
            prompt = str(row.get("prompt_text", "")).replace("|", "\\|")[:100]
            lines.append(
                f"| {row['count']} | {row['split_frequency']:.2f} | "
                f"{row.get('domain')} | {row.get('subcluster_id')} | "
                f"`{row.get('source_id')}` | {prompt} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    reviewed = load_reviewed_rows(Path(args.review_csv))
    features = load_feature_table(Path(args.feature_table))
    rows = join_rows(reviewed, features)
    if not rows:
        raise ValueError("no reviewed rows joined with feature table")

    high_severity_categories = {
        normalize_category(category)
        for category in (args.high_severity_category or DEFAULT_HIGH_SEVERITY_CATEGORIES)
    }
    report = build_stability_report(
        rows,
        num_splits=args.num_splits,
        seed_prefix=args.seed_prefix,
        calibration_fraction=args.calibration_fraction,
        weight_step=args.weight_step,
        threshold_step=args.threshold_step,
        target_fpr=args.target_fpr,
        min_prompt_weight=args.min_prompt_weight,
        max_embedding_weight=args.max_embedding_weight,
        require_prompt_gte_activation=args.require_prompt_gte_activation,
        require_activation_gte_embedding=args.require_activation_gte_embedding,
        calibration_objective=args.calibration_objective,
        floor=args.floor,
        high_severity_categories=high_severity_categories,
        top_k_errors=args.top_k_errors,
    )
    report["review_csv"] = args.review_csv
    report["feature_table"] = args.feature_table

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote split stability evaluation to {output_json}")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown_report(report) + "\n", encoding="utf-8")
        print(f"wrote split stability markdown to {output_md}")

    deltas = report["ramp_fusion_vs_prompt_only_deltas"]
    print(
        "ramp minus prompt-only "
        f"auc={summary_cell(deltas['ramp_fusion_minus_prompt_risk_score_auc'])} "
        f"fpr={summary_cell(deltas['ramp_fusion_minus_prompt_risk_score_false_positive_rate'])} "
        f"fn={summary_cell(deltas['ramp_fusion_minus_prompt_risk_score_fn'], precision=2)}"
    )


if __name__ == "__main__":
    main()
