#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from calibrate_reviewed_fusion import calibrate, compact_candidate
from evaluate_ramp_harness import (
    PRIMARY_SCORE_KEYS,
    build_report,
    evaluate_calibrated_scores,
    evaluate_score_key,
)
from evaluate_reviewed_cumulative_signals import (
    SCORE_KEYS,
    binary_label,
    join_rows,
    load_feature_table,
    load_reviewed_rows,
)
from evaluate_severity_floors import DEFAULT_HIGH_SEVERITY_CATEGORIES, normalize_category


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split reviewed rows, calibrate on one split, evaluate RAMP on holdout."
    )
    parser.add_argument("--review-csv", required=True, help="Reviewed prompt-label CSV.")
    parser.add_argument(
        "--feature-table",
        required=True,
        help="Joined prompt/embedding/activation feature table JSONL.",
    )
    parser.add_argument("--output-json", required=True, help="Output split evaluation JSON.")
    parser.add_argument("--output-md", default=None, help="Optional Markdown summary.")
    parser.add_argument("--calibration-fraction", type=float, default=0.50)
    parser.add_argument("--split-seed", default="ramp_reviewed_split_v0.1")
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--target-fpr", type=float, default=0.10)
    parser.add_argument("--min-prompt-weight", type=float, default=0.40)
    parser.add_argument("--max-embedding-weight", type=float, default=0.20)
    parser.add_argument(
        "--require-prompt-gte-activation",
        action="store_true",
        help="Require prompt weight to be at least activation weight.",
    )
    parser.add_argument(
        "--require-activation-gte-embedding",
        action="store_true",
        help="Require activation weight to be at least embedding weight.",
    )
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
    parser.add_argument("--top-k-errors", type=int, default=20)
    return parser.parse_args()


def binary_feature_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if binary_label(str(row.get("reviewed_label"))) is not None
        and row.get("prompt_risk_score") is not None
        and row.get("embedding_prior_score") is not None
        and row.get("activation_probability") is not None
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
    if calibration_fraction <= 0 or calibration_fraction >= 1:
        raise ValueError("calibration_fraction must be between 0 and 1")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("reviewed_label"))].append(row)

    calibration_rows = []
    holdout_rows = []
    for _, group_rows in sorted(grouped.items()):
        ordered = sorted(group_rows, key=lambda row: split_sort_key(row, seed))
        calibration_count = round(len(ordered) * calibration_fraction)
        calibration_count = min(max(calibration_count, 1), len(ordered) - 1)
        calibration_rows.extend(ordered[:calibration_count])
        holdout_rows.extend(ordered[calibration_count:])
    return calibration_rows, holdout_rows


def calibration_to_runtime_config(
    calibration_report: dict[str, Any],
    objective: str,
) -> dict[str, Any]:
    selected = calibration_report.get(objective)
    if not isinstance(selected, dict):
        raise ValueError(f"calibration objective unavailable: {objective}")
    return {
        "artifact_id": calibration_report["artifact_id"],
        "objective": objective,
        "source_path": None,
        "prompt_weight": float(selected["prompt_weight"]),
        "embedding_weight": float(selected["embedding_weight"]),
        "activation_weight": float(selected["activation_weight"]),
        "threshold": float(selected["threshold"]),
    }


def ablation_reports(
    rows: list[dict[str, Any]],
    *,
    calibration_config: dict[str, Any],
    floor: float,
    high_severity_categories: set[str],
    top_k_errors: int,
) -> dict[str, Any]:
    ablations = {
        score_key: evaluate_score_key(
            rows,
            score_key,
            high_severity_categories=high_severity_categories,
            top_k_errors=top_k_errors,
        )
        for score_key in PRIMARY_SCORE_KEYS
        if score_key in SCORE_KEYS and any(row.get(score_key) is not None for row in rows)
    }
    calibrated = evaluate_calibrated_scores(
        rows,
        calibration_config,
        floor=floor,
        high_severity_categories=high_severity_categories,
        top_k_errors=top_k_errors,
    )
    ablations.update(calibrated["reports"])
    return {"ablations": ablations, "calibrated_fusion": calibrated}


def build_split_report(
    rows: list[dict[str, Any]],
    *,
    calibration_fraction: float,
    split_seed: str,
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
    eval_rows = binary_feature_rows(rows)
    calibration_rows, holdout_rows = stratified_split(
        eval_rows,
        calibration_fraction=calibration_fraction,
        seed=split_seed,
    )
    calibration_report = calibrate(
        calibration_rows,
        weight_step=weight_step,
        threshold_step=threshold_step,
        target_fpr=target_fpr,
        min_prompt_weight=min_prompt_weight,
        max_embedding_weight=max_embedding_weight,
        require_prompt_gte_activation=require_prompt_gte_activation,
        require_activation_gte_embedding=require_activation_gte_embedding,
        top_k=25,
    )
    calibration_report["artifact_id"] = "ramp_split_calibration_v0.1"
    calibration_config = calibration_to_runtime_config(
        calibration_report,
        calibration_objective,
    )
    calibration_eval = ablation_reports(
        calibration_rows,
        calibration_config=calibration_config,
        floor=floor,
        high_severity_categories=high_severity_categories,
        top_k_errors=top_k_errors,
    )
    holdout_eval = ablation_reports(
        holdout_rows,
        calibration_config=calibration_config,
        floor=floor,
        high_severity_categories=high_severity_categories,
        top_k_errors=top_k_errors,
    )
    full_context_report = build_report(
        eval_rows,
        calibration=calibration_config,
        floor=floor,
        high_severity_categories=high_severity_categories,
        top_k_errors=top_k_errors,
    )
    return {
        "artifact_id": "ramp_split_calibrated_evaluation_v0.1",
        "num_binary_eval_rows": len(eval_rows),
        "split": {
            "seed": split_seed,
            "calibration_fraction": calibration_fraction,
            "calibration_rows": len(calibration_rows),
            "holdout_rows": len(holdout_rows),
            "calibration_label_counts": dict(
                Counter(str(row["reviewed_label"]) for row in calibration_rows)
            ),
            "holdout_label_counts": dict(
                Counter(str(row["reviewed_label"]) for row in holdout_rows)
            ),
        },
        "calibration_protocol": {
            "weight_step": weight_step,
            "threshold_step": threshold_step,
            "target_fpr": target_fpr,
            "min_prompt_weight": min_prompt_weight,
            "max_embedding_weight": max_embedding_weight,
            "require_prompt_gte_activation": require_prompt_gte_activation,
            "require_activation_gte_embedding": require_activation_gte_embedding,
            "selected_objective": calibration_objective,
            "selected_config": calibration_config,
            "selected_metrics_on_calibration": compact_candidate(
                calibration_report[calibration_objective]
            ),
        },
        "calibration_report": calibration_report,
        "calibration_eval": calibration_eval,
        "holdout_eval": holdout_eval,
        "full_context_report": full_context_report,
        "interpretation": (
            "This report calibrates fusion weights and threshold only on the calibration split, "
            "then evaluates the selected RAMP fusion on held-out reviewed rows. The reviewed set "
            "is still targeted and hard-case-heavy, so this is a first split-aware research "
            "result rather than final benchmark evidence."
        ),
    }


def metric_row(name: str, report: dict[str, Any]) -> str:
    metric = report["metrics"]
    auc_value = report["auc"]
    auc_text = "n/a" if auc_value is None else f"{auc_value:.4f}"
    return (
        f"| `{name}` | {report['threshold']:.2f} | {auc_text} | "
        f"{metric['accuracy']:.4f} | {metric['precision']:.4f} | "
        f"{metric['recall']:.4f} | {metric['false_positive_rate']:.4f} | "
        f"{metric['fp']} | {metric['fn']} | "
        f"{report['error_counts']['severe_false_negatives']} | "
        f"{report['error_counts']['hard_benign_false_positives']} |"
    )


def markdown_section(title: str, evaluation: dict[str, Any]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        (
            "| Condition | Threshold | AUC | Accuracy | Precision | Recall | FPR | "
            "FP | FN | Severe FN | Hard benign FP |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, ablation in evaluation["ablations"].items():
        lines.append(metric_row(name, ablation))
    return lines


def markdown_report(report: dict[str, Any]) -> str:
    selected = report["calibration_protocol"]["selected_config"]
    lines = [
        "# Split-Calibrated RAMP Evaluation",
        "",
        report["interpretation"],
        "",
        f"Binary rows: {report['num_binary_eval_rows']}",
        f"Calibration rows: {report['split']['calibration_rows']}",
        f"Holdout rows: {report['split']['holdout_rows']}",
        "",
        "## Selected Calibration",
        "",
        f"Objective: `{selected['objective']}`",
        f"Prompt weight: {selected['prompt_weight']:.2f}",
        f"Embedding weight: {selected['embedding_weight']:.2f}",
        f"Activation weight: {selected['activation_weight']:.2f}",
        f"Threshold: {selected['threshold']:.2f}",
        "",
    ]
    lines.extend(markdown_section("Calibration Split Metrics", report["calibration_eval"]))
    lines.extend([""])
    lines.extend(markdown_section("Holdout Split Metrics", report["holdout_eval"]))
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
    report = build_split_report(
        rows,
        calibration_fraction=args.calibration_fraction,
        split_seed=args.split_seed,
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
    print(f"wrote split-calibrated RAMP evaluation to {output_json}")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown_report(report) + "\n", encoding="utf-8")
        print(f"wrote split-calibrated RAMP markdown to {output_md}")

    selected = report["calibration_protocol"]["selected_config"]
    holdout_ramp = report["holdout_eval"]["ablations"]["ramp_fusion"]
    metric = holdout_ramp["metrics"]
    print(
        "selected "
        f"weights=({selected['prompt_weight']:.2f},"
        f"{selected['embedding_weight']:.2f},{selected['activation_weight']:.2f}) "
        f"threshold={selected['threshold']:.2f}"
    )
    print(
        "holdout ramp_fusion "
        f"auc={holdout_ramp['auc'] if holdout_ramp['auc'] is not None else 'n/a'} "
        f"recall={metric['recall']:.4f} fpr={metric['false_positive_rate']:.4f} "
        f"fp={metric['fp']} fn={metric['fn']}"
    )


if __name__ == "__main__":
    main()
