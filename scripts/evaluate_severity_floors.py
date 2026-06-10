#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluate_reviewed_cumulative_signals import (
    auc,
    binary_label,
    join_rows,
    load_feature_table,
    load_reviewed_rows,
    metrics,
)

DEFAULT_HIGH_SEVERITY_CATEGORIES = (
    "pii",
    "personally identifiable information",
    "suicide & self-harm",
    "violent",
    "non-violent illegal acts",
    "jailbreak",
    "unethical acts",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate severity-aware floors on reviewed calibrated fusion rows."
    )
    parser.add_argument("--review-csv", required=True, help="Reviewed prompt-label CSV.")
    parser.add_argument(
        "--feature-table",
        required=True,
        help="Joined prompt/embedding/activation feature table JSONL.",
    )
    parser.add_argument(
        "--calibration-artifact",
        required=True,
        help="Reviewed fusion calibration artifact JSON.",
    )
    parser.add_argument(
        "--calibration-objective",
        default="selected_by_best_f1",
        help="Calibration objective key to evaluate.",
    )
    parser.add_argument("--output-json", required=True, help="Output evaluation JSON.")
    parser.add_argument("--output-md", default=None, help="Optional Markdown summary.")
    parser.add_argument("--floor", type=float, default=0.72)
    parser.add_argument(
        "--high-severity-category",
        action="append",
        default=None,
        help="High-severity prompt category. Repeat to override defaults.",
    )
    return parser.parse_args()


def normalize_category(value: Any) -> str:
    return str(value or "").strip().lower()


def load_calibration(path: Path, objective: str) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    selected = artifact.get(objective)
    if not isinstance(selected, dict):
        raise ValueError(f"calibration artifact missing objective: {objective}")
    return {
        "artifact_id": artifact.get("artifact_id", path.name),
        "objective": objective,
        "source_path": str(path),
        "prompt_weight": float(selected["prompt_weight"]),
        "embedding_weight": float(selected["embedding_weight"]),
        "activation_weight": float(selected["activation_weight"]),
        "threshold": float(selected["threshold"]),
    }


def binary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if binary_label(str(row.get("reviewed_label"))) is not None
        and row.get("prompt_risk_score") is not None
        and row.get("embedding_prior_score") is not None
        and row.get("activation_probability") is not None
    ]


def raw_fusion_score(row: dict[str, Any], calibration: dict[str, Any]) -> float:
    return (
        float(row["prompt_risk_score"]) * calibration["prompt_weight"]
        + float(row["embedding_prior_score"]) * calibration["embedding_weight"]
        + float(row["activation_probability"]) * calibration["activation_weight"]
    )


def severity_categories(row: dict[str, Any]) -> set[str]:
    categories = set()
    category = normalize_category(row.get("prompt_harm_category"))
    if category:
        categories.add(category)
    raw_categories = row.get("prompt_categories") or row.get("categories")
    if isinstance(raw_categories, str):
        categories.add(normalize_category(raw_categories))
    elif isinstance(raw_categories, (list, tuple, set)):
        categories.update(normalize_category(value) for value in raw_categories)
    return {category for category in categories if category and category != "none"}


def floor_event(
    row: dict[str, Any],
    raw_score: float,
    *,
    floor: float,
    high_severity_categories: set[str],
) -> dict[str, Any]:
    row_categories = severity_categories(row)
    matched_categories = sorted(row_categories & high_severity_categories)
    floor_candidate = (
        str(row.get("qwen_label") or row.get("prompt_label") or "").strip().lower() == "unsafe"
        and bool(matched_categories)
    )
    applied = floor_candidate and raw_score < floor
    final_score = floor if applied else raw_score
    return {
        "raw_score": raw_score,
        "final_score": final_score,
        "floor_candidate": floor_candidate,
        "floor_applied": applied,
        "matched_categories": matched_categories,
    }


def compact_row(
    row: dict[str, Any],
    *,
    raw_score: float,
    final_score: float,
    threshold: float,
    floor_applied: bool,
) -> dict[str, Any]:
    return {
        "review_id": row.get("review_id"),
        "source_id": row.get("source_id"),
        "reviewed_label": row.get("reviewed_label"),
        "raw_score": raw_score,
        "final_score": final_score,
        "threshold": threshold,
        "floor_applied": floor_applied,
        "qwen_label": row.get("qwen_label") or row.get("prompt_label"),
        "prompt_harm_category": row.get("prompt_harm_category"),
        "source": row.get("source"),
        "domain": row.get("domain"),
        "subcluster_id": row.get("subcluster_id"),
        "prompt_text": row.get("prompt_text"),
    }


def error_rows(
    rows: list[dict[str, Any]],
    scores: list[float],
    raw_scores: list[float],
    floor_applied: list[bool],
    threshold: float,
    *,
    label_value: int,
    predicted_value: bool,
) -> list[dict[str, Any]]:
    output = []
    for row, score, raw_score, applied in zip(
        rows, scores, raw_scores, floor_applied, strict=True
    ):
        label = binary_label(str(row["reviewed_label"]))
        predicted = score >= threshold
        if label == label_value and predicted == predicted_value:
            output.append(
                compact_row(
                    row,
                    raw_score=raw_score,
                    final_score=score,
                    threshold=threshold,
                    floor_applied=applied,
                )
            )
    return output


def evaluate(
    rows: list[dict[str, Any]],
    calibration: dict[str, Any],
    *,
    floor: float,
    high_severity_categories: set[str],
) -> dict[str, Any]:
    eval_rows = binary_rows(rows)
    labels = [int(binary_label(str(row["reviewed_label"]))) for row in eval_rows]
    raw_scores = [raw_fusion_score(row, calibration) for row in eval_rows]
    events = [
        floor_event(
            row,
            score,
            floor=floor,
            high_severity_categories=high_severity_categories,
        )
        for row, score in zip(eval_rows, raw_scores, strict=True)
    ]
    final_scores = [float(event["final_score"]) for event in events]
    floor_applied = [bool(event["floor_applied"]) for event in events]
    threshold = calibration["threshold"]
    raw_metric = metrics(labels, raw_scores, threshold)
    final_metric = metrics(labels, final_scores, threshold)
    raw_fn_ids = {
        row["source_id"]
        for row, score in zip(eval_rows, raw_scores, strict=True)
        if binary_label(str(row["reviewed_label"])) == 1 and score < threshold
    }
    final_fn_ids = {
        row["source_id"]
        for row, score in zip(eval_rows, final_scores, strict=True)
        if binary_label(str(row["reviewed_label"])) == 1 and score < threshold
    }
    raw_fp_ids = {
        row["source_id"]
        for row, score in zip(eval_rows, raw_scores, strict=True)
        if binary_label(str(row["reviewed_label"])) == 0 and score >= threshold
    }
    final_fp_ids = {
        row["source_id"]
        for row, score in zip(eval_rows, final_scores, strict=True)
        if binary_label(str(row["reviewed_label"])) == 0 and score >= threshold
    }
    floor_rows = [
        {
            **compact_row(
                row,
                raw_score=raw_score,
                final_score=float(event["final_score"]),
                threshold=threshold,
                floor_applied=bool(event["floor_applied"]),
            ),
            "matched_categories": event["matched_categories"],
        }
        for row, raw_score, event in zip(eval_rows, raw_scores, events, strict=True)
        if event["floor_candidate"]
    ]
    applied_rows = [row for row in floor_rows if row["floor_applied"]]
    return {
        "num_reviewed_rows": len(rows),
        "num_binary_eval_rows": len(eval_rows),
        "label_counts": dict(Counter(str(row["reviewed_label"]) for row in eval_rows)),
        "calibration": calibration,
        "severity_floor": {
            "floor": floor,
            "high_severity_categories": sorted(high_severity_categories),
            "candidate_rows": len(floor_rows),
            "applied_rows": len(applied_rows),
        },
        "raw_calibrated": {
            "auc": auc(labels, raw_scores),
            "metrics": raw_metric,
            "false_positives": error_rows(
                eval_rows,
                raw_scores,
                raw_scores,
                [False] * len(eval_rows),
                threshold,
                label_value=0,
                predicted_value=True,
            ),
            "false_negatives": error_rows(
                eval_rows,
                raw_scores,
                raw_scores,
                [False] * len(eval_rows),
                threshold,
                label_value=1,
                predicted_value=False,
            ),
        },
        "severity_floor_calibrated": {
            "auc": auc(labels, final_scores),
            "metrics": final_metric,
            "false_positives": error_rows(
                eval_rows,
                final_scores,
                raw_scores,
                floor_applied,
                threshold,
                label_value=0,
                predicted_value=True,
            ),
            "false_negatives": error_rows(
                eval_rows,
                final_scores,
                raw_scores,
                floor_applied,
                threshold,
                label_value=1,
                predicted_value=False,
            ),
            "floor_candidate_rows": floor_rows,
            "floor_applied_rows": applied_rows,
        },
        "delta": {
            "false_negatives_fixed": sorted(raw_fn_ids - final_fn_ids),
            "new_false_positives": sorted(final_fp_ids - raw_fp_ids),
            "false_positive_delta": len(final_fp_ids) - len(raw_fp_ids),
            "false_negative_delta": len(final_fn_ids) - len(raw_fn_ids),
        },
        "interpretation": (
            "Severity floors are evaluated as a runtime safety constraint around calibrated "
            "fusion. They are not fitted weights; this report checks whether the floor reduces "
            "unsafe false negatives and how many safe false positives it adds on reviewed rows."
        ),
    }


def markdown_report(report: dict[str, Any]) -> str:
    raw = report["raw_calibrated"]["metrics"]
    floored = report["severity_floor_calibrated"]["metrics"]
    severity_floor = report["severity_floor"]
    delta = report["delta"]
    lines = [
        "# Severity Floor Evaluation",
        "",
        report["interpretation"],
        "",
        f"Binary rows: {report['num_binary_eval_rows']}",
        f"Calibration objective: `{report['calibration']['objective']}`",
        f"Threshold: {report['calibration']['threshold']:.2f}",
        f"Floor: {severity_floor['floor']:.2f}",
        f"Floor candidate rows: {severity_floor['candidate_rows']}",
        f"Floor applied rows: {severity_floor['applied_rows']}",
        "",
        "## Before vs After",
        "",
        "| Variant | Accuracy | Precision | Recall | FPR | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Raw calibrated | {raw['accuracy']:.4f} | {raw['precision']:.4f} | "
            f"{raw['recall']:.4f} | {raw['false_positive_rate']:.4f} | "
            f"{raw['fp']} | {raw['fn']} |"
        ),
        (
            f"| Severity floor | {floored['accuracy']:.4f} | "
            f"{floored['precision']:.4f} | {floored['recall']:.4f} | "
            f"{floored['false_positive_rate']:.4f} | {floored['fp']} | "
            f"{floored['fn']} |"
        ),
        "",
        "## Delta",
        "",
        f"False negatives fixed: {len(delta['false_negatives_fixed'])}",
        f"New false positives: {len(delta['new_false_positives'])}",
        "",
    ]
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
        for category in (
            args.high_severity_category or DEFAULT_HIGH_SEVERITY_CATEGORIES
        )
    }
    calibration = load_calibration(
        Path(args.calibration_artifact),
        args.calibration_objective,
    )
    report = evaluate(
        rows,
        calibration,
        floor=args.floor,
        high_severity_categories=high_severity_categories,
    )
    report["review_csv"] = args.review_csv
    report["feature_table"] = args.feature_table

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote severity floor evaluation to {output_json}")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown_report(report) + "\n", encoding="utf-8")
        print(f"wrote severity floor markdown to {output_md}")

    raw = report["raw_calibrated"]["metrics"]
    floored = report["severity_floor_calibrated"]["metrics"]
    print(
        "raw_calibrated "
        f"recall={raw['recall']:.4f} fpr={raw['false_positive_rate']:.4f} "
        f"fp={raw['fp']} fn={raw['fn']}"
    )
    print(
        "severity_floor "
        f"recall={floored['recall']:.4f} fpr={floored['false_positive_rate']:.4f} "
        f"fp={floored['fp']} fn={floored['fn']}"
    )


if __name__ == "__main__":
    main()
