#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-search prompt/embedding/activation fusion weights on reviewed labels."
    )
    parser.add_argument("--review-csv", required=True, help="Reviewed prompt-label CSV.")
    parser.add_argument(
        "--feature-table",
        required=True,
        help="Joined prompt/embedding/activation feature table JSONL.",
    )
    parser.add_argument("--output-json", required=True, help="Output calibration JSON.")
    parser.add_argument("--output-md", default=None, help="Optional Markdown summary.")
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
    parser.add_argument("--top-k", type=int, default=25)
    return parser.parse_args()


def values_from_step(step: float) -> list[float]:
    if step <= 0 or step > 1:
        raise ValueError("step must be in (0, 1]")
    count = round(1.0 / step)
    return [round(idx * step, 10) for idx in range(count + 1)]


def candidate_weights(
    *,
    step: float,
    min_prompt_weight: float,
    max_embedding_weight: float,
    require_prompt_gte_activation: bool,
    require_activation_gte_embedding: bool,
) -> list[dict[str, float]]:
    candidates = []
    values = values_from_step(step)
    tolerance = step / 100
    for prompt_weight in values:
        for embedding_weight in values:
            activation_weight = round(1.0 - prompt_weight - embedding_weight, 10)
            if activation_weight < -tolerance:
                continue
            if activation_weight < 0:
                activation_weight = 0.0
            if abs(prompt_weight + embedding_weight + activation_weight - 1.0) > tolerance:
                continue
            if prompt_weight < min_prompt_weight:
                continue
            if embedding_weight > max_embedding_weight:
                continue
            if require_prompt_gte_activation and prompt_weight < activation_weight:
                continue
            if require_activation_gte_embedding and activation_weight < embedding_weight:
                continue
            candidates.append(
                {
                    "prompt_weight": prompt_weight,
                    "embedding_weight": embedding_weight,
                    "activation_weight": activation_weight,
                }
            )
    return candidates


def binary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if binary_label(str(row.get("reviewed_label"))) is not None
        and row.get("prompt_risk_score") is not None
        and row.get("embedding_prior_score") is not None
        and row.get("activation_probability") is not None
    ]


def fusion_score(row: dict[str, Any], weights: dict[str, float]) -> float:
    return (
        weights["prompt_weight"] * float(row["prompt_risk_score"])
        + weights["embedding_weight"] * float(row["embedding_prior_score"])
        + weights["activation_weight"] * float(row["activation_probability"])
    )


def false_negative_rows(
    rows: list[dict[str, Any]],
    scores: list[float],
    threshold: float,
) -> list[dict[str, Any]]:
    output = []
    for row, score in zip(rows, scores, strict=True):
        if binary_label(str(row["reviewed_label"])) == 1 and score < threshold:
            output.append(
                {
                    "review_id": row.get("review_id"),
                    "source_id": row.get("source_id"),
                    "score": score,
                    "threshold": threshold,
                    "qwen_label": row.get("qwen_label"),
                    "domain": row.get("domain"),
                    "subcluster_id": row.get("subcluster_id"),
                    "prompt_text": row.get("prompt_text"),
                }
            )
    return output


def evaluate_candidate(
    rows: list[dict[str, Any]],
    labels: list[int],
    weights: dict[str, float],
    threshold: float,
) -> dict[str, Any]:
    scores = [fusion_score(row, weights) for row in rows]
    metric = metrics(labels, scores, threshold=threshold)
    return {
        **weights,
        "threshold": threshold,
        "auc": auc(labels, scores),
        "metrics": metric,
        "false_negatives": false_negative_rows(rows, scores, threshold),
    }


def objective_tuple(candidate: dict[str, Any], target_fpr: float) -> tuple[Any, ...]:
    metric = candidate["metrics"]
    meets_target = metric["false_positive_rate"] <= target_fpr
    return (
        1 if meets_target else 0,
        metric["recall"],
        -metric["false_positive_rate"],
        metric["f1"],
        metric["accuracy"],
        candidate["auc"] if candidate["auc"] is not None else -1.0,
        candidate["prompt_weight"],
        candidate["activation_weight"],
        -candidate["embedding_weight"],
    )


def best_no_fn_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    no_fn = [candidate for candidate in candidates if candidate["metrics"]["fn"] == 0]
    if not no_fn:
        return None
    return max(
        no_fn,
        key=lambda candidate: (
            -candidate["metrics"]["false_positive_rate"],
            candidate["metrics"]["precision"],
            candidate["metrics"]["accuracy"],
            candidate["metrics"]["f1"],
        ),
    )


def calibrate(
    rows: list[dict[str, Any]],
    *,
    weight_step: float,
    threshold_step: float,
    target_fpr: float,
    min_prompt_weight: float,
    max_embedding_weight: float,
    require_prompt_gte_activation: bool,
    require_activation_gte_embedding: bool,
    top_k: int,
) -> dict[str, Any]:
    eval_rows = binary_rows(rows)
    labels = [int(binary_label(str(row["reviewed_label"]))) for row in eval_rows]
    weights = candidate_weights(
        step=weight_step,
        min_prompt_weight=min_prompt_weight,
        max_embedding_weight=max_embedding_weight,
        require_prompt_gte_activation=require_prompt_gte_activation,
        require_activation_gte_embedding=require_activation_gte_embedding,
    )
    thresholds = values_from_step(threshold_step)
    candidates = [
        evaluate_candidate(eval_rows, labels, weight, threshold)
        for weight in weights
        for threshold in thresholds
    ]
    if not candidates:
        raise ValueError("no calibration candidates generated")

    ranked = sorted(
        candidates,
        key=lambda candidate: objective_tuple(candidate, target_fpr),
        reverse=True,
    )
    best_f1 = max(
        candidates,
        key=lambda candidate: (
            candidate["metrics"]["f1"],
            candidate["metrics"]["accuracy"],
            -candidate["metrics"]["false_positive_rate"],
        ),
    )
    no_fn = best_no_fn_candidate(candidates)
    return {
        "num_reviewed_rows": len(rows),
        "num_binary_eval_rows": len(eval_rows),
        "label_counts": {
            "safe": labels.count(0),
            "unsafe": labels.count(1),
        },
        "search_space": {
            "weight_step": weight_step,
            "threshold_step": threshold_step,
            "target_fpr": target_fpr,
            "min_prompt_weight": min_prompt_weight,
            "max_embedding_weight": max_embedding_weight,
            "require_prompt_gte_activation": require_prompt_gte_activation,
            "require_activation_gte_embedding": require_activation_gte_embedding,
            "num_weight_candidates": len(weights),
            "num_thresholds": len(thresholds),
            "num_total_candidates": len(candidates),
        },
        "selected_by_target_fpr": ranked[0],
        "selected_by_best_f1": best_f1,
        "selected_by_no_false_negatives": no_fn,
        "top_candidates": ranked[:top_k],
        "interpretation": (
            "Weights and thresholds are selected by grid search on reviewed labels. This is a "
            "calibration artifact, not a final held-out paper result; a larger reviewed set should "
            "be split into calibration and held-out evaluation partitions."
        ),
    }


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    metric = candidate["metrics"]
    return {
        "prompt_weight": candidate["prompt_weight"],
        "embedding_weight": candidate["embedding_weight"],
        "activation_weight": candidate["activation_weight"],
        "threshold": candidate["threshold"],
        "auc": candidate["auc"],
        "accuracy": metric["accuracy"],
        "precision": metric["precision"],
        "recall": metric["recall"],
        "false_positive_rate": metric["false_positive_rate"],
        "f1": metric["f1"],
        "fp": metric["fp"],
        "fn": metric["fn"],
    }


def markdown_report(report: dict[str, Any]) -> str:
    selected = compact_candidate(report["selected_by_target_fpr"])
    best_f1 = compact_candidate(report["selected_by_best_f1"])
    no_fn = (
        compact_candidate(report["selected_by_no_false_negatives"])
        if report["selected_by_no_false_negatives"]
        else None
    )
    lines = [
        "# Reviewed Fusion Calibration",
        "",
        report["interpretation"],
        "",
        f"Binary rows: {report['num_binary_eval_rows']}",
        f"Search candidates: {report['search_space']['num_total_candidates']}",
        "",
        "## Selected Calibrations",
        "",
        (
            "| Objective | Prompt | Embedding | Activation | Threshold | Accuracy | "
            "Recall | FPR | FP | FN |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows = [("target FPR", selected), ("best F1", best_f1)]
    if no_fn:
        rows.append(("no false negatives", no_fn))
    for name, row in rows:
        lines.append(
            f"| {name} | {row['prompt_weight']:.2f} | {row['embedding_weight']:.2f} | "
            f"{row['activation_weight']:.2f} | {row['threshold']:.2f} | "
            f"{row['accuracy']:.4f} | {row['recall']:.4f} | "
            f"{row['false_positive_rate']:.4f} | {row['fp']} | {row['fn']} |"
        )
    lines.extend(
        [
            "",
            "## Top Target-FPR Candidates",
            "",
            (
                "| Rank | Prompt | Embedding | Activation | Threshold | Accuracy | "
                "Recall | FPR | FP | FN |"
            ),
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, candidate in enumerate(report["top_candidates"][:10], start=1):
        row = compact_candidate(candidate)
        lines.append(
            f"| {idx} | {row['prompt_weight']:.2f} | {row['embedding_weight']:.2f} | "
            f"{row['activation_weight']:.2f} | {row['threshold']:.2f} | "
            f"{row['accuracy']:.4f} | {row['recall']:.4f} | "
            f"{row['false_positive_rate']:.4f} | {row['fp']} | {row['fn']} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    reviewed = load_reviewed_rows(Path(args.review_csv))
    features = load_feature_table(Path(args.feature_table))
    rows = join_rows(reviewed, features)
    if not rows:
        raise ValueError("no reviewed rows joined with feature table")

    report = calibrate(
        rows,
        weight_step=args.weight_step,
        threshold_step=args.threshold_step,
        target_fpr=args.target_fpr,
        min_prompt_weight=args.min_prompt_weight,
        max_embedding_weight=args.max_embedding_weight,
        require_prompt_gte_activation=args.require_prompt_gte_activation,
        require_activation_gte_embedding=args.require_activation_gte_embedding,
        top_k=args.top_k,
    )
    report["review_csv"] = args.review_csv
    report["feature_table"] = args.feature_table

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote reviewed fusion calibration to {output_json}")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown_report(report) + "\n", encoding="utf-8")
        print(f"wrote reviewed fusion calibration markdown to {output_md}")

    selected = compact_candidate(report["selected_by_target_fpr"])
    print(
        "selected target_fpr "
        f"weights=({selected['prompt_weight']:.2f},"
        f"{selected['embedding_weight']:.2f},{selected['activation_weight']:.2f}) "
        f"threshold={selected['threshold']:.2f} recall={selected['recall']:.4f} "
        f"fpr={selected['false_positive_rate']:.4f} fp={selected['fp']} fn={selected['fn']}"
    )


if __name__ == "__main__":
    main()
