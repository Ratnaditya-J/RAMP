#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

SCORE_KEYS = (
    "prompt_risk_score",
    "embedding_prior_score",
    "activation_probability",
    "prompt_embedding_score",
    "prompt_activation_score",
    "cumulative_full_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate prompt+embedding+activation signals against reviewed labels."
    )
    parser.add_argument("--review-csv", required=True, help="Reviewed prompt-label CSV.")
    parser.add_argument(
        "--feature-table",
        required=True,
        help="Joined prompt/embedding/activation feature table JSONL.",
    )
    parser.add_argument("--output-json", required=True, help="Output evaluation JSON.")
    parser.add_argument("--output-md", default=None, help="Optional Markdown summary.")
    parser.add_argument(
        "--reviewed-feature-table",
        default=None,
        help="Optional reviewed joined feature table JSONL.",
    )
    return parser.parse_args()


def normalize_label(label: Any) -> str:
    value = str(label).strip().lower()
    if value in {"safe", "unsafe", "controversial", "ambiguous_or_context_needed"}:
        return value
    if value == "bad_benchmark_label":
        return value
    if value in {"", "none", "null"}:
        return ""
    return value


def binary_label(label: str) -> int | None:
    if label == "unsafe":
        return 1
    if label == "safe":
        return 0
    return None


def load_reviewed_rows(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        rows = list(csv.DictReader(input_file))
    reviewed = {}
    for row in rows:
        reviewed_label = normalize_label(row.get("reviewed_label"))
        status = str(row.get("review_status", "")).strip().lower()
        source_id = str(row.get("source_id", "")).strip()
        if status != "reviewed" or not reviewed_label or not source_id:
            continue
        row = dict(row)
        row["reviewed_label"] = reviewed_label
        reviewed[source_id] = row
    return reviewed


def load_feature_table(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row.get("id")
            if row_id is not None:
                rows[str(row_id)] = row
    return rows


def auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranked = sorted(enumerate(scores), key=lambda item: item[1])
    rank_sum = 0.0
    idx = 0
    while idx < len(ranked):
        end = idx + 1
        while end < len(ranked) and ranked[end][1] == ranked[idx][1]:
            end += 1
        average_rank = (idx + 1 + end) / 2
        for row_idx in range(idx, end):
            if labels[ranked[row_idx][0]] == 1:
                rank_sum += average_rank
        idx = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores, strict=True):
        predicted = score >= threshold
        if predicted and label == 1:
            tp += 1
        elif predicted and label == 0:
            fp += 1
        elif not predicted and label == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "accuracy": accuracy,
        "f1": f1,
    }


def threshold_sweep(labels: list[int], scores: list[float]) -> dict[str, Any]:
    reports = [metrics(labels, scores, threshold=idx / 100) for idx in range(101)]
    return {
        "default_0_5": metrics(labels, scores, threshold=0.5),
        "best_f1": max(reports, key=lambda item: (item["f1"], item["accuracy"])),
        "target_fpr_0_10": max(
            (item for item in reports if item["false_positive_rate"] <= 0.10),
            key=lambda item: (item["recall"], item["precision"], item["accuracy"]),
            default=min(reports, key=lambda item: item["false_positive_rate"]),
        ),
    }


def join_rows(
    reviewed_rows: dict[str, dict[str, Any]],
    feature_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for row_id in sorted(set(reviewed_rows) & set(feature_rows)):
        review = reviewed_rows[row_id]
        feature = feature_rows[row_id]
        joined = dict(feature)
        joined.update(
            {
                "review_id": review.get("review_id"),
                "source_id": row_id,
                "reviewed_label": review.get("reviewed_label"),
                "review_status": review.get("review_status"),
                "label_issue_type": review.get("label_issue_type"),
                "reviewer_notes": review.get("reviewer_notes"),
                "prompt_text": review.get("prompt_text", feature.get("span_text")),
                "corpus_label": review.get("corpus_label", feature.get("label")),
                "qwen_label": review.get("qwen_label", feature.get("prompt_label")),
            }
        )
        rows.append(joined)
    return rows


def compact_error(row: dict[str, Any], score_key: str, score: float) -> dict[str, Any]:
    return {
        "review_id": row.get("review_id"),
        "source_id": row.get("source_id"),
        "reviewed_label": row.get("reviewed_label"),
        "score_key": score_key,
        "score": score,
        "qwen_label": row.get("qwen_label"),
        "corpus_label": row.get("corpus_label"),
        "audit_bucket": row.get("audit_bucket"),
        "source": row.get("source"),
        "domain": row.get("domain"),
        "subcluster_id": row.get("subcluster_id"),
        "prompt_text": row.get("prompt_text"),
    }


def evaluate_score(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    labeled = [
        (binary_label(str(row["reviewed_label"])), float(row[score_key]), row)
        for row in rows
        if binary_label(str(row.get("reviewed_label"))) is not None
        and row.get(score_key) is not None
    ]
    labels = [int(label) for label, _, _ in labeled if label is not None]
    scores = [score for _, score, _ in labeled]
    false_positives = [
        compact_error(row, score_key, score)
        for label, score, row in labeled
        if label == 0 and score >= 0.5
    ]
    false_negatives = [
        compact_error(row, score_key, score)
        for label, score, row in labeled
        if label == 1 and score < 0.5
    ]
    return {
        "score_key": score_key,
        "num_rows": len(labeled),
        "auc": auc(labels, scores),
        "metrics": threshold_sweep(labels, scores),
        "false_positives_threshold_0_5": false_positives,
        "false_negatives_threshold_0_5": false_negatives,
        "label_counts": dict(Counter(str(row["reviewed_label"]) for _, _, row in labeled)),
    }


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ablations = {score_key: evaluate_score(rows, score_key) for score_key in SCORE_KEYS}
    prompt_fp_ids = {
        row["source_id"] for row in ablations["prompt_risk_score"]["false_positives_threshold_0_5"]
    }
    cumulative_fp_ids = {
        row["source_id"]
        for row in ablations["cumulative_full_score"]["false_positives_threshold_0_5"]
    }
    activation_fp_ids = {
        row["source_id"]
        for row in ablations["prompt_activation_score"]["false_positives_threshold_0_5"]
    }
    return {
        "num_reviewed_joined_rows": len(rows),
        "num_binary_eval_rows": ablations["prompt_risk_score"]["num_rows"],
        "reviewed_label_counts": dict(Counter(str(row["reviewed_label"]) for row in rows)),
        "audit_bucket_counts": dict(Counter(str(row.get("audit_bucket")) for row in rows)),
        "ablations": ablations,
        "false_positive_delta": {
            "prompt_fp_count": len(prompt_fp_ids),
            "prompt_activation_fp_count": len(activation_fp_ids),
            "cumulative_full_fp_count": len(cumulative_fp_ids),
            "prompt_fp_fixed_by_prompt_activation": sorted(prompt_fp_ids - activation_fp_ids),
            "prompt_fp_fixed_by_cumulative_full": sorted(prompt_fp_ids - cumulative_fp_ids),
            "new_cumulative_full_fp_not_prompt_fp": sorted(cumulative_fp_ids - prompt_fp_ids),
        },
        "interpretation": (
            "This reviewed slice tests whether embedding and activation evidence improve the "
            "prompt-classifier signal on manually reviewed disagreement cases. It is a v0 audit "
            "slice, not a representative production benchmark."
        ),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, separators=(",", ":")) + "\n")


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Reviewed Cumulative Signal Evaluation",
        "",
        report["interpretation"],
        "",
        f"Reviewed joined rows: {report['num_reviewed_joined_rows']}",
        f"Binary safe/unsafe eval rows: {report['num_binary_eval_rows']}",
        "",
        "## Ablations",
        "",
        "| Signal | AUC | Accuracy | Precision | Recall | FPR | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for score_key, result in report["ablations"].items():
        metric = result["metrics"]["default_0_5"]
        auc_value = result["auc"]
        auc_text = "n/a" if auc_value is None else f"{auc_value:.4f}"
        lines.append(
            f"| `{score_key}` | {auc_text} | {metric['accuracy']:.4f} | "
            f"{metric['precision']:.4f} | {metric['recall']:.4f} | "
            f"{metric['false_positive_rate']:.4f} | {metric['fp']} | {metric['fn']} |"
        )
    delta = report["false_positive_delta"]
    lines.extend(
        [
            "",
            "## False-Positive Delta",
            "",
            f"Prompt false positives: {delta['prompt_fp_count']}",
            f"Prompt + activation false positives: {delta['prompt_activation_fp_count']}",
            f"Prompt + embedding + activation false positives: {delta['cumulative_full_fp_count']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    reviewed = load_reviewed_rows(Path(args.review_csv))
    features = load_feature_table(Path(args.feature_table))
    rows = join_rows(reviewed, features)
    if not rows:
        raise ValueError("no reviewed rows joined with feature table")

    report = build_report(rows)
    report["review_csv"] = args.review_csv
    report["feature_table"] = args.feature_table

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote reviewed cumulative signal evaluation to {output_json}")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown_report(report) + "\n", encoding="utf-8")
        print(f"wrote reviewed cumulative signal markdown to {output_md}")

    if args.reviewed_feature_table:
        write_jsonl(Path(args.reviewed_feature_table), rows)
        print(f"wrote reviewed joined feature table to {args.reviewed_feature_table}")

    for score_key, result in report["ablations"].items():
        metric = result["metrics"]["default_0_5"]
        auc_value = result["auc"]
        auc_text = "n/a" if auc_value is None else f"{auc_value:.4f}"
        print(
            f"{score_key}: auc={auc_text} accuracy={metric['accuracy']:.4f} "
            f"recall={metric['recall']:.4f} fpr={metric['false_positive_rate']:.4f} "
            f"fp={metric['fp']} fn={metric['fn']}"
        )


if __name__ == "__main__":
    main()
