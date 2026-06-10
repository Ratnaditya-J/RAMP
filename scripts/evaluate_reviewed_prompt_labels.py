#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate prompt classifier labels against reviewed prompt labels."
    )
    parser.add_argument("--review-csv", required=True, help="Reviewed prompt-label CSV.")
    parser.add_argument("--output-json", required=True, help="Output evaluation JSON.")
    parser.add_argument("--output-md", default=None, help="Optional Markdown summary.")
    parser.add_argument("--reviewed-jsonl", default=None, help="Optional reviewed subset JSONL.")
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


def qwen_score(row: dict[str, Any]) -> float:
    if str(row.get("qwen_risk_score", "")).strip():
        return float(row["qwen_risk_score"])
    label = normalize_label(row.get("qwen_label"))
    if label == "unsafe":
        return 0.92
    if label == "controversial":
        return 0.58
    if label == "safe":
        return 0.08
    return 0.5


def qwen_binary_prediction(row: dict[str, Any], threshold: float) -> int:
    return 1 if qwen_score(row) >= threshold else 0


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


def binary_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
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
    }


def load_reviewed_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        rows = list(csv.DictReader(input_file))
    reviewed = []
    for row in rows:
        reviewed_label = normalize_label(row.get("reviewed_label"))
        status = str(row.get("review_status", "")).strip().lower()
        if status != "reviewed" or not reviewed_label:
            continue
        row = dict(row)
        row["reviewed_label"] = reviewed_label
        row["qwen_label_normalized"] = normalize_label(row.get("qwen_label"))
        row["qwen_score"] = qwen_score(row)
        reviewed.append(row)
    return reviewed


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "review_id": row.get("review_id"),
        "source_id": row.get("source_id"),
        "reviewed_label": row.get("reviewed_label"),
        "qwen_label": row.get("qwen_label"),
        "qwen_score": row.get("qwen_score"),
        "corpus_label": row.get("corpus_label"),
        "audit_bucket": row.get("audit_bucket"),
        "source": row.get("source"),
        "domain": row.get("domain"),
        "subcluster_id": row.get("subcluster_id"),
        "prompt_text": row.get("prompt_text"),
    }


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    binary_rows = [
        row for row in rows if binary_label(str(row.get("reviewed_label"))) is not None
    ]
    labels = [int(binary_label(str(row["reviewed_label"]))) for row in binary_rows]
    scores = [float(row["qwen_score"]) for row in binary_rows]

    cross_tab: Counter[tuple[str, str]] = Counter()
    bucket_counts: Counter[str] = Counter()
    by_bucket: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    for row in rows:
        reviewed_label = str(row.get("reviewed_label"))
        qwen_label = str(row.get("qwen_label_normalized"))
        bucket = str(row.get("audit_bucket", "unknown"))
        cross_tab[(reviewed_label, qwen_label)] += 1
        bucket_counts[bucket] += 1
        by_bucket[bucket][(reviewed_label, qwen_label)] += 1

    errors = []
    for row in binary_rows:
        prediction = qwen_binary_prediction(row, threshold=0.5)
        label = binary_label(str(row["reviewed_label"]))
        if prediction != label:
            errors.append(compact_row(row))

    return {
        "num_reviewed_rows": len(rows),
        "num_binary_eval_rows": len(binary_rows),
        "reviewed_label_counts": dict(Counter(str(row["reviewed_label"]) for row in rows)),
        "qwen_label_counts": dict(Counter(str(row["qwen_label_normalized"]) for row in rows)),
        "audit_bucket_counts": dict(bucket_counts),
        "reviewed_vs_qwen_cross_tab": {
            " | ".join(key): value for key, value in cross_tab.most_common()
        },
        "reviewed_vs_qwen_by_bucket": {
            bucket: {" | ".join(key): value for key, value in counter.most_common()}
            for bucket, counter in sorted(by_bucket.items())
        },
        "binary_qwen_score_auc": auc(labels, scores),
        "binary_metrics_threshold_0_5": binary_metrics(labels, scores, threshold=0.5),
        "binary_errors_threshold_0_5": errors,
        "interpretation": (
            "This is a small reviewed-label v0 slice from disagreement candidates. It validates "
            "the audit loop and gives a first prompt-classifier measurement against human labels, "
            "but it is intentionally not a representative production benchmark."
        ),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(compact_row(row), separators=(",", ":")) + "\n")


def markdown_report(report: dict[str, Any]) -> str:
    metrics = report["binary_metrics_threshold_0_5"]
    auc_value = report["binary_qwen_score_auc"]
    auc_text = "n/a" if auc_value is None else f"{auc_value:.4f}"
    lines = [
        "# Reviewed Prompt-Label Evaluation",
        "",
        report["interpretation"],
        "",
        f"Reviewed rows: {report['num_reviewed_rows']}",
        f"Binary safe/unsafe eval rows: {report['num_binary_eval_rows']}",
        "",
        "## Qwen3Guard Binary Metrics",
        "",
        "| AUC | Accuracy | Precision | Recall | FPR | Threshold |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {auc_text} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['false_positive_rate']:.4f} | "
            f"{metrics['threshold']:.2f} |"
        ),
        "",
        "## Reviewed Labels",
        "",
        "| Label | Rows |",
        "| --- | ---: |",
    ]
    for label, count in sorted(report["reviewed_label_counts"].items()):
        lines.append(f"| `{label}` | {count} |")
    lines.extend(["", "## Reviewed vs Qwen", "", "| Pair | Rows |", "| --- | ---: |"])
    for pair, count in report["reviewed_vs_qwen_cross_tab"].items():
        lines.append(f"| `{pair}` | {count} |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    reviewed_rows = load_reviewed_rows(Path(args.review_csv))
    if not reviewed_rows:
        raise ValueError("no reviewed rows found")
    report = evaluate(reviewed_rows)
    report["review_csv"] = args.review_csv

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote reviewed prompt-label evaluation to {output_json}")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown_report(report) + "\n", encoding="utf-8")
        print(f"wrote reviewed prompt-label markdown to {output_md}")

    if args.reviewed_jsonl:
        write_jsonl(Path(args.reviewed_jsonl), reviewed_rows)
        print(f"wrote reviewed prompt-label JSONL to {args.reviewed_jsonl}")

    metrics = report["binary_metrics_threshold_0_5"]
    auc_value = report["binary_qwen_score_auc"]
    auc_text = "n/a" if auc_value is None else f"{auc_value:.4f}"
    print(
        f"reviewed_rows={report['num_reviewed_rows']} binary_rows={report['num_binary_eval_rows']} "
        f"auc={auc_text} accuracy={metrics['accuracy']:.4f} recall={metrics['recall']:.4f} "
        f"fpr={metrics['false_positive_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
