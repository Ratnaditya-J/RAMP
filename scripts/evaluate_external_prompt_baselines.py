#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate prompt-classifier baselines against benchmark-derived corpus labels."
        )
    )
    parser.add_argument("--corpus", required=True)
    parser.add_argument(
        "--score",
        action="append",
        required=True,
        help="Named score file in the form name=path.jsonl. Repeat for each baseline.",
    )
    parser.add_argument("--score-key", default="prompt_risk_score")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--top-k-errors", type=int, default=50)
    return parser.parse_args()


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
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


def label_to_binary(label: Any) -> int | None:
    value = str(label).strip().lower()
    if value in {"unsafe", "harmful", "1", "true"}:
        return 1
    if value in {"safe", "benign", "0", "false"}:
        return 0
    return None


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
    reports = [metrics(labels, scores, idx / 100) for idx in range(101)]
    return {
        "default_0_5": metrics(labels, scores, 0.5),
        "best_f1": max(reports, key=lambda item: (item["f1"], item["accuracy"])),
        "target_fpr_0_05": max(
            (item for item in reports if item["false_positive_rate"] <= 0.05),
            key=lambda item: (item["recall"], item["precision"], item["accuracy"]),
            default=min(reports, key=lambda item: item["false_positive_rate"]),
        ),
        "target_fpr_0_10": max(
            (item for item in reports if item["false_positive_rate"] <= 0.10),
            key=lambda item: (item["recall"], item["precision"], item["accuracy"]),
            default=min(reports, key=lambda item: item["false_positive_rate"]),
        ),
        "target_recall_0_90": min(
            (item for item in reports if item["recall"] >= 0.90),
            key=lambda item: (item["false_positive_rate"], -item["precision"]),
            default=max(reports, key=lambda item: item["recall"]),
        ),
        "target_recall_0_95": min(
            (item for item in reports if item["recall"] >= 0.95),
            key=lambda item: (item["false_positive_rate"], -item["precision"]),
            default=max(reports, key=lambda item: item["recall"]),
        ),
    }


def compact_error(row: dict[str, Any], score: float) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "label": row.get("label"),
        "score": score,
        "source": row.get("source"),
        "domain": row.get("domain"),
        "subcluster_role": row.get("subcluster_role"),
        "subcluster_id": row.get("subcluster_id"),
        "span_text": row.get("span_text"),
    }


def join_rows(
    corpus_rows: dict[str, dict[str, Any]],
    score_rows: dict[str, dict[str, Any]],
    score_key: str,
) -> list[dict[str, Any]]:
    rows = []
    for row_id in sorted(set(corpus_rows) & set(score_rows)):
        label = label_to_binary(corpus_rows[row_id].get("label"))
        score = score_rows[row_id].get(score_key)
        if label is None or score is None:
            continue
        row = dict(corpus_rows[row_id])
        row["score"] = float(score)
        row["binary_label"] = label
        row["prompt_label"] = score_rows[row_id].get("prompt_label")
        row["prompt_classifier_version"] = score_rows[row_id].get(
            "prompt_classifier_version"
        )
        rows.append(row)
    return rows


def evaluate_rows(rows: list[dict[str, Any]], *, top_k_errors: int) -> dict[str, Any]:
    labels = [int(row["binary_label"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    false_positives = [
        compact_error(row, score)
        for row, label, score in zip(rows, labels, scores, strict=True)
        if label == 0 and score >= 0.5
    ]
    false_negatives = [
        compact_error(row, score)
        for row, label, score in zip(rows, labels, scores, strict=True)
        if label == 1 and score < 0.5
    ]
    return {
        "rows": len(rows),
        "label_counts": dict(Counter(str(row.get("label")) for row in rows)),
        "auc": auc(labels, scores),
        "thresholds": threshold_sweep(labels, scores),
        "score_distribution": {
            "mean": statistics.fmean(scores) if scores else None,
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
        },
        "false_positives_at_0_5": sorted(
            false_positives,
            key=lambda row: row["score"],
            reverse=True,
        )[:top_k_errors],
        "false_negatives_at_0_5": sorted(false_negatives, key=lambda row: row["score"])[
            :top_k_errors
        ],
    }


def grouped_reports(
    rows: list[dict[str, Any]],
    *,
    key: str,
    top_k_errors: int,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    return {
        name: evaluate_rows(group_rows, top_k_errors=top_k_errors)
        for name, group_rows in sorted(grouped.items())
    }


def parse_score_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--score must be in the form name=path.jsonl")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise ValueError("--score must be in the form name=path.jsonl")
    return name.strip(), Path(path)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    corpus_rows = load_jsonl_by_id(Path(args.corpus))
    baseline_reports = {}
    for score_arg in args.score:
        name, path = parse_score_arg(score_arg)
        score_rows = load_jsonl_by_id(path)
        rows = join_rows(corpus_rows, score_rows, args.score_key)
        baseline_reports[name] = {
            "score_path": str(path),
            "score_key": args.score_key,
            "coverage": {
                "corpus_rows": len(corpus_rows),
                "score_rows": len(score_rows),
                "joined_binary_rows": len(rows),
            },
            "overall": evaluate_rows(rows, top_k_errors=args.top_k_errors),
            "by_source": grouped_reports(
                rows,
                key="source",
                top_k_errors=min(args.top_k_errors, 10),
            ),
            "by_domain": grouped_reports(
                rows,
                key="domain",
                top_k_errors=min(args.top_k_errors, 10),
            ),
        }
    return {
        "artifact_id": "ramp_external_prompt_baselines_v0.1",
        "corpus": args.corpus,
        "baselines": baseline_reports,
        "interpretation": (
            "These are prompt-classifier baselines over benchmark-derived corpus labels. "
            "They are external to the reviewed-label calibration loop, but inherit each "
            "source dataset's label quality and mapping assumptions."
        ),
    }


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# External Prompt Baseline Evaluation",
        "",
        report["interpretation"],
        "",
        (
            "| Baseline | Rows | AUC | Recall@5%FPR | Recall@10%FPR | "
            "FPR@90%Recall | FPR@95%Recall | F1@0.5 |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, baseline in report["baselines"].items():
        overall = baseline["overall"]
        thresholds = overall["thresholds"]
        lines.append(
            f"| `{name}` | {overall['rows']} | {pct(overall['auc'])} | "
            f"{pct(thresholds['target_fpr_0_05']['recall'])} | "
            f"{pct(thresholds['target_fpr_0_10']['recall'])} | "
            f"{pct(thresholds['target_recall_0_90']['false_positive_rate'])} | "
            f"{pct(thresholds['target_recall_0_95']['false_positive_rate'])} | "
            f"{pct(thresholds['default_0_5']['f1'])} |"
        )
    lines.extend(["", "## Source Slices", ""])
    for name, baseline in report["baselines"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                "| Source | Rows | AUC | Recall@5%FPR | F1@0.5 |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for source, source_report in baseline["by_source"].items():
            lines.append(
                f"| `{source}` | {source_report['rows']} | {pct(source_report['auc'])} | "
                f"{pct(source_report['thresholds']['target_fpr_0_05']['recall'])} | "
                f"{pct(source_report['thresholds']['default_0_5']['f1'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = build_report(args)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote external prompt baseline report to {output_json}")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown_report(report) + "\n")
        print(f"wrote external prompt baseline markdown to {output_md}")
    for name, baseline in report["baselines"].items():
        overall = baseline["overall"]
        target = overall["thresholds"]["target_fpr_0_05"]
        print(
            f"{name}: auc={overall['auc']:.4f} "
            f"recall@5fpr={target['recall']:.4f} "
            f"f1@0.5={overall['thresholds']['default_0_5']['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
