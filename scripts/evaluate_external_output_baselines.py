#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_external_prompt_baselines import (
    auc,
    compact_error,
    grouped_reports,
    parse_score_arg,
    pct,
    threshold_sweep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate output-classifier baselines against native response labels."
    )
    parser.add_argument("--corpus", required=True)
    parser.add_argument(
        "--score",
        action="append",
        required=True,
        help="Named output-score file in the form name=path.jsonl.",
    )
    parser.add_argument("--score-key", default="output_risk_score")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--top-k-errors", type=int, default=50)
    return parser.parse_args()


def load_jsonl_by_keys(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            for key_name in ("eval_id", "id", "source_id"):
                row_id = row.get(key_name)
                if row_id is not None:
                    rows[str(row_id)] = row
    return rows


def load_corpus_by_primary_key(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[corpus_key(row)] = row
    return rows


def label_to_binary(label: Any) -> int | None:
    value = str(label).strip().lower()
    if value in {"unsafe", "harmful", "1", "true"}:
        return 1
    if value in {"safe", "benign", "0", "false"}:
        return 0
    return None


def corpus_key(row: dict[str, Any]) -> str:
    return str(row.get("eval_id") or row.get("id") or row.get("source_id"))


def join_rows(
    corpus_rows: dict[str, dict[str, Any]],
    score_rows: dict[str, dict[str, Any]],
    score_key: str,
) -> list[dict[str, Any]]:
    rows = []
    for row_id in sorted(corpus_rows):
        corpus_row = corpus_rows[row_id]
        score_row = score_rows.get(row_id)
        label = label_to_binary(
            corpus_row.get("output_reviewed_label") or corpus_row.get("label")
        )
        score = score_row.get(score_key) if score_row else None
        if label is None or score is None:
            continue
        row = dict(corpus_row)
        row["score"] = float(score)
        row["binary_label"] = label
        row["output_label"] = score_row.get("output_label")
        row["output_classifier_version"] = score_row.get("output_classifier_version")
        row["span_text"] = row.get("output_text")
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
        "label_counts": {
            "safe": sum(1 for label in labels if label == 0),
            "unsafe": sum(1 for label in labels if label == 1),
        },
        "auc": auc(labels, scores),
        "thresholds": threshold_sweep(labels, scores),
        "false_positives_at_0_5": sorted(
            false_positives,
            key=lambda row: row["score"],
            reverse=True,
        )[:top_k_errors],
        "false_negatives_at_0_5": sorted(false_negatives, key=lambda row: row["score"])[
            :top_k_errors
        ],
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    corpus_rows = load_corpus_by_primary_key(Path(args.corpus))
    baseline_reports = {}
    for score_arg in args.score:
        name, path = parse_score_arg(score_arg)
        score_rows = load_jsonl_by_keys(path)
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
        }
    return {
        "artifact_id": "ramp_external_output_baselines_v0.1",
        "corpus": args.corpus,
        "baselines": baseline_reports,
        "interpretation": (
            "These are output-classifier baselines over native response labels. "
            "They evaluate generated/benchmark response text, not prompt text."
        ),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# External Output Baseline Evaluation",
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
    print(f"wrote external output baseline report to {output_json}")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown_report(report) + "\n")
        print(f"wrote external output baseline markdown to {output_md}")
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
