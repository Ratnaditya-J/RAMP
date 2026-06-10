#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluate_session_risk import binary_label, metrics, threshold_sweep, turn_risk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate session-level classifier scores against session labels."
    )
    parser.add_argument("--session-corpus", required=True)
    parser.add_argument("--session-scores", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--score-name", default="compact_session_classifier")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def session_id_from_score(row: dict[str, Any]) -> str:
    if row.get("session_id"):
        return str(row["session_id"])
    row_id = str(row.get("id") or "")
    return row_id.rsplit(".", 1)[0]


def build_rows(
    sessions: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    score_name: str,
) -> list[dict[str, Any]]:
    score_by_session = {session_id_from_score(row): row for row in scores}
    rows = []
    for session in sessions:
        session_id = str(session.get("session_id") or "")
        score_row = score_by_session.get(session_id)
        turns = list(session.get("turns") or [])
        risks = [turn_risk(turn) for turn in turns]
        classifier_score = (
            float(score_row.get("prompt_risk_score") or 0.0) if score_row else 0.0
        )
        rows.append(
            {
                "session_id": session_id,
                "source": session.get("source"),
                "session_label": session.get("session_label"),
                "session_risk_type": session.get("session_risk_type"),
                "single_turn_max_score": max(risks) if risks else 0.0,
                f"{score_name}_score": classifier_score,
                f"{score_name}_label": score_row.get("prompt_label") if score_row else None,
                f"{score_name}_version": score_row.get("prompt_classifier_version")
                if score_row
                else None,
                "score_row_matched": score_row is not None,
            }
        )
    return rows


def build_report(
    sessions: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    threshold: float,
    score_name: str,
) -> dict[str, Any]:
    rows = build_rows(sessions, scores, score_name)
    score_key = f"{score_name}_score"
    caught = [
        {
            "session_id": row["session_id"],
            "source": row["source"],
            "session_label": row["session_label"],
            "single_turn_max_score": row["single_turn_max_score"],
            score_key: row[score_key],
        }
        for row in rows
        if binary_label(row.get("session_label")) == 1
        and row["single_turn_max_score"] < threshold
        and row[score_key] >= threshold
    ]
    return {
        "artifact_id": "ramp_session_classifier_score_eval_v0.1",
        "score_name": score_name,
        "num_sessions": len(rows),
        "matched_scores": sum(1 for row in rows if row["score_row_matched"]),
        "threshold": threshold,
        "label_counts": dict(Counter(str(row.get("session_label")) for row in rows)),
        "metrics": {
            "single_turn_max": metrics(rows, "single_turn_max_score", threshold),
            score_name: metrics(rows, score_key, threshold),
        },
        "threshold_sweeps": {
            "single_turn_max": threshold_sweep(rows, "single_turn_max_score"),
            score_name: threshold_sweep(rows, score_key),
        },
        "single_turn_false_negatives_caught": caught,
        "scored_sessions": rows,
    }


def format_auc(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Session Classifier Score Evaluation v0.1",
        "",
        f"Sessions: {report['num_sessions']}",
        f"Matched scores: {report['matched_scores']}",
        f"Threshold: {report['threshold']}",
        "",
        "| Condition | AUC | Accuracy | Recall | FPR | TP | FP | TN | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metric in report["metrics"].items():
        lines.append(
            "| `{}` | {} | {:.4f} | {:.4f} | {:.4f} | {} | {} | {} | {} |".format(
                name,
                format_auc(report["threshold_sweeps"][name]["auc"]),
                metric["accuracy"],
                metric["recall"],
                metric["false_positive_rate"],
                metric["tp"],
                metric["fp"],
                metric["tn"],
                metric["fn"],
            )
        )
    lines.extend(
        [
            "",
            "Single-turn false negatives caught by session classifier: "
            f"{len(report['single_turn_false_negatives_caught'])}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = build_report(
        load_jsonl(Path(args.session_corpus)),
        load_jsonl(Path(args.session_scores)),
        args.threshold,
        args.score_name,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(Path(args.output_md), report)
    print(f"wrote session classifier score report to {args.output_json}")
    if args.output_md:
        print(f"wrote session classifier score markdown to {args.output_md}")
    for name, metric in report["metrics"].items():
        auc = format_auc(report["threshold_sweeps"][name]["auc"])
        print(
            f"{name}: auc={auc} recall={metric['recall']:.4f} "
            f"fpr={metric['false_positive_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
