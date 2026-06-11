#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluate_session_classifier_scores import session_id_from_score
from evaluate_session_risk import binary_label, metrics, threshold_sweep, turn_risk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate calibrated session-signal fusion across single-turn, compact-session, "
            "and full-transcript classifier scores."
        )
    )
    parser.add_argument("--session-corpus", required=True)
    parser.add_argument("--compact-scores", default=None)
    parser.add_argument("--full-transcript-scores", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--target-fpr", type=float, default=0.25)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def score_by_session(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    return {session_id_from_score(row): row for row in load_jsonl(Path(path))}


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_rows(
    sessions: list[dict[str, Any]],
    compact_scores: dict[str, dict[str, Any]],
    full_scores: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session in sessions:
        session_id = str(session.get("session_id") or "")
        turns = list(session.get("turns") or [])
        compact = compact_scores.get(session_id)
        full = full_scores.get(session_id)
        rows.append(
            {
                "session_id": session_id,
                "source": session.get("source"),
                "session_label": session.get("session_label"),
                "session_risk_type": session.get("session_risk_type"),
                "turn_count": len(turns),
                "single_turn_max_score": max((turn_risk(turn) for turn in turns), default=0.0),
                "compact_session_classifier_score": safe_float(
                    compact.get("prompt_risk_score") if compact else None
                ),
                "full_transcript_session_classifier_score": safe_float(
                    full.get("prompt_risk_score") if full else None
                ),
                "compact_score_row_matched": compact is not None,
                "full_score_row_matched": full is not None,
            }
        )
    return rows


def add_fusion_scores(
    rows: list[dict[str, Any]],
    weights: dict[str, float],
    output_key: str,
) -> None:
    for row in rows:
        row[output_key] = (
            weights.get("single_turn_max", 0.0) * row["single_turn_max_score"]
            + weights.get("compact_session_classifier", 0.0)
            * row["compact_session_classifier_score"]
            + weights.get("full_transcript_session_classifier", 0.0)
            * row["full_transcript_session_classifier_score"]
        )


def weight_values(step: float) -> list[float]:
    if step <= 0 or step > 1:
        raise ValueError("--weight-step must be in (0, 1].")
    count = int(round(1.0 / step))
    return [round(i * step, 10) for i in range(count + 1)]


def threshold_values(step: float) -> list[float]:
    if step <= 0 or step > 1:
        raise ValueError("--threshold-step must be in (0, 1].")
    count = int(round(1.0 / step))
    return [round(i * step, 10) for i in range(count + 1)]


def candidate_weight_sets(score_keys: list[str], step: float) -> list[dict[str, float]]:
    values = weight_values(step)
    if len(score_keys) == 1:
        return [{score_keys[0]: 1.0}]
    if len(score_keys) == 2:
        first, second = score_keys
        return [{first: value, second: round(1.0 - value, 10)} for value in values]
    if len(score_keys) == 3:
        first, second, third = score_keys
        candidates = []
        for first_value in values:
            for second_value in values:
                third_value = round(1.0 - first_value - second_value, 10)
                if third_value < -1e-9:
                    continue
                if third_value > 1.0 + 1e-9:
                    continue
                candidates.append(
                    {
                        first: round(first_value, 10),
                        second: round(second_value, 10),
                        third: max(0.0, third_value),
                    }
                )
        return candidates
    raise ValueError("Only one to three session score keys are supported.")


def calibrate(
    base_rows: list[dict[str, Any]],
    score_keys: list[str],
    weight_step: float,
    threshold_step: float,
    target_fpr: float,
) -> dict[str, Any] | None:
    if not any(binary_label(row.get("session_label")) == 0 for row in base_rows):
        return None
    if not any(binary_label(row.get("session_label")) == 1 for row in base_rows):
        return None

    best_f1: dict[str, Any] | None = None
    best_target_fpr: dict[str, Any] | None = None
    thresholds = threshold_values(threshold_step)
    for weights in candidate_weight_sets(score_keys, weight_step):
        rows = [dict(row) for row in base_rows]
        add_fusion_scores(rows, weights, "calibrated_session_fusion_score")
        for threshold in thresholds:
            metric = metrics(rows, "calibrated_session_fusion_score", threshold)
            candidate = {
                "weights": weights,
                "threshold": threshold,
                "metrics": metric,
            }
            if best_f1 is None or (
                metric["f1"],
                metric["recall"],
                -metric["false_positive_rate"],
                metric["accuracy"],
            ) > (
                best_f1["metrics"]["f1"],
                best_f1["metrics"]["recall"],
                -best_f1["metrics"]["false_positive_rate"],
                best_f1["metrics"]["accuracy"],
            ):
                best_f1 = candidate
            if metric["false_positive_rate"] <= target_fpr and (
                best_target_fpr is None
                or (
                    metric["recall"],
                    metric["f1"],
                    metric["accuracy"],
                    -threshold,
                )
                > (
                    best_target_fpr["metrics"]["recall"],
                    best_target_fpr["metrics"]["f1"],
                    best_target_fpr["metrics"]["accuracy"],
                    -best_target_fpr["threshold"],
                )
            ):
                best_target_fpr = candidate
    return {
        "score_keys": score_keys,
        "weight_step": weight_step,
        "threshold_step": threshold_step,
        "target_fpr": target_fpr,
        "selected_by_best_f1": best_f1,
        "selected_by_target_fpr": best_target_fpr,
    }


def catches(
    rows: list[dict[str, Any]], score_key: str, threshold: float, baseline_key: str
) -> list[dict[str, Any]]:
    return [
        {
            "session_id": row["session_id"],
            "source": row["source"],
            "session_label": row["session_label"],
            "session_risk_type": row["session_risk_type"],
            baseline_key: row[baseline_key],
            score_key: row[score_key],
        }
        for row in rows
        if binary_label(row.get("session_label")) == 1
        and row[baseline_key] < threshold
        and row[score_key] >= threshold
    ]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    rows = build_rows(
        load_jsonl(Path(args.session_corpus)),
        score_by_session(args.compact_scores),
        score_by_session(args.full_transcript_scores),
    )
    score_keys = ["single_turn_max_score"]
    if args.compact_scores:
        score_keys.append("compact_session_classifier_score")
    if args.full_transcript_scores:
        score_keys.append("full_transcript_session_classifier_score")

    for row in rows:
        row["max_session_signal_score"] = max(row[key] for key in score_keys)

    named_scores = {
        "single_turn_max": "single_turn_max_score",
        "max_session_signal": "max_session_signal_score",
    }
    if args.compact_scores:
        named_scores["compact_session_classifier"] = "compact_session_classifier_score"
    if args.full_transcript_scores:
        named_scores["full_transcript_session_classifier"] = (
            "full_transcript_session_classifier_score"
        )

    calibration = calibrate(
        rows,
        [
            key.removesuffix("_score")
            for key in score_keys
            if key != "single_turn_max_score"
        ]
        + (["single_turn_max"] if "single_turn_max_score" in score_keys else []),
        args.weight_step,
        args.threshold_step,
        args.target_fpr,
    )
    if calibration:
        selected = calibration["selected_by_best_f1"]
        weights = selected["weights"]
        add_fusion_scores(rows, weights, "calibrated_session_fusion_score")
        named_scores["calibrated_session_fusion"] = "calibrated_session_fusion_score"

    report = {
        "artifact_id": "ramp_session_signal_fusion_eval_v0.1",
        "num_sessions": len(rows),
        "threshold": args.threshold,
        "label_counts": dict(Counter(str(row.get("session_label")) for row in rows)),
        "matched_scores": {
            "compact": sum(1 for row in rows if row["compact_score_row_matched"]),
            "full_transcript": sum(1 for row in rows if row["full_score_row_matched"]),
        },
        "score_columns": named_scores,
        "metrics": {},
        "threshold_sweeps": {},
        "single_turn_false_negatives_caught": {},
        "calibration": calibration,
        "scored_sessions": rows,
    }
    for name, key in named_scores.items():
        report["metrics"][name] = metrics(rows, key, args.threshold)
        report["threshold_sweeps"][name] = threshold_sweep(rows, key)
        report["single_turn_false_negatives_caught"][name] = catches(
            rows, key, args.threshold, "single_turn_max_score"
        )
    return report


def format_auc(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Session Signal Fusion Evaluation v0.1",
        "",
        f"Sessions: {report['num_sessions']}",
        f"Threshold: {report['threshold']}",
        f"Matched compact scores: {report['matched_scores']['compact']}",
        f"Matched full-transcript scores: {report['matched_scores']['full_transcript']}",
        "",
        (
            "| Condition | AUC | Accuracy | Recall | FPR | F1 | TP | FP | TN | FN | "
            "Single-turn FNs caught |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metric in report["metrics"].items():
        lines.append(
            "| `{}` | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {} | {} | {} | {} | {} |".format(
                name,
                format_auc(report["threshold_sweeps"][name]["auc"]),
                metric["accuracy"],
                metric["recall"],
                metric["false_positive_rate"],
                metric["f1"],
                metric["tp"],
                metric["fp"],
                metric["tn"],
                metric["fn"],
                len(report["single_turn_false_negatives_caught"][name]),
            )
        )
    calibration = report.get("calibration")
    if calibration:
        lines.extend(["", "## Calibration", ""])
        for label in ("selected_by_best_f1", "selected_by_target_fpr"):
            selected = calibration.get(label)
            if not selected:
                continue
            metric = selected["metrics"]
            lines.append(
                (
                    "- `{}`: weights `{}`, threshold `{:.2f}`, recall `{:.4f}`, "
                    "FPR `{:.4f}`, F1 `{:.4f}`"
                ).format(
                    label,
                    selected["weights"],
                    selected["threshold"],
                    metric["recall"],
                    metric["false_positive_rate"],
                    metric["f1"],
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = build_report(args)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(Path(args.output_md), report)
    print(f"wrote session signal fusion report to {args.output_json}")
    if args.output_md:
        print(f"wrote session signal fusion markdown to {args.output_md}")
    for name, metric in report["metrics"].items():
        auc = format_auc(report["threshold_sweeps"][name]["auc"])
        print(
            f"{name}: auc={auc} recall={metric['recall']:.4f} "
            f"fpr={metric['false_positive_rate']:.4f} f1={metric['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
