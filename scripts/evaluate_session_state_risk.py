#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluate_session_risk import (
    binary_label,
    compact_error,
    metrics,
    threshold_sweep,
    turn_risk,
)

from ramp.features.session_state_risk import (
    CompactSessionRiskScorer,
    FullTranscriptLexicalSessionScorer,
    SessionStateUpdater,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate compact session-state risk against full-transcript baseline."
    )
    parser.add_argument("--session-corpus", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--threshold", type=float, default=0.55)
    return parser.parse_args()


def load_sessions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def score_sessions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updater = SessionStateUpdater()
    compact_scorer = CompactSessionRiskScorer()
    full_scorer = FullTranscriptLexicalSessionScorer()
    scored_rows = []
    for session in rows:
        scored = dict(session)
        turns = list(session.get("turns") or [])
        risks = [turn_risk(turn) for turn in turns]
        compact_state = updater.update(session)
        compact_score, compact_contributions = compact_scorer.score(compact_state)
        full_score, full_contributions = full_scorer.score(session)
        scored.update(
            {
                "single_turn_max_score": max(risks) if risks else 0.0,
                "compact_session_state_score": compact_score,
                "full_transcript_lexical_score": full_score,
                "compact_session_state": {
                    "turn_count": compact_state.turn_count,
                    "domains_seen": compact_state.domains_seen,
                    "subclusters_seen": compact_state.subclusters_seen,
                    "highest_severity": compact_state.highest_severity,
                    "risk_trend": compact_state.risk_trend,
                    "intent_progression": compact_state.intent_progression,
                    "evasion_attempts": compact_state.evasion_attempts,
                    "operational_details_requested": compact_state.operational_details_requested,
                    "benign_cover_story": compact_state.benign_cover_story,
                    "cross_turn_composition": compact_state.cross_turn_composition,
                    "safety_summary": compact_state.safety_summary,
                    "salient_turns": [
                        {
                            "turn_id": turn.turn_id,
                            "turn_index": turn.turn_index,
                            "role": turn.role,
                            "text": turn.text,
                            "risk_score": turn.risk_score,
                            "salience_reasons": turn.salience_reasons,
                        }
                        for turn in compact_state.salient_turns
                    ],
                },
                "compact_session_contributions": compact_contributions,
                "full_transcript_contributions": full_contributions,
            }
        )
        scored_rows.append(scored)
    return scored_rows


def build_report(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    scored_rows = score_sessions(rows)
    score_keys = {
        "single_turn_max": "single_turn_max_score",
        "compact_session_state": "compact_session_state_score",
        "full_transcript_lexical": "full_transcript_lexical_score",
    }
    metric_table = {
        name: metrics(scored_rows, score_key, threshold)
        for name, score_key in score_keys.items()
    }
    sweeps = {
        name: threshold_sweep(scored_rows, score_key)
        for name, score_key in score_keys.items()
    }
    compact_catches = [
        compact_error(row, "compact_session_state_score")
        for row in scored_rows
        if binary_label(row.get("session_label")) == 1
        and row["single_turn_max_score"] < threshold
        and row["compact_session_state_score"] >= threshold
    ]
    full_catches = [
        compact_error(row, "full_transcript_lexical_score")
        for row in scored_rows
        if binary_label(row.get("session_label")) == 1
        and row["single_turn_max_score"] < threshold
        and row["full_transcript_lexical_score"] >= threshold
    ]
    return {
        "artifact_id": "ramp_session_state_risk_eval_v0.1",
        "num_sessions": len(scored_rows),
        "threshold": threshold,
        "label_counts": dict(Counter(str(row.get("session_label")) for row in scored_rows)),
        "source_counts": dict(Counter(str(row.get("source")) for row in scored_rows)),
        "metrics": metric_table,
        "threshold_sweeps": sweeps,
        "single_turn_false_negatives_caught": {
            "compact_session_state": compact_catches,
            "full_transcript_lexical": full_catches,
        },
        "scored_sessions": scored_rows,
    }


def format_auc(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Session-State Risk Evaluation v0.1",
        "",
        f"Sessions: {report['num_sessions']}",
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
            "## False Negatives Caught Over Single-Turn Max",
            "",
            "| Condition | Count |",
            "| --- | ---: |",
        ]
    )
    for name, rows in report["single_turn_false_negatives_caught"].items():
        lines.append(f"| `{name}` | {len(rows)} |")
    lines.extend(
        [
            "",
            "## Threshold Sweep",
            "",
            "| Condition | Operating Point | Threshold | F1 | Recall | FPR |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, sweep in report["threshold_sweeps"].items():
        best = sweep["best_f1"]
        lines.append(
            "| `{}` | best_f1 | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                name,
                best["threshold"],
                best["f1"],
                best["recall"],
                best["false_positive_rate"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = build_report(load_sessions(Path(args.session_corpus)), args.threshold)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown(Path(args.output_md), report)
    print(f"wrote session-state report to {args.output_json}")
    if args.output_md:
        print(f"wrote session-state markdown to {args.output_md}")
    for name, metric in report["metrics"].items():
        auc = format_auc(report["threshold_sweeps"][name]["auc"])
        print(
            f"{name}: auc={auc} recall={metric['recall']:.4f} "
            f"fpr={metric['false_positive_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
