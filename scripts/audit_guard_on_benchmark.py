#!/usr/bin/env python
"""Audit a published guard on a native-labeled, multi-source public benchmark.

Companion to audit_external_guard.py (which uses RAMP's adaptive/blind split). This one
takes an EXTERNAL benchmark with its OWN labels and natural sub-sources (e.g. ToxicChat's
two temporal releases), so the result does not depend on RAMP's judge labels — fixing the
main caveat of the v0.1 audit. It measures per-source performance and cross-source
threshold transfer (does an operating point tuned on one source survive on another?).

Reuses the metric and verdict logic from audit_external_guard.py. Zero SIEVE import.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from audit_external_guard import (
    VERDICT_CLAIMS,
    _hash,
    derive_verdict,
    evaluate_set,
    load_guard,
    per_source,
    source_shift_transfer,
)


def load_corpus(path: Path) -> dict[str, tuple[int, str]]:
    """source_id -> (label_int, source) from a CSV with source_id,source,label columns."""
    rows: dict[str, tuple[int, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as input_file:
        for row in csv.DictReader(input_file):
            source_id = str(row.get("source_id", "")).strip()
            label = str(row.get("label", "")).strip().lower()
            source = str(row.get("source", "unknown")).strip()
            if not source_id or label not in {"safe", "unsafe"}:
                continue
            rows[source_id] = (1 if label == "unsafe" else 0, source)
    return rows


def build_card(report: dict[str, Any], *, guard_name: str, benchmark: str) -> dict[str, Any]:
    claims = VERDICT_CLAIMS[report["verdict"]["verdict"]]
    return {
        "card_version": "0.1",
        "axis": "evaluation_robustness",
        "verdict_vocabulary": list(VERDICT_CLAIMS.keys()),
        "scope": {
            "signal": guard_name,
            "signal_description": (
                f"published external guard ({guard_name}) on {benchmark} native labels"
            ),
            "n": report["overall"]["n"],
            "axis_specific": {
                "benchmark": benchmark,
                "sources": sorted(report["per_source"].keys()),
                "label_provenance": "benchmark_native",
                "guard_kind": report["guard_kind"],
            },
        },
        "verdict": report["verdict"]["verdict"],
        "status": "ok",
        "diagnostics": {
            "overall": report["overall"],
            "decision_reasons": report["verdict"]["reasons"],
        },
        "allowed_claims": claims["allowed"],
        "disallowed_claims": claims["disallowed"],
        "residual_risks": claims["residual"]
        + [
            f"native {benchmark} labels; binary guards limited to operating-point analysis",
        ],
        "protocol_version": "ramp_guard_benchmark_audit_v0.1",
        "config_hash": _hash(
            {"target_fpr": report["target_fpr"], "threshold": report["threshold"]}
        ),
        "inputs_hash": _hash(report["inputs"]),
    }


def markdown(report: dict[str, Any], card: dict[str, Any]) -> str:
    def fmt(x: Any) -> str:
        return "n/a" if x is None else f"{x:.3f}"

    o = report["overall"]
    benchmark = card["scope"]["axis_specific"]["benchmark"]
    lines = [
        f"# Guard Benchmark Audit — {card['scope']['signal']} on {benchmark}",
        "",
        f"Verdict: **{report['verdict']['verdict']}** ({'; '.join(report['verdict']['reasons'])})",
        "",
        f"Overall (native labels): n={o['n']}, AUROC={fmt(o['auc'])}, recall={fmt(o['recall'])}, "
        f"FPR={fmt(o['fpr'])}, F1={fmt(o['f1'])}",
        "",
        "## Per-source",
        "",
        "| Source | n | unsafe rate | AUROC | recall | FPR | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source, m in sorted(report["per_source"].items()):
        lines.append(
            f"| {source} | {m['n']} | {fmt(m['unsafe_rate'])} | {fmt(m['auc'])} | "
            f"{fmt(m['recall'])} | {fmt(m['fpr'])} | {fmt(m['f1'])} |"
        )
    if report["source_shift"]:
        lines += [
            "",
            "## Cross-source threshold transfer (tune on others, apply to held-out)",
            "",
            "| Held-out source | calib FPR | held-out FPR | calib recall | held-out recall |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for src, t in sorted(report["source_shift"].items()):
            lines.append(
                f"| {src} | {fmt(t['calibration_fpr'])} | {fmt(t['heldout_fpr'])} | "
                f"{fmt(t['calibration_recall'])} | {fmt(t['heldout_recall'])} |"
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a guard on a native-labeled benchmark.")
    parser.add_argument(
        "--labeled-corpus", required=True, help="CSV: source_id,source,label,prompt_text."
    )
    parser.add_argument("--guard-scores", required=True)
    parser.add_argument("--guard-name", required=True)
    parser.add_argument("--guard-kind", choices=["binary", "continuous"], required=True)
    parser.add_argument("--benchmark", default="toxicchat")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target-fpr", type=float, default=0.10)
    parser.add_argument("--min-class", type=int, default=20)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--card-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = load_corpus(Path(args.labeled_corpus))
    scores = load_guard(Path(args.guard_scores))
    rows = [(i, corpus[i][0], scores[i], corpus[i][1]) for i in corpus if i in scores]
    if not rows:
        raise ValueError("no rows joined between corpus and guard scores")

    overall = evaluate_set(rows, args.threshold)
    shift = (
        source_shift_transfer(rows, target_fpr=args.target_fpr, min_class=args.min_class)
        if args.guard_kind == "continuous"
        else {}
    )
    # Reuse the verdict rule; with no adaptive/blind split, pass overall as both so the
    # eval-inflation term is zero and source shift drives the verdict.
    verdict = derive_verdict(overall, overall, shift)
    report = {
        "guard_name": args.guard_name,
        "guard_kind": args.guard_kind,
        "benchmark": args.benchmark,
        "threshold": args.threshold,
        "target_fpr": args.target_fpr,
        "overall": overall,
        "per_source": per_source(rows, args.threshold),
        "source_shift": shift,
        "verdict": verdict,
        "inputs": {"labeled_corpus": args.labeled_corpus, "guard_scores": args.guard_scores},
        "n_joined": len(rows),
    }
    card = build_card(report, guard_name=args.guard_name, benchmark=args.benchmark)

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote benchmark audit to {out}; verdict={verdict['verdict']}; n={len(rows)}")
    if args.card_output:
        Path(args.card_output).write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        Path(args.output_md).write_text(markdown(report, card) + "\n")
        print(f"wrote markdown to {args.output_md}")


if __name__ == "__main__":
    main()
