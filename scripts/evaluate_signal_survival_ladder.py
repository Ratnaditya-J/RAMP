#!/usr/bin/env python
"""Signal survival ladder: the same signals under increasingly honest protocols.

Rungs (see docs/fragility-study-design.md):
  naive     weights/threshold tuned on all rows; probe trained and scored in-sample;
            evaluated on the same rows.
  split     weights tuned on calibration half, evaluated on holdout half; probe trained
            once on ALL rows (in-sample for every row). Reproduces the v0.1 leakage.
  crossfit  leakage-free v0.2 protocol: per-split probe, out-of-fold calibration
            scores, out-of-split holdout scores.
  blind     calibrate on the full adaptive set (out-of-fold probe scores), evaluate on
            a blind reviewed set the system has never seen. Requires --blind-review-csv.
  shifted   hold out one source at a time: calibrate and train the probe on the other
            sources, evaluate on the held-out source.

Survival rule v0 (pre-registered): a signal survives a rung when its calibrated
combination beats prompt-only calibrated on BOTH mean AUROC and mean F1; exactly one
improvement is "mixed"; neither is "fails".
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluate_calibrated_signal_combinations import (
    COMBINATIONS,
    aggregate,
    calibrate_combination,
    evaluate_combination,
    stratified_split,
)
from evaluate_leakage_free_activation_fusion import (
    attach_split_activation_scores,
    binary_feature_rows,
    load_activation_vectors,
)
from evaluate_reviewed_cumulative_signals import (
    binary_label,
    join_rows,
    load_feature_table,
    load_reviewed_rows,
)
from evaluate_severity_floors import DEFAULT_HIGH_SEVERITY_CATEGORIES, normalize_category
from train_activation_probes import require_numpy, train_probe

RUNG_ORDER = ("naive", "split", "crossfit", "blind", "shifted")

SIGNAL_CONDITIONS = {
    "prompt": "prompt_only_calibrated",
    "embedding": "prompt_embedding_calibrated",
    "activation": "prompt_activation_calibrated",
    "full_fusion": "prompt_embedding_activation_calibrated",
}

BASELINE_CONDITION = "prompt_only_calibrated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate prompt/embedding/activation signal combinations under the "
            "naive/split/crossfit/blind/shifted evaluation ladder and emit a "
            "per-signal survival table."
        )
    )
    parser.add_argument("--review-csv", required=True, help="Adaptive reviewed CSV.")
    parser.add_argument("--feature-table", required=True)
    parser.add_argument("--activation", required=True, help="Activation JSONL.")
    parser.add_argument("--layer", default="19")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument(
        "--blind-review-csv",
        default=None,
        help="Blind reviewed CSV for the blind rung. Omit to mark the rung pending.",
    )
    parser.add_argument(
        "--rungs",
        default=",".join(RUNG_ORDER),
        help="Comma-separated subset of rungs to run.",
    )
    parser.add_argument("--num-splits", type=int, default=30)
    parser.add_argument("--seed-prefix", default="ramp_signal_survival_v0.1")
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--calibration-folds", type=int, default=5)
    parser.add_argument(
        "--min-shifted-class-rows",
        type=int,
        default=5,
        help="Minimum rows per class a source needs to be a shifted holdout.",
    )
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--min-prompt-weight", type=float, default=0.40)
    parser.add_argument("--max-embedding-weight", type=float, default=0.20)
    parser.add_argument("--require-prompt-gte-activation", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.10)
    parser.add_argument("--l2", type=float, default=0.001)
    return parser.parse_args()


def calibrate_all(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    return {
        name: calibrate_combination(
            rows,
            keys,
            weight_step=args.weight_step,
            threshold_step=args.threshold_step,
            min_prompt_weight=args.min_prompt_weight,
            max_embedding_weight=args.max_embedding_weight,
            max_output_weight=0.0,
            require_prompt_gte_activation=args.require_prompt_gte_activation,
        )
        for name, keys in COMBINATIONS.items()
    }


def evaluate_all(
    rows: list[dict[str, Any]],
    calibrations: dict[str, dict[str, Any]],
    high_severity_categories: set[str],
) -> dict[str, dict[str, Any]]:
    return {
        name: evaluate_combination(
            rows,
            name,
            calibration,
            high_severity_categories=high_severity_categories,
            top_k_errors=0,
        )
        for name, calibration in calibrations.items()
    }


def attach_in_sample_activation_scores(
    np: Any,
    rows: list[dict[str, Any]],
    activation_vectors: dict[str, list[float]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    labels = [int(binary_label(str(row["reviewed_label"]))) for row in rows]
    vectors = [activation_vectors[str(row["source_id"])] for row in rows]
    indices = list(range(len(rows)))
    probe = train_probe(
        np,
        vectors,
        labels,
        train_indices=indices,
        test_indices=indices,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    probabilities = [float(value) for value in probe["test_probabilities"].tolist()]
    for row, probability in zip(rows, probabilities, strict=True):
        row["activation_probability"] = probability
    return {"mode": "in_sample", "rows": len(rows), "metrics": probe["test_metrics"]}


def run_naive_rung(
    np: Any,
    eval_rows: list[dict[str, Any]],
    activation_vectors: dict[str, list[float]],
    high_severity_categories: set[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows = [dict(row) for row in eval_rows]
    probe_report = attach_in_sample_activation_scores(np, rows, activation_vectors, args)
    calibrations = calibrate_all(rows, args)
    holdout = evaluate_all(rows, calibrations, high_severity_categories)
    splits = [{"split_index": 0, "calibrations": calibrations, "holdout": holdout}]
    return {
        "status": "completed",
        "protocol": (
            "Probe trained and scored in-sample on all rows; weights and threshold "
            "tuned on all rows; evaluated on the same rows."
        ),
        "activation_probe": probe_report,
        "num_evaluations": 1,
        "aggregate_holdout_metrics": aggregate(splits, COMBINATIONS),
    }


def run_split_rung(
    np: Any,
    eval_rows: list[dict[str, Any]],
    activation_vectors: dict[str, list[float]],
    high_severity_categories: set[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows = [dict(row) for row in eval_rows]
    probe_report = attach_in_sample_activation_scores(np, rows, activation_vectors, args)
    splits = []
    for split_index in range(args.num_splits):
        calibration_rows, holdout_rows = stratified_split(
            rows,
            calibration_fraction=args.calibration_fraction,
            seed=f"{args.seed_prefix}:split:{split_index:03d}",
        )
        calibrations = calibrate_all(calibration_rows, args)
        holdout = evaluate_all(holdout_rows, calibrations, high_severity_categories)
        splits.append(
            {"split_index": split_index, "calibrations": calibrations, "holdout": holdout}
        )
    return {
        "status": "completed",
        "protocol": (
            "Probe trained once on ALL rows (in-sample for every row); weights and "
            "threshold tuned on calibration halves; evaluated on holdout halves. "
            "Reproduces the v0.1 leakage pattern."
        ),
        "activation_probe": probe_report,
        "num_evaluations": args.num_splits,
        "aggregate_holdout_metrics": aggregate(splits, COMBINATIONS),
    }


def run_crossfit_rung(
    np: Any,
    eval_rows: list[dict[str, Any]],
    activation_vectors: dict[str, list[float]],
    high_severity_categories: set[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    splits = []
    for split_index in range(args.num_splits):
        calibration_rows, holdout_rows = stratified_split(
            eval_rows,
            calibration_fraction=args.calibration_fraction,
            seed=f"{args.seed_prefix}:crossfit:{split_index:03d}",
        )
        calibration_rows = [dict(row) for row in calibration_rows]
        holdout_rows = [dict(row) for row in holdout_rows]
        probe_report = attach_split_activation_scores(
            np,
            calibration_rows,
            holdout_rows,
            activation_vectors,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            calibration_folds=args.calibration_folds,
            seed=f"{args.seed_prefix}:crossfit:{split_index:03d}",
        )
        calibrations = calibrate_all(calibration_rows, args)
        holdout = evaluate_all(holdout_rows, calibrations, high_severity_categories)
        splits.append(
            {
                "split_index": split_index,
                "activation_probe": probe_report,
                "calibrations": calibrations,
                "holdout": holdout,
            }
        )
    return {
        "status": "completed",
        "protocol": (
            "Per-split probe trained on calibration rows only; calibration rows use "
            "out-of-fold probe scores; holdout rows use out-of-split scores."
        ),
        "num_evaluations": args.num_splits,
        "aggregate_holdout_metrics": aggregate(splits, COMBINATIONS),
    }


def run_blind_rung(
    np: Any,
    eval_rows: list[dict[str, Any]],
    blind_rows: list[dict[str, Any]] | None,
    activation_vectors: dict[str, list[float]],
    high_severity_categories: set[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if blind_rows is None:
        return {
            "status": "pending",
            "reason": "no --blind-review-csv provided; blind reviewed set not built yet",
        }
    adaptive_ids = {str(row["source_id"]) for row in eval_rows}
    blind_rows = [row for row in blind_rows if str(row["source_id"]) not in adaptive_ids]
    if not blind_rows:
        return {
            "status": "skipped",
            "reason": "all blind rows overlap the adaptive set after dedup",
        }
    calibration_rows = [dict(row) for row in eval_rows]
    holdout_rows = [dict(row) for row in blind_rows]
    probe_report = attach_split_activation_scores(
        np,
        calibration_rows,
        holdout_rows,
        activation_vectors,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        calibration_folds=args.calibration_folds,
        seed=f"{args.seed_prefix}:blind",
    )
    calibrations = calibrate_all(calibration_rows, args)
    holdout = evaluate_all(holdout_rows, calibrations, high_severity_categories)
    splits = [{"split_index": 0, "calibrations": calibrations, "holdout": holdout}]
    return {
        "status": "completed",
        "protocol": (
            "Calibrated on the full adaptive set with out-of-fold probe scores; probe "
            "trained on the adaptive set only; evaluated once on blind rows."
        ),
        "activation_probe": probe_report,
        "blind_rows": len(holdout_rows),
        "num_evaluations": 1,
        "aggregate_holdout_metrics": aggregate(splits, COMBINATIONS),
    }


def shifted_holdout_sources(
    rows: list[dict[str, Any]],
    *,
    min_class_rows: int,
) -> list[str]:
    by_source: dict[str, Counter] = {}
    for row in rows:
        source = str(row.get("source", "unknown"))
        label = binary_label(str(row.get("reviewed_label")))
        by_source.setdefault(source, Counter())[label] += 1
    eligible = []
    for source, counts in sorted(by_source.items()):
        if counts[0] >= min_class_rows and counts[1] >= min_class_rows:
            remainder = Counter()
            for other, other_counts in by_source.items():
                if other != source:
                    remainder.update(other_counts)
            if remainder[0] >= min_class_rows and remainder[1] >= min_class_rows:
                eligible.append(source)
    return eligible


def run_shifted_rung(
    np: Any,
    eval_rows: list[dict[str, Any]],
    activation_vectors: dict[str, list[float]],
    high_severity_categories: set[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    sources = shifted_holdout_sources(eval_rows, min_class_rows=args.min_shifted_class_rows)
    if not sources:
        return {
            "status": "skipped",
            "reason": (
                "no source has both classes with at least "
                f"{args.min_shifted_class_rows} rows each on both sides of the split"
            ),
        }
    splits = []
    per_source = {}
    for split_index, source in enumerate(sources):
        holdout_rows = [
            dict(row) for row in eval_rows if str(row.get("source", "unknown")) == source
        ]
        calibration_rows = [
            dict(row) for row in eval_rows if str(row.get("source", "unknown")) != source
        ]
        probe_report = attach_split_activation_scores(
            np,
            calibration_rows,
            holdout_rows,
            activation_vectors,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            calibration_folds=args.calibration_folds,
            seed=f"{args.seed_prefix}:shifted:{source}",
        )
        calibrations = calibrate_all(calibration_rows, args)
        holdout = evaluate_all(holdout_rows, calibrations, high_severity_categories)
        splits.append(
            {"split_index": split_index, "calibrations": calibrations, "holdout": holdout}
        )
        per_source[source] = {
            "holdout_rows": len(holdout_rows),
            "calibration_rows": len(calibration_rows),
            "activation_probe": probe_report,
            "metrics": {
                name: {
                    "auc": holdout[name]["auc"],
                    "f1": holdout[name]["metrics"]["f1"],
                    "false_positive_rate": holdout[name]["metrics"]["false_positive_rate"],
                }
                for name in COMBINATIONS
            },
        }
    return {
        "status": "completed",
        "protocol": (
            "Each eligible source held out in turn; probe trained and weights tuned "
            "on the remaining sources with out-of-fold calibration scores."
        ),
        "holdout_sources": sources,
        "per_source": per_source,
        "num_evaluations": len(sources),
        "aggregate_holdout_metrics": aggregate(splits, COMBINATIONS),
    }


def rung_mean(rung: dict[str, Any], condition: str, metric: str) -> float | None:
    if rung.get("status") != "completed":
        return None
    summary = rung["aggregate_holdout_metrics"][condition].get(metric)
    if not summary:
        return None
    return summary.get("mean")


def survival_verdict(rung: dict[str, Any], condition: str) -> dict[str, Any]:
    if rung.get("status") != "completed":
        return {"verdict": rung.get("status", "pending")}
    auc_delta = None
    f1_delta = None
    auc_mean = rung_mean(rung, condition, "auc")
    base_auc = rung_mean(rung, BASELINE_CONDITION, "auc")
    if auc_mean is not None and base_auc is not None:
        auc_delta = auc_mean - base_auc
    f1_mean = rung_mean(rung, condition, "f1")
    base_f1 = rung_mean(rung, BASELINE_CONDITION, "f1")
    if f1_mean is not None and base_f1 is not None:
        f1_delta = f1_mean - base_f1
    improvements = sum(1 for delta in (auc_delta, f1_delta) if delta is not None and delta > 0)
    if auc_delta is None or f1_delta is None:
        verdict = "incomplete"
    elif improvements == 2:
        verdict = "survives"
    elif improvements == 1:
        verdict = "mixed"
    else:
        verdict = "fails"
    return {"verdict": verdict, "auc_delta": auc_delta, "f1_delta": f1_delta}


def build_survival_table(rungs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for signal, condition in SIGNAL_CONDITIONS.items():
        table[signal] = {}
        for rung_name in RUNG_ORDER:
            rung = rungs.get(rung_name, {"status": "not_run"})
            if signal == "prompt":
                completed = rung.get("status") == "completed"
                table[signal][rung_name] = {
                    "verdict": "baseline" if completed else rung.get("status", "not_run")
                }
            else:
                table[signal][rung_name] = survival_verdict(rung, condition)
    return table


VERDICT_GLYPHS = {
    "survives": "yes",
    "mixed": "mixed",
    "fails": "no",
    "baseline": "baseline",
    "pending": "pending",
    "skipped": "skipped",
    "not_run": "not run",
    "incomplete": "incomplete",
}


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Signal Survival Ladder",
        "",
        f"Adaptive binary rows: {report['num_binary_eval_rows']}",
        "",
        "## Survival Table",
        "",
        "| Signal | " + " | ".join(RUNG_ORDER) + " |",
        "| --- | " + " | ".join("---" for _ in RUNG_ORDER) + " |",
    ]
    for signal, cells in report["survival_table"].items():
        row = [signal]
        for rung_name in RUNG_ORDER:
            cell = cells[rung_name]
            text = VERDICT_GLYPHS.get(cell["verdict"], cell["verdict"])
            if cell.get("auc_delta") is not None:
                text += f" (dAUC {cell['auc_delta']:+.4f})"
            row.append(text)
        lines.append("| " + " | ".join(row) + " |")
    for rung_name in RUNG_ORDER:
        rung = report["rungs"].get(rung_name)
        if not rung or rung.get("status") != "completed":
            continue
        lines.extend(
            [
                "",
                f"## Rung: {rung_name}",
                "",
                rung["protocol"],
                "",
                "| Condition | AUC | F1 | Recall | FPR |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for condition in COMBINATIONS:
            values = [
                rung_mean(rung, condition, metric)
                for metric in ("auc", "f1", "recall", "false_positive_rate")
            ]
            cells = ["n/a" if value is None else f"{value:.4f}" for value in values]
            lines.append(f"| `{condition}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    np = require_numpy()
    requested_rungs = [name.strip() for name in args.rungs.split(",") if name.strip()]
    unknown = set(requested_rungs) - set(RUNG_ORDER)
    if unknown:
        raise ValueError(f"unknown rungs: {sorted(unknown)}")

    reviewed = load_reviewed_rows(Path(args.review_csv))
    features = load_feature_table(Path(args.feature_table))
    joined = join_rows(reviewed, features)
    activation_vectors = load_activation_vectors(Path(args.activation))
    eval_rows = binary_feature_rows(
        joined, activation_vectors=activation_vectors, include_output=False
    )
    if not eval_rows:
        raise ValueError("no binary reviewed rows with features and activation vectors")

    blind_rows = None
    if args.blind_review_csv:
        blind_reviewed = load_reviewed_rows(Path(args.blind_review_csv))
        blind_joined = join_rows(blind_reviewed, features)
        blind_rows = binary_feature_rows(
            blind_joined, activation_vectors=activation_vectors, include_output=False
        )

    high_severity_categories = {
        normalize_category(category) for category in DEFAULT_HIGH_SEVERITY_CATEGORIES
    }

    rungs: dict[str, dict[str, Any]] = {}
    for rung_name in RUNG_ORDER:
        if rung_name not in requested_rungs:
            rungs[rung_name] = {"status": "not_run"}
            continue
        if rung_name == "naive":
            rungs[rung_name] = run_naive_rung(
                np, eval_rows, activation_vectors, high_severity_categories, args
            )
        elif rung_name == "split":
            rungs[rung_name] = run_split_rung(
                np, eval_rows, activation_vectors, high_severity_categories, args
            )
        elif rung_name == "crossfit":
            rungs[rung_name] = run_crossfit_rung(
                np, eval_rows, activation_vectors, high_severity_categories, args
            )
        elif rung_name == "blind":
            rungs[rung_name] = run_blind_rung(
                np,
                eval_rows,
                blind_rows,
                activation_vectors,
                high_severity_categories,
                args,
            )
        elif rung_name == "shifted":
            rungs[rung_name] = run_shifted_rung(
                np, eval_rows, activation_vectors, high_severity_categories, args
            )
        print(f"rung {rung_name}: {rungs[rung_name].get('status')}")

    report = {
        "artifact_id": "ramp_signal_survival_ladder_v0.1",
        "design_doc": "docs/fragility-study-design.md",
        "layer": args.layer,
        "review_csv": args.review_csv,
        "blind_review_csv": args.blind_review_csv,
        "feature_table": args.feature_table,
        "activation": args.activation,
        "num_binary_eval_rows": len(eval_rows),
        "num_splits": args.num_splits,
        "calibration_folds": args.calibration_folds,
        "survival_rule": (
            "survives = mean AUROC delta > 0 AND mean F1 delta > 0 vs prompt-only "
            "calibrated on the same rung; one improvement = mixed; none = fails"
        ),
        "rungs": rungs,
    }
    report["survival_table"] = build_survival_table(rungs)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote survival ladder report to {output_json}")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown_report(report) + "\n")
        print(f"wrote survival ladder markdown to {output_md}")

    for signal, cells in report["survival_table"].items():
        rendered = ", ".join(
            "{}={}".format(
                rung_name,
                VERDICT_GLYPHS.get(cells[rung_name]["verdict"], cells[rung_name]["verdict"]),
            )
            for rung_name in RUNG_ORDER
        )
        print(f"{signal}: {rendered}")


if __name__ == "__main__":
    main()
