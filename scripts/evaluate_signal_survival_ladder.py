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
import hashlib
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
    weighted_score,
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

CANDIDATE_CONDITIONS = tuple(name for name in COMBINATIONS if name != BASELINE_CONDITION)


# --- Paired bootstrap for single-evaluation rungs (blind, shifted) -----------------
# The naive/split/crossfit rungs report stdev across many splits. The blind rung and
# each per-source shifted holdout are single evaluations, so per-row paired bootstrap
# CIs (pre-registered: 10,000 resamples, 95% CI excluding zero) are how significance is
# assessed there. AUC uses the same average-rank formula as
# evaluate_reviewed_cumulative_signals.auc; F1 uses score >= threshold, matching metrics().


def _seed_int(seed: str) -> int:
    return int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:8], "big")


def _fast_auc(np: Any, labels_a: Any, scores_a: Any) -> float | None:
    positives = int(labels_a.sum())
    total = int(labels_a.size)
    negatives = total - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores_a, kind="mergesort")
    s_sorted = scores_a[order]
    ranks_sorted = np.empty(total, dtype=np.float64)
    index = 0
    while index < total:
        end = index
        value = s_sorted[index]
        while end + 1 < total and s_sorted[end + 1] == value:
            end += 1
        ranks_sorted[index : end + 1] = (index + end) / 2.0 + 1.0
        index = end + 1
    ranks = np.empty(total, dtype=np.float64)
    ranks[order] = ranks_sorted
    positive_rank_sum = float(ranks[labels_a == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _f1_at(np: Any, labels_a: Any, scores_a: Any, threshold: Any) -> float:
    predicted = scores_a >= threshold
    tp = int(np.sum(predicted & (labels_a == 1)))
    fp = int(np.sum(predicted & (labels_a == 0)))
    fn = int(np.sum(~predicted & (labels_a == 1)))
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else 0.0


def _delta_summary(np: Any, deltas: list[float], *, point: float | None) -> dict[str, Any]:
    if not deltas:
        return {"delta": point, "ci95": [None, None], "significant": False, "resamples": 0}
    array = np.asarray(deltas, dtype=np.float64)
    low = float(np.percentile(array, 2.5))
    high = float(np.percentile(array, 97.5))
    return {
        "delta": point,
        "ci95": [low, high],
        "significant": bool(low > 0 or high < 0),
        "resamples": len(deltas),
    }


def paired_bootstrap(
    np: Any,
    labels: list[int],
    base_scores: list[float],
    base_threshold: Any,
    candidates: list[tuple[str, list[float], Any]],
    *,
    num_resamples: int,
    seed: str,
) -> dict[str, dict[str, Any]]:
    """Per-row paired bootstrap of AUROC and F1 deltas vs the prompt-only baseline.

    Thresholds may be scalars (blind rung) or per-row arrays (pooled shifted rung, where
    each row was scored by its own leave-one-source-out calibration).
    """
    labels_a = np.asarray(labels, dtype=np.int64)
    base_s = np.asarray(base_scores, dtype=np.float64)
    base_t = np.asarray(base_threshold, dtype=np.float64)
    cand = [
        (name, np.asarray(scores, dtype=np.float64), np.asarray(threshold, dtype=np.float64))
        for name, scores, threshold in candidates
    ]
    total = int(labels_a.size)
    base_auc_point = _fast_auc(np, labels_a, base_s)
    base_f1_point = _f1_at(np, labels_a, base_s, base_t)

    rng = np.random.default_rng(_seed_int(seed))
    draws = rng.integers(0, total, size=(num_resamples, total))
    accumulator = {name: {"auc": [], "f1": []} for name, _, _ in cand}
    used = 0
    for resample in range(num_resamples):
        idx = draws[resample]
        labels_b = labels_a[idx]
        positives = int(labels_b.sum())
        if positives == 0 or positives == total:
            continue
        used += 1
        base_sb = base_s[idx]
        base_tb = base_t[idx] if base_t.ndim else base_t
        base_auc = _fast_auc(np, labels_b, base_sb)
        base_f1 = _f1_at(np, labels_b, base_sb, base_tb)
        for name, scores, threshold in cand:
            cand_sb = scores[idx]
            cand_tb = threshold[idx] if threshold.ndim else threshold
            cand_auc = _fast_auc(np, labels_b, cand_sb)
            if base_auc is not None and cand_auc is not None:
                accumulator[name]["auc"].append(cand_auc - base_auc)
            accumulator[name]["f1"].append(_f1_at(np, labels_b, cand_sb, cand_tb) - base_f1)

    output: dict[str, dict[str, Any]] = {}
    for name, scores, threshold in cand:
        cand_auc_point = _fast_auc(np, labels_a, scores)
        auc_point = (
            None
            if base_auc_point is None or cand_auc_point is None
            else cand_auc_point - base_auc_point
        )
        f1_point = _f1_at(np, labels_a, scores, threshold) - base_f1_point
        output[name] = {
            "auc": _delta_summary(np, accumulator[name]["auc"], point=auc_point),
            "f1": _delta_summary(np, accumulator[name]["f1"], point=f1_point),
            "resamples_used": used,
            "num_rows": total,
        }
    return output


def _row_weighted_delta(
    np: Any,
    blocks: list[dict[str, Any]],
    index_per_block: list[Any],
    condition: str,
) -> tuple[float | None, float]:
    """Row-weighted AUROC and F1 deltas (candidate - baseline) across source blocks.

    Mirrors the row_weighted_metrics verdict statistic exactly: each block contributes its
    own metric weighted by its holdout-row count. AUC delta is None if any block resample
    is single-class (its AUC undefined).
    """
    weight_total = 0
    auc_base = auc_cand = 0.0
    f1_base = f1_cand = 0.0
    auc_ok = True
    for block, idx in zip(blocks, index_per_block, strict=True):
        labels_b = block["labels"][idx]
        weight = int(idx.size)
        weight_total += weight
        base_scores, base_threshold = block["scores"][BASELINE_CONDITION]
        cand_scores, cand_threshold = block["scores"][condition]
        base_scores = base_scores[idx]
        cand_scores = cand_scores[idx]
        base_auc = _fast_auc(np, labels_b, base_scores)
        cand_auc = _fast_auc(np, labels_b, cand_scores)
        if base_auc is None or cand_auc is None:
            auc_ok = False
        else:
            auc_base += base_auc * weight
            auc_cand += cand_auc * weight
        f1_base += _f1_at(np, labels_b, base_scores, base_threshold) * weight
        f1_cand += _f1_at(np, labels_b, cand_scores, cand_threshold) * weight
    auc_delta = (auc_cand - auc_base) / weight_total if auc_ok and weight_total else None
    f1_delta = (f1_cand - f1_base) / weight_total if weight_total else 0.0
    return auc_delta, f1_delta


def stratified_paired_bootstrap(
    np: Any,
    source_data: list[tuple[list[int], dict[str, tuple[list[float], float]]]],
    *,
    num_resamples: int,
    seed: str,
) -> dict[str, dict[str, Any]]:
    """Stratified (within-source) paired bootstrap of the row-weighted verdict statistic.

    Used for the shifted rung so the CI annotates the SAME row-weighted AUROC/F1 delta the
    verdict reports, rather than a pooled cross-source estimand.
    """
    blocks = [
        {
            "labels": np.asarray(labels, dtype=np.int64),
            "scores": {
                name: (np.asarray(scores, dtype=np.float64), float(threshold))
                for name, (scores, threshold) in data.items()
            },
        }
        for labels, data in source_data
    ]
    full_index = [np.arange(block["labels"].size) for block in blocks]
    total_rows = sum(int(block["labels"].size) for block in blocks)

    rng = np.random.default_rng(_seed_int(seed))
    resampled_index = [
        rng.integers(0, block["labels"].size, size=(num_resamples, block["labels"].size))
        for block in blocks
    ]

    output: dict[str, dict[str, Any]] = {}
    for condition in CANDIDATE_CONDITIONS:
        auc_point, f1_point = _row_weighted_delta(np, blocks, full_index, condition)
        auc_deltas: list[float] = []
        f1_deltas: list[float] = []
        for resample in range(num_resamples):
            index_per_block = [resampled_index[b][resample] for b in range(len(blocks))]
            auc_delta, f1_delta = _row_weighted_delta(np, blocks, index_per_block, condition)
            if auc_delta is not None:
                auc_deltas.append(auc_delta)
            f1_deltas.append(f1_delta)
        output[condition] = {
            "auc": _delta_summary(np, auc_deltas, point=auc_point),
            "f1": _delta_summary(np, f1_deltas, point=f1_point),
            "resamples_used": len(f1_deltas),
            "num_rows": total_rows,
        }
    return output


def collect_eval_data(
    rows: list[dict[str, Any]],
    calibrations: dict[str, dict[str, Any]],
) -> tuple[list[int], dict[str, tuple[list[float], float]]]:
    labels = [int(binary_label(str(row["reviewed_label"]))) for row in rows]
    data = {
        name: (
            [weighted_score(row, calibration["weights"]) for row in rows],
            float(calibration["threshold"]),
        )
        for name, calibration in calibrations.items()
    }
    return labels, data


def bootstrap_for_holdout(
    np: Any,
    rows: list[dict[str, Any]],
    calibrations: dict[str, dict[str, Any]],
    *,
    num_resamples: int,
    seed: str,
) -> dict[str, dict[str, Any]]:
    labels, data = collect_eval_data(rows, calibrations)
    base_scores, base_threshold = data[BASELINE_CONDITION]
    candidates = [
        (name, data[name][0], data[name][1]) for name in CANDIDATE_CONDITIONS if name in data
    ]
    return paired_bootstrap(
        np,
        labels,
        base_scores,
        base_threshold,
        candidates,
        num_resamples=num_resamples,
        seed=seed,
    )


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
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10000,
        help="Paired-bootstrap resamples for the single-evaluation rungs (blind, shifted).",
    )
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
    bootstrap = bootstrap_for_holdout(
        np,
        holdout_rows,
        calibrations,
        num_resamples=args.bootstrap_resamples,
        seed=f"{args.seed_prefix}:blind:bootstrap",
    )
    return {
        "status": "completed",
        "protocol": (
            "Calibrated on the full adaptive set with out-of-fold probe scores; probe "
            "trained on the adaptive set only; evaluated once on blind rows. Significance "
            "from a per-row paired bootstrap of AUROC/F1 deltas vs prompt-only."
        ),
        "activation_probe": probe_report,
        "blind_rows": len(holdout_rows),
        "num_evaluations": 1,
        "aggregate_holdout_metrics": aggregate(splits, COMBINATIONS),
        "bootstrap": bootstrap,
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
    source_data: list[tuple[list[int], dict[str, tuple[list[float], float]]]] = []
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
        source_labels, source_scores = collect_eval_data(holdout_rows, calibrations)
        source_data.append((source_labels, source_scores))
        per_source[source] = {
            "holdout_rows": len(holdout_rows),
            "calibration_rows": len(calibration_rows),
            "activation_probe": probe_report,
            "metrics": {
                name: {
                    "auc": holdout[name]["auc"],
                    "f1": holdout[name]["metrics"]["f1"],
                    "recall": holdout[name]["metrics"]["recall"],
                    "false_positive_rate": holdout[name]["metrics"]["false_positive_rate"],
                }
                for name in COMBINATIONS
            },
            "bootstrap": paired_bootstrap(
                np,
                source_labels,
                source_scores[BASELINE_CONDITION][0],
                source_scores[BASELINE_CONDITION][1],
                [
                    (name, source_scores[name][0], source_scores[name][1])
                    for name in CANDIDATE_CONDITIONS
                ],
                num_resamples=args.bootstrap_resamples,
                seed=f"{args.seed_prefix}:shifted:{source}:bootstrap",
            ),
        }
    stratified_bootstrap = stratified_paired_bootstrap(
        np,
        source_data,
        num_resamples=args.bootstrap_resamples,
        seed=f"{args.seed_prefix}:shifted:stratified:bootstrap",
    )
    return {
        "status": "completed",
        "protocol": (
            "Each eligible source held out in turn; probe trained and weights tuned "
            "on the remaining sources with out-of-fold calibration scores. Survival "
            "verdicts use holdout-row-weighted means across sources (pre-registered); "
            "unweighted split means are retained for reference. Significance from a "
            "stratified (within-source) paired bootstrap of the same row-weighted "
            "AUROC/F1 delta the verdict uses; per-source bootstraps are also reported."
        ),
        "holdout_sources": sources,
        "per_source": per_source,
        "num_evaluations": len(sources),
        "aggregate_holdout_metrics": aggregate(splits, COMBINATIONS),
        "row_weighted_holdout_metrics": row_weighted_metrics(per_source),
        "bootstrap": stratified_bootstrap,
    }


def row_weighted_metrics(per_source: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    output: dict[str, dict[str, float | None]] = {}
    metric_names = ("auc", "f1", "recall", "false_positive_rate")
    for name in COMBINATIONS:
        output[name] = {}
        for metric in metric_names:
            weighted = [
                (detail["metrics"][name][metric], detail["holdout_rows"])
                for detail in per_source.values()
                if detail["metrics"][name].get(metric) is not None
            ]
            total = sum(weight for _, weight in weighted)
            output[name][metric] = (
                sum(value * weight for value, weight in weighted) / total if total else None
            )
    return output


def rung_mean(rung: dict[str, Any], condition: str, metric: str) -> float | None:
    if rung.get("status") != "completed":
        return None
    row_weighted = rung.get("row_weighted_holdout_metrics")
    if row_weighted is not None:
        return row_weighted[condition].get(metric)
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
    cell = {"verdict": verdict, "auc_delta": auc_delta, "f1_delta": f1_delta}
    # The verdict rule is the pre-registered mean-delta sign and is unchanged. For the
    # single-evaluation rungs we additionally surface the paired-bootstrap CIs so a
    # survived/failed verdict can be read against its statistical significance.
    bootstrap = rung.get("bootstrap", {}).get(condition)
    if bootstrap is not None:
        cell["auc_ci95"] = bootstrap["auc"]["ci95"]
        cell["auc_significant"] = bootstrap["auc"]["significant"]
        cell["f1_ci95"] = bootstrap["f1"]["ci95"]
        cell["f1_significant"] = bootstrap["f1"]["significant"]
    return cell


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
                text += f" (dAUC {cell['auc_delta']:+.4f}"
                if "auc_significant" in cell:
                    text += "*" if cell["auc_significant"] else " ns"
                text += ")"
            row.append(text)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(
        "`*` = paired-bootstrap 95% CI on the AUROC delta excludes zero (blind/shifted "
        "rungs only); `ns` = CI includes zero."
    )
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
