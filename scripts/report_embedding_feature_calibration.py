#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build calibration and failure-mode reports for embedding-risk scores."
    )
    parser.add_argument(
        "--scores",
        action="append",
        required=True,
        help="Scored JSONL from scripts/score_embedding_centroids.py. Repeat for comparisons.",
    )
    parser.add_argument(
        "--names",
        default=None,
        help="Optional comma-separated names matching --scores order.",
    )
    parser.add_argument("--output-json", required=True, help="Output JSON report.")
    parser.add_argument("--output-md", default=None, help="Optional Markdown report.")
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.05,
        help="Target false-positive rate for a conservative threshold recommendation.",
    )
    parser.add_argument(
        "--ambiguous-margin",
        type=float,
        default=0.08,
        help="Absolute margin band treated as hard-neighbor ambiguity.",
    )
    parser.add_argument(
        "--hard-neighbor-threshold",
        type=float,
        default=0.0,
        help="Safe rows at or above this margin are reported as hard benign collisions.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def label_to_binary(label: Any) -> int | None:
    normalized = str(label).lower()
    if normalized in {"unsafe", "harmful", "1", "true"}:
        return 1
    if normalized in {"safe", "benign", "0", "false"}:
        return 0
    return None


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * pct)))
    return sorted_values[idx]


def distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": mean(values) if values else 0.0,
        "p05": percentile(values, 0.05),
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
    }


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


def confusion_at_threshold(
    rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, float | int]:
    tp = fp = tn = fn = 0
    for row in rows:
        label = label_to_binary(row.get("label"))
        if label is None:
            continue
        predicted = float(row["risk_margin"]) >= threshold
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
    tnr = tn / (tn + fp) if tn + fp else 0.0
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
        "balanced_accuracy": (recall + tnr) / 2,
        "f1": f1,
    }


def threshold_grid(rows: list[dict[str, Any]]) -> list[float]:
    margins = [
        float(row["risk_margin"])
        for row in rows
        if label_to_binary(row.get("label")) is not None
    ]
    if not margins:
        return [0.0]
    low = min(margins)
    high = max(margins)
    if math.isclose(low, high):
        return [low]
    step = (high - low) / 200
    return [low + idx * step for idx in range(201)]


def threshold_recommendations(
    rows: list[dict[str, Any]],
    *,
    target_fpr: float,
) -> dict[str, Any]:
    sweep = [confusion_at_threshold(rows, threshold) for threshold in threshold_grid(rows)]
    best_f1 = max(sweep, key=lambda item: (item["f1"], item["balanced_accuracy"]))
    conservative_candidates = [
        item for item in sweep if item["false_positive_rate"] <= target_fpr
    ]
    conservative = (
        max(conservative_candidates, key=lambda item: (item["recall"], item["precision"]))
        if conservative_candidates
        else min(sweep, key=lambda item: item["false_positive_rate"])
    )
    zero = confusion_at_threshold(rows, 0.0)
    return {
        "zero_margin": zero,
        "best_f1": best_f1,
        "target_fpr": target_fpr,
        "conservative": conservative,
    }


def grouped_distributions(
    rows: list[dict[str, Any]],
    group_key: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key, "unknown"))].append(row)

    output: dict[str, dict[str, Any]] = {}
    for group, group_rows in sorted(grouped.items()):
        by_label: dict[str, list[float]] = defaultdict(list)
        for row in group_rows:
            by_label[str(row.get("label", "unknown"))].append(float(row["risk_margin"]))
        labels = [label_to_binary(row.get("label")) for row in group_rows]
        scores = [float(row["risk_margin"]) for row in group_rows]
        valid_pairs = [
            (label, score)
            for label, score in zip(labels, scores, strict=True)
            if label is not None
        ]
        output[group] = {
            "num_rows": len(group_rows),
            "labels": dict(Counter(str(row.get("label", "unknown")) for row in group_rows)),
            "risk_margin_by_label": {
                label: distribution(values) for label, values in sorted(by_label.items())
            },
            "auc": (
                auc([label for label, _ in valid_pairs], [score for _, score in valid_pairs])
                if valid_pairs
                else None
            ),
        }
    return output


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "label": row.get("label"),
        "source": row.get("source"),
        "domain": row.get("domain"),
        "subcluster_role": row.get("subcluster_role"),
        "subcluster_id": row.get("subcluster_id"),
        "risk_margin": row.get("risk_margin"),
        "harm_similarity": row.get("harm_similarity"),
        "benign_similarity": row.get("benign_similarity"),
        "top_risk_cluster": row.get("top_risk_cluster"),
        "top_risk_domain": row.get("top_risk_domain"),
        "top_benign_cluster": row.get("top_benign_cluster"),
        "same_domain_benign_cluster": row.get("same_domain_benign_cluster"),
        "any_domain_benign_cluster": row.get("any_domain_benign_cluster"),
        "span_text": row.get("span_text"),
    }


def hard_neighbor_report(
    rows: list[dict[str, Any]],
    *,
    hard_neighbor_threshold: float,
    ambiguous_margin: float,
    top_k: int,
) -> dict[str, Any]:
    safe_high = [
        row
        for row in rows
        if label_to_binary(row.get("label")) == 0
        and float(row["risk_margin"]) >= hard_neighbor_threshold
    ]
    unsafe_low = [
        row
        for row in rows
        if label_to_binary(row.get("label")) == 1 and float(row["risk_margin"]) <= 0.0
    ]
    ambiguous = [
        row
        for row in rows
        if abs(float(row["risk_margin"])) <= ambiguous_margin
    ]
    safe_high.sort(key=lambda row: float(row["risk_margin"]), reverse=True)
    unsafe_low.sort(key=lambda row: float(row["risk_margin"]))

    collisions = Counter(
        (
            str(row.get("domain")),
            str(row.get("subcluster_id")),
            str(row.get("top_risk_cluster")),
            str(row.get("top_benign_cluster")),
        )
        for row in safe_high
    )
    return {
        "hard_neighbor_threshold": hard_neighbor_threshold,
        "ambiguous_margin": ambiguous_margin,
        "num_safe_high_margin": len(safe_high),
        "num_unsafe_nonpositive_margin": len(unsafe_low),
        "num_ambiguous_band_rows": len(ambiguous),
        "top_safe_high_margin": [compact_row(row) for row in safe_high[:top_k]],
        "top_unsafe_nonpositive_margin": [compact_row(row) for row in unsafe_low[:top_k]],
        "top_safe_collision_patterns": [
            {
                "domain": key[0],
                "subcluster_id": key[1],
                "top_risk_cluster": key[2],
                "top_benign_cluster": key[3],
                "count": count,
            }
            for key, count in collisions.most_common(top_k)
        ],
    }


def recommendation(
    rows: list[dict[str, Any]],
    threshold_report: dict[str, Any],
    auc_value: float | None,
) -> dict[str, Any]:
    by_label: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_label[str(row.get("label", "unknown"))].append(float(row["risk_margin"]))
    safe_p90 = percentile(by_label.get("safe", []), 0.90)
    unsafe_p10 = percentile(by_label.get("unsafe", []), 0.10)
    overlap = safe_p90 >= unsafe_p10
    zero = threshold_report["zero_margin"]
    conservative = threshold_report["conservative"]

    if auc_value is not None and auc_value >= 0.80 and overlap:
        role = "supporting_semantic_prior"
    elif auc_value is not None and auc_value >= 0.90 and not overlap:
        role = "candidate_decision_feature_after_heldout_validation"
    else:
        role = "weak_routing_signal"

    return {
        "recommended_role": role,
        "reason": (
            "Input embeddings separate broad semantic neighborhoods but overlap hard benign "
            "near-neighbors; use the signal as a low-to-medium weight prior, not a "
            "standalone block."
        ),
        "safe_p90": safe_p90,
        "unsafe_p10": unsafe_p10,
        "overlap_between_safe_p90_and_unsafe_p10": overlap,
        "zero_margin_false_positive_rate": zero["false_positive_rate"],
        "zero_margin_recall": zero["recall"],
        "conservative_threshold": conservative["threshold"],
        "conservative_false_positive_rate": conservative["false_positive_rate"],
        "conservative_recall": conservative["recall"],
    }


def build_report_for_rows(
    rows: list[dict[str, Any]],
    *,
    name: str,
    target_fpr: float,
    ambiguous_margin: float,
    hard_neighbor_threshold: float,
    top_k: int,
) -> dict[str, Any]:
    valid = [row for row in rows if label_to_binary(row.get("label")) is not None]
    labels = [label_to_binary(row["label"]) for row in valid]
    scores = [float(row["risk_margin"]) for row in valid]
    auc_value = auc([label for label in labels if label is not None], scores)
    by_label: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_label[str(row.get("label", "unknown"))].append(float(row["risk_margin"]))
    thresholds = threshold_recommendations(valid, target_fpr=target_fpr)
    return {
        "name": name,
        "num_rows": len(rows),
        "labels": dict(Counter(str(row.get("label", "unknown")) for row in rows)),
        "similarity_modes": dict(Counter(str(row.get("similarity_mode")) for row in rows)),
        "benign_contrast_modes": dict(
            Counter(str(row.get("benign_contrast_mode")) for row in rows)
        ),
        "auc": auc_value,
        "risk_margin_by_label": {
            label: distribution(values) for label, values in sorted(by_label.items())
        },
        "thresholds": thresholds,
        "domain_calibration": grouped_distributions(rows, "domain"),
        "source_slices": grouped_distributions(rows, "source"),
        "hard_neighbors": hard_neighbor_report(
            rows,
            hard_neighbor_threshold=hard_neighbor_threshold,
            ambiguous_margin=ambiguous_margin,
            top_k=top_k,
        ),
        "recommendation": recommendation(valid, thresholds, auc_value),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Embedding Feature Calibration Report",
        "",
        "This report evaluates centroid margin scores as a supporting RAMP signal.",
        "",
    ]
    for run in report["runs"]:
        rec = run["recommendation"]
        lines.extend(
            [
                f"## {run['name']}",
                "",
                f"- Rows: {run['num_rows']}",
                f"- Labels: `{run['labels']}`",
                f"- AUC: `{run['auc']:.4f}`" if run["auc"] is not None else "- AUC: `n/a`",
                f"- Recommended role: `{rec['recommended_role']}`",
                f"- Safe p90 margin: `{rec['safe_p90']:.4f}`",
                f"- Unsafe p10 margin: `{rec['unsafe_p10']:.4f}`",
                "- Zero-margin FPR / recall: "
                f"`{rec['zero_margin_false_positive_rate']:.4f}` / "
                f"`{rec['zero_margin_recall']:.4f}`",
                f"- Conservative threshold: `{rec['conservative_threshold']:.4f}`",
                "- Conservative FPR / recall: "
                f"`{rec['conservative_false_positive_rate']:.4f}` / "
                f"`{rec['conservative_recall']:.4f}`",
                "",
                "| Label | Count | Mean | P10 | P50 | P90 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, dist in run["risk_margin_by_label"].items():
            lines.append(
                f"| {label} | {dist['count']} | {dist['mean']:.4f} | "
                f"{dist['p10']:.4f} | {dist['p50']:.4f} | {dist['p90']:.4f} |"
            )
        lines.extend(["", "### Highest-Margin Safe Near-Neighbors", ""])
        for row in run["hard_neighbors"]["top_safe_high_margin"][:5]:
            lines.append(
                f"- `{row['domain']}/{row['subcluster_id']}` margin `{row['risk_margin']:.4f}` "
                f"risk `{row['top_risk_cluster']}` vs benign `{row['top_benign_cluster']}`"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    score_paths = [Path(path) for path in args.scores]
    names = (
        [name.strip() for name in args.names.split(",")]
        if args.names
        else [path.stem for path in score_paths]
    )
    if len(names) != len(score_paths):
        raise ValueError("--names must contain the same number of entries as --scores")

    runs = [
        build_report_for_rows(
            load_rows(path),
            name=name,
            target_fpr=args.target_fpr,
            ambiguous_margin=args.ambiguous_margin,
            hard_neighbor_threshold=args.hard_neighbor_threshold,
            top_k=args.top_k,
        )
        for path, name in zip(score_paths, names, strict=True)
    ]
    report = {
        "target_fpr": args.target_fpr,
        "ambiguous_margin": args.ambiguous_margin,
        "hard_neighbor_threshold": args.hard_neighbor_threshold,
        "runs": runs,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote embedding calibration report to {output_json}")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown_report(report), encoding="utf-8")
        print(f"wrote embedding calibration markdown to {output_md}")

    for run in runs:
        rec = run["recommendation"]
        print(
            f"{run['name']}: role={rec['recommended_role']} "
            f"auc={run['auc']:.4f} conservative_threshold={rec['conservative_threshold']:.4f}"
        )


if __name__ == "__main__":
    main()
