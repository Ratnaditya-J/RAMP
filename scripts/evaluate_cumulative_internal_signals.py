#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from ramp.features.activation_probe import LinearActivationProbe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cumulative value of embedding and activation internal signals."
    )
    parser.add_argument("--embedding-scores", required=True, help="Embedding score JSONL.")
    parser.add_argument("--activation", required=True, help="Activation JSONL for selected layer.")
    parser.add_argument("--activation-probe", required=True, help="Activation probe artifact JSON.")
    parser.add_argument(
        "--prompt-scores",
        default=None,
        help="Optional prompt classifier score JSONL.",
    )
    parser.add_argument("--output-json", required=True, help="Output evaluation report JSON.")
    parser.add_argument("--output-md", default=None, help="Optional Markdown report.")
    parser.add_argument(
        "--feature-table",
        default=None,
        help="Optional joined feature table JSONL.",
    )
    parser.add_argument(
        "--embedding-weight",
        type=float,
        default=0.25,
        help="Weight for embedding prior in fixed cumulative fusion.",
    )
    parser.add_argument(
        "--activation-weight",
        type=float,
        default=0.75,
        help="Weight for activation evidence in fixed cumulative fusion.",
    )
    parser.add_argument(
        "--prompt-weight",
        type=float,
        default=1.0,
        help="Weight for prompt classifier score when --prompt-scores is provided.",
    )
    parser.add_argument(
        "--embedding-margin-scale",
        type=float,
        default=4.0,
        help="Scale applied before sigmoid-transforming embedding margin.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def label_to_binary(label: Any) -> int | None:
    normalized = str(label).lower()
    if normalized in {"unsafe", "harmful", "1", "true"}:
        return 1
    if normalized in {"safe", "benign", "0", "false"}:
        return 0
    return None


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row.get("id")
            if row_id is not None:
                rows[str(row_id)] = row
    return rows


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


def metrics(labels: list[int], scores: list[float], *, threshold: float) -> dict[str, float | int]:
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


def threshold_sweep(labels: list[int], scores: list[float]) -> dict[str, Any]:
    reports = [metrics(labels, scores, threshold=idx / 100) for idx in range(101)]
    return {
        "default_0_5": metrics(labels, scores, threshold=0.5),
        "best_f1": max(reports, key=lambda item: (item["f1"], item["balanced_accuracy"])),
        "best_balanced_accuracy": max(
            reports,
            key=lambda item: (item["balanced_accuracy"], item["f1"]),
        ),
        "target_fpr_0_05": max(
            (item for item in reports if item["false_positive_rate"] <= 0.05),
            key=lambda item: (item["recall"], item["precision"]),
            default=min(reports, key=lambda item: item["false_positive_rate"]),
        ),
    }


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    sorted_values = sorted(values)
    return {
        "count": len(values),
        "mean": mean(values),
        "min": sorted_values[0],
        "p50": sorted_values[len(sorted_values) // 2],
        "max": sorted_values[-1],
    }


def evaluate_score(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    labeled = [
        (label_to_binary(row["label"]), float(row[score_key]), row)
        for row in rows
        if label_to_binary(row.get("label")) is not None
    ]
    labels = [int(label) for label, _, _ in labeled if label is not None]
    scores = [score for _, score, _ in labeled]
    by_label: dict[str, list[float]] = defaultdict(list)
    for _, score, row in labeled:
        by_label[str(row["label"])].append(score)
    return {
        "score_key": score_key,
        "auc": auc(labels, scores),
        "metrics": threshold_sweep(labels, scores),
        "score_by_label": {
            label: distribution(values) for label, values in sorted(by_label.items())
        },
    }


def slice_report(
    rows: list[dict[str, Any]],
    score_key: str,
    group_key: str,
    threshold: float,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key, "unknown"))].append(row)

    output: dict[str, Any] = {}
    for group, group_rows in sorted(grouped.items()):
        labeled = [
            (label_to_binary(row["label"]), float(row[score_key]))
            for row in group_rows
            if label_to_binary(row.get("label")) is not None
        ]
        if not labeled:
            continue
        labels = [int(label) for label, _ in labeled if label is not None]
        scores = [score for _, score in labeled]
        output[group] = {
            "rows": len(group_rows),
            "labels": dict(Counter(str(row.get("label", "unknown")) for row in group_rows)),
            "auc": auc(labels, scores),
            "metrics": metrics(labels, scores, threshold=threshold),
        }
    return output


def hard_rows(
    rows: list[dict[str, Any]],
    score_key: str,
    *,
    top_k: int,
) -> dict[str, Any]:
    safe_high = [
        row for row in rows if label_to_binary(row.get("label")) == 0
    ]
    unsafe_low = [
        row for row in rows if label_to_binary(row.get("label")) == 1
    ]
    safe_high.sort(key=lambda row: float(row[score_key]), reverse=True)
    unsafe_low.sort(key=lambda row: float(row[score_key]))

    def compact(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row["id"],
            "label": row["label"],
            "source": row["source"],
            "domain": row["domain"],
            "subcluster_id": row["subcluster_id"],
            "score": row[score_key],
            "prompt_risk_score": row.get("prompt_risk_score"),
            "embedding_margin": row["embedding_margin"],
            "activation_probability": row["activation_probability"],
            "span_text": row["span_text"],
        }

    return {
        "top_safe_high_score": [compact(row) for row in safe_high[:top_k]],
        "top_unsafe_low_score": [compact(row) for row in unsafe_low[:top_k]],
    }


def build_feature_table(
    *,
    embedding_rows: dict[str, dict[str, Any]],
    activation_rows: dict[str, dict[str, Any]],
    prompt_rows: dict[str, dict[str, Any]] | None,
    probe: LinearActivationProbe,
    prompt_weight: float,
    embedding_weight: float,
    activation_weight: float,
    embedding_margin_scale: float,
) -> list[dict[str, Any]]:
    total_weight = embedding_weight + activation_weight
    if total_weight <= 0:
        raise ValueError("embedding and activation weights must sum to a positive value")

    rows = []
    joined_ids = set(embedding_rows) & set(activation_rows)
    if prompt_rows is not None:
        joined_ids &= set(prompt_rows)

    for row_id in sorted(joined_ids):
        embedding = embedding_rows[row_id]
        activation = activation_rows[row_id]
        prompt = prompt_rows[row_id] if prompt_rows is not None else None
        label = embedding.get("label", activation.get("label"))
        if label_to_binary(label) is None:
            continue
        prompt_risk_score = (
            float(prompt["prompt_risk_score"])
            if prompt is not None and prompt.get("prompt_risk_score") is not None
            else None
        )
        embedding_margin = float(embedding["risk_margin"])
        embedding_prior_score = sigmoid(embedding_margin * embedding_margin_scale)
        activation_probability = probe.probability(activation["embedding"])
        cumulative_fixed_score = (
            embedding_weight * embedding_prior_score
            + activation_weight * activation_probability
        ) / total_weight
        full_total_weight = prompt_weight + embedding_weight + activation_weight
        cumulative_full_score = (
            (
                prompt_weight * prompt_risk_score
                + embedding_weight * embedding_prior_score
                + activation_weight * activation_probability
            )
            / full_total_weight
            if prompt_risk_score is not None and full_total_weight > 0
            else None
        )
        rows.append(
            {
                "id": row_id,
                "label": label,
                "source": embedding.get("source", activation.get("source")),
                "domain": embedding.get("domain", activation.get("domain")),
                "subcluster_role": embedding.get(
                    "subcluster_role",
                    activation.get("subcluster_role"),
                ),
                "subcluster_id": embedding.get("subcluster_id", activation.get("subcluster_id")),
                "span_text": embedding.get("span_text", activation.get("span_text")),
                "prompt_risk_score": prompt_risk_score,
                "prompt_confidence": prompt.get("prompt_confidence") if prompt else None,
                "prompt_label": prompt.get("prompt_label") if prompt else None,
                "prompt_harm_category": prompt.get("prompt_harm_category") if prompt else None,
                "prompt_classifier_version": (
                    prompt.get("prompt_classifier_version") if prompt else None
                ),
                "embedding_margin": embedding_margin,
                "embedding_prior_score": embedding_prior_score,
                "embedding_top_risk_cluster": embedding.get("top_risk_cluster"),
                "embedding_top_benign_cluster": embedding.get("top_benign_cluster"),
                "activation_probability": activation_probability,
                "activation_layer": probe.layer_id,
                "cumulative_fixed_score": cumulative_fixed_score,
                "cumulative_full_score": cumulative_full_score,
                "prompt_embedding_score": (
                    (
                        prompt_weight * prompt_risk_score
                        + embedding_weight * embedding_prior_score
                    )
                    / (prompt_weight + embedding_weight)
                    if prompt_risk_score is not None and prompt_weight + embedding_weight > 0
                    else None
                ),
                "prompt_activation_score": (
                    (
                        prompt_weight * prompt_risk_score
                        + activation_weight * activation_probability
                    )
                    / (prompt_weight + activation_weight)
                    if prompt_risk_score is not None and prompt_weight + activation_weight > 0
                    else None
                ),
            }
        )
    return rows


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Cumulative Internal-Signal Evaluation",
        "",
        "This report evaluates accumulated internal evidence, not signal replacement.",
        "",
        "| Ablation | AUC | Recall at <=5% FPR | FPR | Threshold |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, result in report["ablations"].items():
        metric = result["metrics"]["target_fpr_0_05"]
        auc_value = result["auc"]
        auc_text = "n/a" if auc_value is None else f"{auc_value:.4f}"
        lines.append(
            f"| {name} | {auc_text} | {metric['recall']:.4f} | "
            f"{metric['false_positive_rate']:.4f} | {metric['threshold']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    embedding_rows = load_jsonl_by_id(Path(args.embedding_scores))
    activation_rows = load_jsonl_by_id(Path(args.activation))
    prompt_rows = load_jsonl_by_id(Path(args.prompt_scores)) if args.prompt_scores else None
    probe = LinearActivationProbe.from_artifact(args.activation_probe)
    feature_rows = build_feature_table(
        embedding_rows=embedding_rows,
        activation_rows=activation_rows,
        prompt_rows=prompt_rows,
        probe=probe,
        prompt_weight=args.prompt_weight,
        embedding_weight=args.embedding_weight,
        activation_weight=args.activation_weight,
        embedding_margin_scale=args.embedding_margin_scale,
    )
    if not feature_rows:
        raise ValueError("no joined labeled rows found")

    ablations = {
        "embedding_only": evaluate_score(feature_rows, "embedding_prior_score"),
        "activation_only": evaluate_score(feature_rows, "activation_probability"),
        "cumulative_fixed": evaluate_score(feature_rows, "cumulative_fixed_score"),
    }
    if prompt_rows is not None:
        ablations.update(
            {
                "prompt_only": evaluate_score(feature_rows, "prompt_risk_score"),
                "prompt_embedding": evaluate_score(feature_rows, "prompt_embedding_score"),
                "prompt_activation": evaluate_score(feature_rows, "prompt_activation_score"),
                "prompt_embedding_activation": evaluate_score(
                    feature_rows,
                    "cumulative_full_score",
                ),
            }
        )
    cumulative_threshold = ablations["cumulative_fixed"]["metrics"]["target_fpr_0_05"][
        "threshold"
    ]
    report = {
        "num_rows": len(feature_rows),
        "labels": dict(Counter(str(row["label"]) for row in feature_rows)),
        "embedding_scores_path": args.embedding_scores,
        "prompt_scores_path": args.prompt_scores,
        "activation_path": args.activation,
        "activation_probe_path": args.activation_probe,
        "fusion": {
            "method": "fixed_weighted_sum",
            "prompt_weight": args.prompt_weight if prompt_rows is not None else None,
            "embedding_weight": args.embedding_weight,
            "activation_weight": args.activation_weight,
            "embedding_margin_scale": args.embedding_margin_scale,
        },
        "ablations": ablations,
        "cumulative_domain_slices": slice_report(
            feature_rows,
            "cumulative_fixed_score",
            "domain",
            cumulative_threshold,
        ),
        "cumulative_source_slices": slice_report(
            feature_rows,
            "cumulative_fixed_score",
            "source",
            cumulative_threshold,
        ),
        "hard_rows": hard_rows(
            feature_rows,
            "cumulative_fixed_score",
            top_k=args.top_k,
        ),
        "interpretation": (
            "Embedding proximity is treated as an early semantic prior and activation probability "
            "as later internal-state evidence. The fixed cumulative score tests whether these "
            "signals can be accumulated into one risk estimate rather than choosing a single "
            "winner."
        ),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote cumulative internal-signal report to {output_json}")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown_report(report) + "\n", encoding="utf-8")
        print(f"wrote cumulative internal-signal markdown to {output_md}")

    if args.feature_table:
        feature_table = Path(args.feature_table)
        feature_table.parent.mkdir(parents=True, exist_ok=True)
        with feature_table.open("w", encoding="utf-8") as output_file:
            for row in feature_rows:
                output_file.write(json.dumps(row, separators=(",", ":")) + "\n")
        print(f"wrote cumulative internal-signal feature table to {feature_table}")

    for name, result in ablations.items():
        metric = result["metrics"]["target_fpr_0_05"]
        auc_value = result["auc"]
        auc_text = "n/a" if auc_value is None else f"{auc_value:.4f}"
        print(
            f"{name}: auc={auc_text} recall={metric['recall']:.4f} "
            f"fpr={metric['false_positive_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
