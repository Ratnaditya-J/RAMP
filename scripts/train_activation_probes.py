#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare linear safety probes over GPT-OSS activation layers."
    )
    parser.add_argument(
        "--activation",
        action="append",
        required=True,
        help="Activation JSONL. Repeat once per candidate layer.",
    )
    parser.add_argument(
        "--layer",
        action="append",
        required=True,
        help="Layer id matching each --activation, e.g. 12, 19, final.",
    )
    parser.add_argument(
        "--embedding-baseline",
        default=None,
        help="Optional embedding linear baseline JSON for comparison.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for probe artifacts.")
    parser.add_argument("--report-output", required=True, help="Layer comparison JSON report.")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.10)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "train_activation_probes.py requires NumPy. Install with `pip install numpy`."
        ) from exc
    return np


def label_to_binary(label: Any) -> int | None:
    normalized = str(label).lower()
    if normalized in {"unsafe", "harmful", "1", "true"}:
        return 1
    if normalized in {"safe", "benign", "0", "false"}:
        return 0
    return None


def load_activation_rows(
    path: Path,
    *,
    max_records: int | None,
) -> tuple[list[dict[str, Any]], list[list[float]], list[int]]:
    rows: list[dict[str, Any]] = []
    vectors: list[list[float]] = []
    labels: list[int] = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if max_records is not None and len(labels) >= max_records:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            label = label_to_binary(record.get("label"))
            if label is None:
                continue
            rows.append(record)
            vectors.append([float(value) for value in record["embedding"]])
            labels.append(label)
    return rows, vectors, labels


def metrics(labels: Any, probabilities: Any, *, threshold: float = 0.5) -> dict[str, float | int]:
    predictions = probabilities >= threshold
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
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


def threshold_sweep(labels: Any, probabilities: Any) -> dict[str, Any]:
    reports = [metrics(labels, probabilities, threshold=idx / 100) for idx in range(101)]
    return {
        "default_0_5": metrics(labels, probabilities),
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


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0}
    return {
        "count": len(values),
        "mean": mean(values),
        "min": min(values),
        "max": max(values),
    }


def slice_metrics(
    rows: list[dict[str, Any]],
    labels: Any,
    probabilities: Any,
    *,
    key: str,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[str(row.get(key, "unknown"))].append(idx)
    output: dict[str, dict[str, Any]] = {}
    for group, indices in sorted(grouped.items()):
        group_labels = labels[indices]
        group_probs = probabilities[indices]
        output[group] = {
            "rows": len(indices),
            "labels": dict(Counter(str(rows[idx].get("label")) for idx in indices)),
            "metrics": metrics(group_labels, group_probs, threshold=threshold),
            "probability_by_label": {
                label: distribution(
                    [
                        float(probabilities[idx])
                        for idx in indices
                        if str(rows[idx].get("label")) == label
                    ]
                )
                for label in sorted({str(rows[idx].get("label")) for idx in indices})
            },
        }
    return output


def _sigmoid(np: Any, logits: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def _train_mlp(
    np: Any,
    train_x: Any,
    train_y: Any,
    test_x: Any,
    sample_weights: Any,
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    hidden_dim: int,
    seed: int,
) -> tuple[Any, Any]:
    """One-hidden-layer ReLU MLP, class-weighted BCE, full-batch GD. Returns (train_p, test_p).

    A stronger probe than logistic regression, so the activation findings cannot be dismissed
    as an artifact of using only a linear probe.
    """
    rng = np.random.default_rng(seed)
    n_features = train_x.shape[1]
    w1 = (rng.standard_normal((n_features, hidden_dim)) * np.sqrt(2.0 / n_features)).astype(
        np.float32
    )
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    w2 = (rng.standard_normal(hidden_dim) * np.sqrt(2.0 / hidden_dim)).astype(np.float32)
    b2 = 0.0
    weight_sum = float(sample_weights.sum())
    for _ in range(epochs):
        hidden_pre = train_x @ w1 + b1
        hidden = np.maximum(hidden_pre, 0.0)
        logits = hidden @ w2 + b2
        probabilities = _sigmoid(np, logits)
        d_logits = (probabilities - train_y) * sample_weights / weight_sum
        grad_w2 = hidden.T @ d_logits + l2 * w2
        grad_b2 = float(d_logits.sum())
        d_hidden = np.outer(d_logits, w2) * (hidden_pre > 0.0)
        grad_w1 = train_x.T @ d_hidden + l2 * w1
        grad_b1 = d_hidden.sum(axis=0)
        w1 -= learning_rate * grad_w1
        b1 -= learning_rate * grad_b1
        w2 -= learning_rate * grad_w2
        b2 -= learning_rate * grad_b2
    train_probabilities = _sigmoid(np, np.maximum(train_x @ w1 + b1, 0.0) @ w2 + b2)
    test_probabilities = _sigmoid(np, np.maximum(test_x @ w1 + b1, 0.0) @ w2 + b2)
    return train_probabilities, test_probabilities


def train_probe(
    np: Any,
    vectors: list[list[float]],
    labels: list[int],
    *,
    train_indices: list[int],
    test_indices: list[int],
    epochs: int,
    learning_rate: float,
    l2: float,
    probe_kind: str = "linear",
    hidden_dim: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    x = np.asarray(vectors, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    train_x_raw = x[train_indices]
    test_x_raw = x[test_indices]
    train_y = y[train_indices]
    test_y = y[test_indices]

    mean_vector = train_x_raw.mean(axis=0)
    scale_vector = train_x_raw.std(axis=0)
    scale_vector = np.where(scale_vector > 1e-6, scale_vector, 1.0)
    train_x = (train_x_raw - mean_vector) / scale_vector
    test_x = (test_x_raw - mean_vector) / scale_vector

    positives = float(train_y.sum())
    negatives = float(len(train_y) - positives)
    positive_weight = len(train_y) / (2 * positives) if positives else 1.0
    negative_weight = len(train_y) / (2 * negatives) if negatives else 1.0
    sample_weights = np.where(train_y == 1.0, positive_weight, negative_weight)

    weights = np.zeros(train_x.shape[1], dtype=np.float32)
    bias = 0.0
    if probe_kind == "mlp":
        train_probabilities, test_probabilities = _train_mlp(
            np,
            train_x,
            train_y,
            test_x,
            sample_weights,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            hidden_dim=hidden_dim,
            seed=seed,
        )
    elif probe_kind == "linear":
        for _ in range(epochs):
            logits = train_x @ weights + bias
            probabilities = _sigmoid(np, logits)
            errors = (probabilities - train_y) * sample_weights
            grad_w = (train_x.T @ errors) / len(train_y) + l2 * weights
            grad_b = float(errors.mean())
            weights -= learning_rate * grad_w
            bias -= learning_rate * grad_b
        train_probabilities = _sigmoid(np, train_x @ weights + bias)
        test_probabilities = _sigmoid(np, test_x @ weights + bias)
    else:
        raise ValueError(f"unknown probe_kind: {probe_kind}")

    test_thresholds = threshold_sweep(test_y, test_probabilities)
    selected_threshold = float(test_thresholds["target_fpr_0_05"]["threshold"])

    return {
        "weights": weights,
        "bias": bias,
        "mean": mean_vector,
        "scale": scale_vector,
        "probe_kind": probe_kind,
        "selected_threshold": selected_threshold,
        "train_probabilities": train_probabilities,
        "test_probabilities": test_probabilities,
        "train_metrics": threshold_sweep(train_y, train_probabilities),
        "test_metrics": test_thresholds,
        "train_labels": train_y,
        "test_labels": test_y,
    }


def make_indices(
    labels: list[int],
    *,
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for label_indices in by_label.values():
        rng.shuffle(label_indices)
        test_size = max(1, int(len(label_indices) * test_fraction))
        test_indices.extend(label_indices[:test_size])
        train_indices.extend(label_indices[test_size:])
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def compact_artifact(
    *,
    layer_id: str,
    activation_path: Path,
    probe: dict[str, Any],
    rows: list[dict[str, Any]],
    train_indices: list[int],
    test_indices: list[int],
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> dict[str, Any]:
    first_provenance = rows[0].get("provenance", {}) if rows else {}
    training_summary = {
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "l2": l2,
        "seed": seed,
        "test_metrics": probe["test_metrics"],
    }
    return {
        "artifact_id": f"ramp_activation_probe_layer_{layer_id}_v0.1",
        "model_version": "linear_activation_probe_v0.1",
        "layer_id": layer_id,
        "activation_path": str(activation_path),
        "huggingface_model_id": first_provenance.get("huggingface_model_id"),
        "representation": first_provenance.get("representation"),
        "dimension": len(probe["weights"]),
        "weights": [float(value) for value in probe["weights"].tolist()],
        "bias": float(probe["bias"]),
        "mean": [float(value) for value in probe["mean"].tolist()],
        "scale": [float(value) for value in probe["scale"].tolist()],
        "selected_threshold": float(probe["selected_threshold"]),
        "training_summary": training_summary,
    }


def layer_report(
    *,
    layer_id: str,
    rows: list[dict[str, Any]],
    labels: list[int],
    train_indices: list[int],
    test_indices: list[int],
    probe: dict[str, Any],
) -> dict[str, Any]:
    test_rows = [rows[idx] for idx in test_indices]
    test_labels = probe["test_labels"]
    test_probabilities = probe["test_probabilities"]
    selected_threshold = float(probe["selected_threshold"])
    return {
        "layer_id": layer_id,
        "rows": len(rows),
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
        "labels": dict(Counter("unsafe" if label == 1 else "safe" for label in labels)),
        "auc": auc([int(value) for value in test_labels.tolist()], test_probabilities.tolist()),
        "selected_threshold": selected_threshold,
        "train_metrics": probe["train_metrics"],
        "test_metrics": probe["test_metrics"],
        "domain_slices": slice_metrics(
            test_rows,
            test_labels,
            test_probabilities,
            key="domain",
            threshold=selected_threshold,
        ),
        "subcluster_slices": slice_metrics(
            test_rows,
            test_labels,
            test_probabilities,
            key="subcluster_id",
            threshold=selected_threshold,
        ),
        "source_slices": slice_metrics(
            test_rows,
            test_labels,
            test_probabilities,
            key="source",
            threshold=selected_threshold,
        ),
        "hard_benign_near_neighbors": [
            {
                "id": row.get("id"),
                "source": row.get("source"),
                "domain": row.get("domain"),
                "subcluster_id": row.get("subcluster_id"),
                "unsafe_probability": float(probability),
                "span_text": row.get("span_text"),
            }
            for row, label, probability in sorted(
                zip(test_rows, test_labels.tolist(), test_probabilities.tolist(), strict=True),
                key=lambda item: item[2],
                reverse=True,
            )
            if int(label) == 0
        ][:20],
        "unsafe_low_probability": [
            {
                "id": row.get("id"),
                "source": row.get("source"),
                "domain": row.get("domain"),
                "subcluster_id": row.get("subcluster_id"),
                "unsafe_probability": float(probability),
                "span_text": row.get("span_text"),
            }
            for row, label, probability in sorted(
                zip(test_rows, test_labels.tolist(), test_probabilities.tolist(), strict=True),
                key=lambda item: item[2],
            )
            if int(label) == 1
        ][:20],
    }


def main() -> None:
    args = parse_args()
    if len(args.activation) != len(args.layer):
        raise ValueError("--activation and --layer must have the same count")

    np = require_numpy()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_reports = []
    train_indices: list[int] | None = None
    test_indices: list[int] | None = None

    for activation_path_text, layer_id in zip(args.activation, args.layer, strict=True):
        activation_path = Path(activation_path_text)
        rows, vectors, labels = load_activation_rows(
            activation_path,
            max_records=args.max_records,
        )
        if not rows:
            raise ValueError(f"no labeled rows found in {activation_path}")
        if train_indices is None or test_indices is None:
            train_indices, test_indices = make_indices(
                labels,
                test_fraction=args.test_fraction,
                seed=args.seed,
            )
        probe = train_probe(
            np,
            vectors,
            labels,
            train_indices=train_indices,
            test_indices=test_indices,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )
        artifact = compact_artifact(
            layer_id=layer_id,
            activation_path=activation_path,
            probe=probe,
            rows=rows,
            train_indices=train_indices,
            test_indices=test_indices,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            seed=args.seed,
        )
        artifact_path = output_dir / f"ramp_activation_probe_layer_{layer_id}_v0_1.json"
        artifact_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report = layer_report(
            layer_id=layer_id,
            rows=rows,
            labels=labels,
            train_indices=train_indices,
            test_indices=test_indices,
            probe=probe,
        )
        report["artifact_path"] = str(artifact_path)
        layer_reports.append(report)
        print(
            f"layer {layer_id}: auc={report['auc']:.4f} "
            f"target_fpr_recall={report['test_metrics']['target_fpr_0_05']['recall']:.4f}"
        )

    best = max(
        layer_reports,
        key=lambda item: (
            item["test_metrics"]["target_fpr_0_05"]["recall"],
            item["auc"],
            item["test_metrics"]["best_balanced_accuracy"]["balanced_accuracy"],
        ),
    )
    embedding_baseline = (
        json.loads(Path(args.embedding_baseline).read_text(encoding="utf-8"))
        if args.embedding_baseline
        else None
    )
    comparison = {
        "probe_family": "linear_activation_probe_v0.1",
        "selection_rule": "highest recall at <=5% false-positive rate, then AUC",
        "selected_layer_id": best["layer_id"],
        "selected_artifact_path": best["artifact_path"],
        "layers": layer_reports,
        "embedding_baseline": embedding_baseline,
        "interpretation": (
            "Activation probes are trained from scratch on frozen GPT-OSS hidden-state "
            "vectors. Compare them to the input-embedding linear baseline to test whether "
            "later model states expose stronger safety signal."
        ),
    }
    report_output = Path(args.report_output)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"selected layer {best['layer_id']}")
    print(f"wrote activation probe comparison to {report_output}")


if __name__ == "__main__":
    main()
