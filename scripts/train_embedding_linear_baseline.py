#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small embedding-only linear baseline for comparison to centroids."
    )
    parser.add_argument("--embeddings", required=True, help="Embedding JSONL with labels.")
    parser.add_argument("--output", required=True, help="Output JSON report.")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "train_embedding_linear_baseline.py requires NumPy. "
            "Install it only for research runs, e.g. `pip install numpy`."
        ) from exc
    return np


def label_to_binary(label: Any) -> int | None:
    normalized = str(label).lower()
    if normalized in {"unsafe", "harmful", "1", "true"}:
        return 1
    if normalized in {"safe", "benign", "0", "false"}:
        return 0
    return None


def load_dataset(path: Path, *, max_records: int | None) -> tuple[list[list[float]], list[int]]:
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
            vectors.append([float(value) for value in record["embedding"]])
            labels.append(label)
    return vectors, labels


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
    thresholds = [idx / 100 for idx in range(101)]
    reports = [metrics(labels, probabilities, threshold=threshold) for threshold in thresholds]
    return {
        "default_0_5": metrics(labels, probabilities),
        "best_f1": max(reports, key=lambda item: (item["f1"], item["balanced_accuracy"])),
        "best_balanced_accuracy": max(
            reports,
            key=lambda item: (item["balanced_accuracy"], item["f1"]),
        ),
        "target_fpr_0_05": max(
            (
                item
                for item in reports
                if item["false_positive_rate"] <= 0.05
            ),
            key=lambda item: (item["recall"], item["precision"]),
            default=min(reports, key=lambda item: item["false_positive_rate"]),
        ),
    }


def main() -> None:
    args = parse_args()
    np = require_numpy()
    vectors, labels = load_dataset(Path(args.embeddings), max_records=args.max_records)
    if not vectors:
        raise ValueError("no labeled embedding rows found")

    rng = random.Random(args.seed)
    indices = list(range(len(labels)))
    rng.shuffle(indices)
    test_size = max(1, int(len(indices) * args.test_fraction))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    x = np.asarray(vectors, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    train_x = x[train_indices]
    train_y = y[train_indices]
    test_x = x[test_indices]
    test_y = y[test_indices]

    weights = np.zeros(train_x.shape[1], dtype=np.float32)
    bias = 0.0
    positives = float(train_y.sum())
    negatives = float(len(train_y) - positives)
    positive_weight = len(train_y) / (2 * positives) if positives else 1.0
    negative_weight = len(train_y) / (2 * negatives) if negatives else 1.0
    sample_weights = np.where(train_y == 1.0, positive_weight, negative_weight)
    for _ in range(args.epochs):
        logits = train_x @ weights + bias
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        errors = (probabilities - train_y) * sample_weights
        grad_w = (train_x.T @ errors) / len(train_y) + args.l2 * weights
        grad_b = float(errors.mean())
        weights -= args.learning_rate * grad_w
        bias -= args.learning_rate * grad_b

    train_probabilities = 1.0 / (
        1.0 + np.exp(-np.clip(train_x @ weights + bias, -30.0, 30.0))
    )
    test_probabilities = 1.0 / (
        1.0 + np.exp(-np.clip(test_x @ weights + bias, -30.0, 30.0))
    )
    report = {
        "embedding_rows": len(labels),
        "dimension": int(x.shape[1]),
        "train_rows": int(len(train_y)),
        "test_rows": int(len(test_y)),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "seed": args.seed,
        "class_weights": {
            "safe": float(negative_weight),
            "unsafe": float(positive_weight),
        },
        "train_metrics": threshold_sweep(train_y, train_probabilities),
        "test_metrics": threshold_sweep(test_y, test_probabilities),
        "interpretation": (
            "This is an embedding-only linear baseline. Use it to check whether input "
            "vectors contain more label information than centroid margins, not as a "
            "production classifier."
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote embedding linear baseline report to {output}")
    print(json.dumps(report["test_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
