#!/usr/bin/env python
"""Compare two binary safe/unsafe label sets on their shared rows (Cohen's kappa).

Used to (a) measure inter-judge agreement between two independent LLM judges on the blind
set, and later (b) measure judge-vs-human agreement on the audit subset. High agreement
between independent judges makes the silver blind labels more credible ahead of the human
audit; the human audit remains the final check.

Reads the reviewed-schema CSVs (review_status, reviewed_label, source_id). Only rows that
are 'reviewed' with a binary safe/unsafe label in BOTH sets are compared.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cohen's kappa between two label sets.")
    parser.add_argument("--labels-a", required=True)
    parser.add_argument("--labels-b", required=True)
    parser.add_argument("--name-a", default="a")
    parser.add_argument("--name-b", default="b")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def load_binary_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with path.open(encoding="utf-8-sig", newline="") as input_file:
        for row in csv.DictReader(input_file):
            status = str(row.get("review_status", "")).strip().lower()
            label = str(row.get("reviewed_label", "")).strip().lower()
            source_id = str(row.get("source_id", "")).strip()
            if status != "reviewed" or not source_id:
                continue
            if label == "unsafe":
                labels[source_id] = 1
            elif label == "safe":
                labels[source_id] = 0
    return labels


def cohen_kappa(labels_a: dict[str, int], labels_b: dict[str, int]) -> dict[str, Any]:
    shared = sorted(set(labels_a) & set(labels_b))
    n = len(shared)
    if n == 0:
        return {"n": 0, "kappa": None, "raw_agreement": None}
    agree = sum(1 for i in shared if labels_a[i] == labels_b[i])
    po = agree / n
    # marginals
    a_pos = sum(labels_a[i] for i in shared) / n
    b_pos = sum(labels_b[i] for i in shared) / n
    pe = a_pos * b_pos + (1 - a_pos) * (1 - b_pos)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else None
    # 2x2 confusion: rows = a label, cols = b label
    confusion = {"a0_b0": 0, "a0_b1": 0, "a1_b0": 0, "a1_b1": 0}
    for i in shared:
        confusion[f"a{labels_a[i]}_b{labels_b[i]}"] += 1
    return {
        "n": n,
        "raw_agreement": po,
        "kappa": kappa,
        "a_unsafe_rate": a_pos,
        "b_unsafe_rate": b_pos,
        "confusion": confusion,
    }


def main() -> None:
    args = parse_args()
    labels_a = load_binary_labels(Path(args.labels_a))
    labels_b = load_binary_labels(Path(args.labels_b))
    result = cohen_kappa(labels_a, labels_b)
    result["name_a"] = args.name_a
    result["name_b"] = args.name_b
    result["labels_a_count"] = len(labels_a)
    result["labels_b_count"] = len(labels_b)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    kappa = result["kappa"]
    raw = result["raw_agreement"]
    print(f"{args.name_a} vs {args.name_b}: shared={result['n']}")
    if kappa is not None:
        print(f"  raw agreement={raw:.4f}  Cohen's kappa={kappa:.4f}")
        print(
            f"  {args.name_a} unsafe rate={result['a_unsafe_rate']:.3f}  "
            f"{args.name_b} unsafe rate={result['b_unsafe_rate']:.3f}"
        )
        print(f"  confusion={result['confusion']}")


if __name__ == "__main__":
    main()
