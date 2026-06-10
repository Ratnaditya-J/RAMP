#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from evaluate_reviewed_cumulative_signals import (
    binary_label,
    join_rows,
    load_feature_table,
    load_reviewed_rows,
)

BUCKET_ORDER = (
    "prompt_activation_false_negative",
    "prompt_false_positive_hard_benign",
    "high_risk_true_positive",
    "low_risk_true_negative",
    "output_policy_boundary",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a prompt/response evaluation set for output classifier research."
    )
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--feature-table", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--max-rows", type=int, default=300)
    parser.add_argument("--max-per-bucket", type=int, default=80)
    parser.add_argument("--max-per-domain", type=int, default=25)
    parser.add_argument("--policy-threshold", type=float, default=0.53)
    return parser.parse_args()


def score(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def prompt_activation_score(row: dict[str, Any]) -> float:
    return 0.25 * score(row, "prompt_risk_score") + 0.75 * score(
        row, "activation_probability"
    )


def is_hard_benign(row: dict[str, Any]) -> bool:
    if str(row.get("subcluster_role", "")).strip().lower() == "benign_near_neighbor":
        return True
    return str(row.get("label_issue_type", "")).strip().lower() in {
        "prompt_classifier_wrong",
        "ambiguous_policy_boundary",
    }


def candidate_bucket(row: dict[str, Any], *, threshold: float) -> str:
    label = binary_label(str(row.get("reviewed_label")))
    policy_score = prompt_activation_score(row)
    prompt = score(row, "prompt_risk_score")
    activation = score(row, "activation_probability")
    if label == 1 and policy_score < threshold:
        return "prompt_activation_false_negative"
    if label == 0 and policy_score >= threshold and is_hard_benign(row):
        return "prompt_false_positive_hard_benign"
    if label == 1 and policy_score >= threshold:
        return "high_risk_true_positive"
    if label == 0 and policy_score < threshold:
        return "low_risk_true_negative"
    if abs(prompt - activation) >= 0.35:
        return "output_policy_boundary"
    return "output_policy_boundary"


def priority(row: dict[str, Any], bucket: str, *, threshold: float) -> float:
    policy_score = prompt_activation_score(row)
    prompt = score(row, "prompt_risk_score")
    activation = score(row, "activation_probability")
    embedding = score(row, "embedding_prior_score")
    if bucket == "prompt_activation_false_negative":
        return (threshold - policy_score) + prompt + embedding
    if bucket == "prompt_false_positive_hard_benign":
        return (policy_score - threshold) + prompt + activation
    if bucket == "high_risk_true_positive":
        return policy_score + prompt + activation
    if bucket == "low_risk_true_negative":
        return (1.0 - policy_score) + (1.0 - prompt)
    return abs(prompt - activation) + embedding


def select_rows(
    rows: list[dict[str, Any]],
    *,
    max_rows: int,
    max_per_bucket: int,
    max_per_domain: int,
    threshold: float,
) -> list[dict[str, Any]]:
    bucketed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if binary_label(str(row.get("reviewed_label"))) is None:
            continue
        if not str(row.get("prompt_text") or row.get("span_text") or "").strip():
            continue
        bucket = candidate_bucket(row, threshold=threshold)
        candidate = dict(row)
        candidate["selection_bucket"] = bucket
        candidate["selection_priority"] = priority(candidate, bucket, threshold=threshold)
        candidate["input_side_policy_score"] = prompt_activation_score(candidate)
        bucketed[bucket].append(candidate)

    grouped: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for bucket in BUCKET_ORDER:
        for row in sorted(
            bucketed[bucket],
            key=lambda item: float(item["selection_priority"]),
            reverse=True,
        )[:max_per_bucket]:
            grouped[(bucket, str(row.get("domain", "unknown")))].append(row)

    active_keys = deque(sorted(grouped))
    selected = []
    selected_ids = set()
    domain_counts: Counter[str] = Counter()
    while active_keys and len(selected) < max_rows:
        key = active_keys.popleft()
        group = grouped[key]
        if not group:
            continue
        domain = key[1]
        if domain_counts[domain] >= max_per_domain:
            continue
        row = group.popleft()
        row_id = str(row.get("source_id") or row.get("id"))
        if row_id not in selected_ids:
            selected.append(row)
            selected_ids.add(row_id)
            domain_counts[domain] += 1
        if group and domain_counts[domain] < max_per_domain:
            active_keys.append(key)
    return selected


def eval_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "eval_id": f"output_eval_v0_1_{index:04d}",
        "source_id": row.get("source_id") or row.get("id"),
        "reviewed_label": row.get("reviewed_label"),
        "prompt_text": row.get("prompt_text") or row.get("span_text"),
        "output_text": "",
        "output_review_status": "needs_generation",
        "output_reviewed_label": "",
        "output_reviewer_notes": "",
        "selection_bucket": row.get("selection_bucket"),
        "selection_priority": row.get("selection_priority"),
        "input_side_policy_score": row.get("input_side_policy_score"),
        "prompt_risk_score": row.get("prompt_risk_score"),
        "activation_probability": row.get("activation_probability"),
        "embedding_prior_score": row.get("embedding_prior_score"),
        "source": row.get("source"),
        "domain": row.get("domain"),
        "subcluster_role": row.get("subcluster_role"),
        "subcluster_id": row.get("subcluster_id"),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    reviewed = load_reviewed_rows(Path(args.review_csv))
    features = load_feature_table(Path(args.feature_table))
    joined_rows = join_rows(reviewed, features)
    selected = select_rows(
        joined_rows,
        max_rows=args.max_rows,
        max_per_bucket=args.max_per_bucket,
        max_per_domain=args.max_per_domain,
        threshold=args.policy_threshold,
    )
    output_rows = [eval_row(row, index + 1) for index, row in enumerate(selected)]
    write_jsonl(Path(args.output_jsonl), output_rows)
    if args.output_csv:
        write_csv(Path(args.output_csv), output_rows)
    summary = {
        "artifact_id": "ramp_output_eval_set_v0.1",
        "num_rows": len(output_rows),
        "review_csv": args.review_csv,
        "feature_table": args.feature_table,
        "policy_threshold": args.policy_threshold,
        "by_selection_bucket": dict(Counter(row["selection_bucket"] for row in output_rows)),
        "by_domain": dict(Counter(row["domain"] for row in output_rows)),
        "next_step": "Fill output_text with generated responses before batch scoring output risk.",
    }
    if args.summary_output:
        output_path = Path(args.summary_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote output eval JSONL to {args.output_jsonl}")
    if args.output_csv:
        print(f"wrote output eval CSV to {args.output_csv}")
    print(f"rows={len(output_rows)} by_selection_bucket={summary['by_selection_bucket']}")


if __name__ == "__main__":
    main()
