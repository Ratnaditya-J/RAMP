#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from build_review_batch_v0_2 import (
    REVIEW_LABEL_OPTIONS,
    is_hard_benign,
    load_jsonl,
    normalized_corpus_label,
)
from evaluate_ramp_harness import HIGH_SEVERITY_DOMAINS

BUCKET_ORDER = (
    "activation_false_negative_candidate",
    "severe_activation_false_negative_candidate",
    "embedding_false_positive_candidate",
    "embedding_activation_conflict_candidate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a targeted v0.4 review batch for activation-FN and embedding-FP "
            "weight-policy uncertainty."
        )
    )
    parser.add_argument("--feature-table", required=True, help="Full feature table JSONL.")
    parser.add_argument(
        "--exclude-csv",
        action="append",
        default=[],
        help="CSV with source_id rows to exclude. Repeat for reviewed and queued batches.",
    )
    parser.add_argument("--output-jsonl", required=True, help="Reviewer batch JSONL.")
    parser.add_argument("--output-csv", default=None, help="Optional reviewer CSV.")
    parser.add_argument("--summary-output", default=None, help="Optional summary JSON.")
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--max-per-bucket", type=int, default=150)
    parser.add_argument("--max-per-stratum", type=int, default=24)
    parser.add_argument("--activation-fn-max", type=float, default=0.25)
    parser.add_argument("--embedding-fp-min", type=float, default=0.45)
    parser.add_argument("--conflict-margin-min", type=float, default=0.35)
    return parser.parse_args()


def source_ids_from_csv(path: Path) -> set[str]:
    ids = set()
    if not path.exists():
        return ids
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        for row in csv.DictReader(input_file):
            source_id = str(row.get("source_id", "")).strip()
            if source_id:
                ids.add(source_id)
    return ids


def excluded_ids(paths: list[str]) -> set[str]:
    ids = set()
    for path in paths:
        ids.update(source_ids_from_csv(Path(path)))
    return ids


def score(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def is_high_severity(row: dict[str, Any]) -> bool:
    return str(row.get("domain", "")).strip().lower() in HIGH_SEVERITY_DOMAINS


def candidate_buckets(
    row: dict[str, Any],
    *,
    activation_fn_max: float,
    embedding_fp_min: float,
    conflict_margin_min: float,
) -> list[str]:
    corpus_label = normalized_corpus_label(row)
    activation = score(row, "activation_probability")
    embedding = score(row, "embedding_prior_score")
    prompt = score(row, "prompt_risk_score")
    buckets = []

    if corpus_label == "unsafe" and activation <= activation_fn_max:
        if is_high_severity(row):
            buckets.append("severe_activation_false_negative_candidate")
        else:
            buckets.append("activation_false_negative_candidate")

    if (
        corpus_label == "safe"
        and is_hard_benign(row)
        and embedding >= embedding_fp_min
    ):
        buckets.append("embedding_false_positive_candidate")

    if abs(embedding - activation) >= conflict_margin_min:
        if corpus_label == "unsafe" and activation <= activation_fn_max:
            buckets.append("embedding_activation_conflict_candidate")
        elif corpus_label == "safe" and is_hard_benign(row) and embedding >= embedding_fp_min:
            buckets.append("embedding_activation_conflict_candidate")
        elif prompt >= 0.50 and activation <= activation_fn_max:
            buckets.append("embedding_activation_conflict_candidate")

    return buckets


def priority(row: dict[str, Any], bucket: str) -> float:
    activation = score(row, "activation_probability")
    embedding = score(row, "embedding_prior_score")
    prompt = score(row, "prompt_risk_score")
    margin = score(row, "embedding_margin")
    severity_bonus = 0.4 if is_high_severity(row) else 0.0

    if bucket == "activation_false_negative_candidate":
        return (1.0 - activation) + max(prompt, embedding) + severity_bonus
    if bucket == "severe_activation_false_negative_candidate":
        return (1.0 - activation) + prompt + embedding + 1.0
    if bucket == "embedding_false_positive_candidate":
        return embedding + margin + prompt + (0.25 if is_hard_benign(row) else 0.0)
    if bucket == "embedding_activation_conflict_candidate":
        return abs(embedding - activation) + abs(prompt - activation) + severity_bonus
    return 0.0


def stratum_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("selection_bucket", "unknown")),
        str(row.get("source", "unknown")),
        str(row.get("domain", "unknown")),
    )


def select_rows(
    rows: list[dict[str, Any]],
    *,
    excluded_source_ids: set[str],
    max_rows: int,
    max_per_bucket: int,
    max_per_stratum: int,
    activation_fn_max: float,
    embedding_fp_min: float,
    conflict_margin_min: float,
) -> list[dict[str, Any]]:
    bucketed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_bucket_ids: set[tuple[str, str]] = set()
    for row in rows:
        row_id = str(row.get("id", "")).strip()
        if not row_id or row_id in excluded_source_ids:
            continue
        if (
            row.get("prompt_risk_score") is None
            or row.get("embedding_prior_score") is None
            or row.get("activation_probability") is None
        ):
            continue
        for bucket in candidate_buckets(
            row,
            activation_fn_max=activation_fn_max,
            embedding_fp_min=embedding_fp_min,
            conflict_margin_min=conflict_margin_min,
        ):
            key = (bucket, row_id)
            if key in seen_bucket_ids:
                continue
            seen_bucket_ids.add(key)
            candidate = dict(row)
            candidate["selection_bucket"] = bucket
            candidate["selection_priority"] = priority(row, bucket)
            bucketed[bucket].append(candidate)

    grouped: dict[tuple[str, str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for bucket in BUCKET_ORDER:
        for row in sorted(
            bucketed[bucket],
            key=lambda item: float(item.get("selection_priority", 0.0)),
            reverse=True,
        )[:max_per_bucket]:
            grouped[stratum_key(row)].append(row)

    keys_by_bucket = {
        bucket: [key for key in sorted(grouped) if key[0] == bucket]
        for bucket in BUCKET_ORDER
    }
    max_bucket_strata = max((len(keys) for keys in keys_by_bucket.values()), default=0)
    active_keys = deque(
        keys_by_bucket[bucket][idx]
        for idx in range(max_bucket_strata)
        for bucket in BUCKET_ORDER
        if idx < len(keys_by_bucket[bucket])
    )

    selected = []
    selected_ids = set()
    stratum_counts: Counter[tuple[str, str, str]] = Counter()
    while active_keys and len(selected) < max_rows:
        key = active_keys.popleft()
        group = grouped[key]
        if not group or stratum_counts[key] >= max_per_stratum:
            continue
        row = group.popleft()
        row_id = str(row.get("id"))
        if row_id not in selected_ids:
            selected.append(row)
            selected_ids.add(row_id)
            stratum_counts[key] += 1
        if group and stratum_counts[key] < max_per_stratum:
            active_keys.append(key)
    return selected


def review_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "review_id": f"prompt_review_v0_4_{index:04d}",
        "source_id": row.get("id"),
        "review_status": "unreviewed",
        "reviewed_label": "",
        "review_label_options": list(REVIEW_LABEL_OPTIONS),
        "label_issue_type": "",
        "reviewer_notes": "",
        "reviewed_by": "",
        "reviewed_at": "",
        "prompt_text": row.get("span_text", ""),
        "corpus_label": row.get("label"),
        "qwen_label": row.get("prompt_label"),
        "qwen_risk_score": row.get("prompt_risk_score"),
        "qwen_harm_category": row.get("prompt_harm_category"),
        "selection_bucket": row.get("selection_bucket"),
        "selection_priority": row.get("selection_priority"),
        "source": row.get("source"),
        "domain": row.get("domain"),
        "subcluster_role": row.get("subcluster_role"),
        "subcluster_id": row.get("subcluster_id"),
        "activation_probability": row.get("activation_probability"),
        "embedding_prior_score": row.get("embedding_prior_score"),
        "embedding_margin": row.get("embedding_margin"),
        "embedding_top_risk_cluster": row.get("embedding_top_risk_cluster"),
        "embedding_top_benign_cluster": row.get("embedding_top_benign_cluster"),
        "cumulative_full_score": row.get("cumulative_full_score"),
        "prompt_classifier_version": row.get("prompt_classifier_version"),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "review_id",
        "review_status",
        "reviewed_label",
        "label_issue_type",
        "reviewer_notes",
        "prompt_text",
        "corpus_label",
        "qwen_label",
        "qwen_risk_score",
        "qwen_harm_category",
        "selection_bucket",
        "source",
        "domain",
        "subcluster_role",
        "subcluster_id",
        "activation_probability",
        "embedding_prior_score",
        "embedding_margin",
        "embedding_top_risk_cluster",
        "embedding_top_benign_cluster",
        "cumulative_full_score",
        "source_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_summary(rows: list[dict[str, Any]], *, excluded_count: int) -> dict[str, Any]:
    return {
        "artifact_id": "ramp_prompt_review_batch_v0.4",
        "purpose": (
            "Expand reviewed labels around activation false-negative candidates and "
            "embedding false-positive candidates before final weight-policy selection."
        ),
        "num_rows": len(rows),
        "excluded_source_ids": excluded_count,
        "review_label_options": list(REVIEW_LABEL_OPTIONS),
        "by_selection_bucket": dict(Counter(str(row.get("selection_bucket")) for row in rows)),
        "by_source": dict(Counter(str(row.get("source")) for row in rows)),
        "by_domain": dict(Counter(str(row.get("domain")) for row in rows)),
    }


def main() -> None:
    args = parse_args()
    excluded_source_ids = excluded_ids(args.exclude_csv)
    selected = select_rows(
        load_jsonl(Path(args.feature_table)),
        excluded_source_ids=excluded_source_ids,
        max_rows=args.max_rows,
        max_per_bucket=args.max_per_bucket,
        max_per_stratum=args.max_per_stratum,
        activation_fn_max=args.activation_fn_max,
        embedding_fp_min=args.embedding_fp_min,
        conflict_margin_min=args.conflict_margin_min,
    )
    review_rows = [review_row(row, index + 1) for index, row in enumerate(selected)]

    write_jsonl(Path(args.output_jsonl), review_rows)
    print(f"wrote v0.4 review JSONL to {args.output_jsonl}")
    if args.output_csv:
        write_csv(Path(args.output_csv), review_rows)
        print(f"wrote v0.4 review CSV to {args.output_csv}")

    summary = build_summary(review_rows, excluded_count=len(excluded_source_ids))
    if args.summary_output:
        output_path = Path(args.summary_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"wrote v0.4 review summary to {args.summary_output}")

    print(f"review_rows={summary['num_rows']}")
    print(f"by_selection_bucket={summary['by_selection_bucket']}")


if __name__ == "__main__":
    main()
