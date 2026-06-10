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
    normalized_prompt_label,
    reviewed_ids,
)

BUCKET_ORDER = (
    "stability_severe_fn_slice",
    "stability_hard_benign_fp_slice",
    "stability_ramp_fn_slice",
    "stability_ramp_fp_slice",
    "fusion_prompt_disagreement",
    "undercovered_domain",
    "uncertain_margin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v0.3 review rows from repeated split-stability error distribution."
    )
    parser.add_argument("--feature-table", required=True, help="Full feature table JSONL.")
    parser.add_argument("--review-csv", required=True, help="Existing reviewed CSV to exclude.")
    parser.add_argument(
        "--stability-artifact",
        required=True,
        help="JSON output from scripts/evaluate_split_stability.py.",
    )
    parser.add_argument("--output-jsonl", required=True, help="Reviewer batch JSONL.")
    parser.add_argument("--output-csv", default=None, help="Optional reviewer CSV.")
    parser.add_argument("--summary-output", default=None, help="Optional summary JSON.")
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--max-per-bucket", type=int, default=125)
    parser.add_argument("--max-per-stratum", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--uncertainty-window", type=float, default=0.08)
    parser.add_argument("--min-domain-binary-reviewed", type=int, default=10)
    parser.add_argument("--max-error-slices", type=int, default=12)
    return parser.parse_args()


def load_stability(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def reviewed_domain_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        for row in csv.DictReader(input_file):
            status = str(row.get("review_status", "")).strip().lower()
            label = str(row.get("reviewed_label", "")).strip().lower()
            if status == "reviewed" and label in {"safe", "unsafe"}:
                counts[str(row.get("domain", "unknown"))] += 1
    return counts


def error_slice_keys(
    stability: dict[str, Any],
    key: str,
    *,
    max_error_slices: int,
) -> set[tuple[str, str]]:
    output = set()
    for row in stability.get("error_distribution", {}).get(key, [])[:max_error_slices]:
        domain = str(row.get("domain") or "")
        subcluster_id = str(row.get("subcluster_id") or "")
        if domain or subcluster_id:
            output.add((domain, subcluster_id))
    return output


def stability_slice_map(
    stability: dict[str, Any],
    *,
    max_error_slices: int,
) -> dict[str, set[tuple[str, str]]]:
    return {
        "stability_severe_fn_slice": error_slice_keys(
            stability,
            "ramp_fusion_severe_false_negatives",
            max_error_slices=max_error_slices,
        ),
        "stability_hard_benign_fp_slice": error_slice_keys(
            stability,
            "ramp_fusion_hard_benign_false_positives",
            max_error_slices=max_error_slices,
        ),
        "stability_ramp_fn_slice": error_slice_keys(
            stability,
            "ramp_fusion_false_negatives",
            max_error_slices=max_error_slices,
        ),
        "stability_ramp_fp_slice": error_slice_keys(
            stability,
            "ramp_fusion_false_positives",
            max_error_slices=max_error_slices,
        ),
    }


def score_value(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def approximate_ramp_score(row: dict[str, Any], stability: dict[str, Any]) -> float:
    config = (
        stability.get("selected_config_summary", {})
        .get("most_common_configs", [{}])[0]
    )
    prompt_weight = float(config.get("prompt_weight", 0.6))
    embedding_weight = float(config.get("embedding_weight", 0.2))
    activation_weight = float(config.get("activation_weight", 0.2))
    return (
        prompt_weight * score_value(row, "prompt_risk_score")
        + embedding_weight * score_value(row, "embedding_prior_score")
        + activation_weight * score_value(row, "activation_probability")
    )


def candidate_buckets(
    row: dict[str, Any],
    *,
    stability: dict[str, Any],
    slice_map: dict[str, set[tuple[str, str]]],
    undercovered_domains: set[str],
    threshold: float,
    uncertainty_window: float,
) -> list[str]:
    domain = str(row.get("domain") or "")
    subcluster_id = str(row.get("subcluster_id") or "")
    slice_key = (domain, subcluster_id)
    buckets = []
    corpus_label = normalized_corpus_label(row)
    prompt_label = normalized_prompt_label(row)
    ramp_score = approximate_ramp_score(row, stability)

    for bucket, slices in slice_map.items():
        if slice_key in slices:
            if bucket.endswith("_fn_slice") and corpus_label == "unsafe":
                buckets.append(bucket)
            elif bucket.endswith("_fp_slice") and corpus_label == "safe":
                buckets.append(bucket)

    if corpus_label == "safe" and ramp_score >= threshold and is_hard_benign(row):
        buckets.append("stability_hard_benign_fp_slice")

    if corpus_label != "unknown" and prompt_label != "unknown":
        prompt_score = score_value(row, "prompt_risk_score")
        if abs(ramp_score - prompt_score) >= 0.20:
            buckets.append("fusion_prompt_disagreement")

    if domain in undercovered_domains:
        buckets.append("undercovered_domain")

    if abs(ramp_score - threshold) <= uncertainty_window:
        buckets.append("uncertain_margin")

    return buckets


def bucket_priority(
    row: dict[str, Any],
    bucket: str,
    *,
    ramp_score: float,
    threshold: float,
) -> float:
    prompt_score = score_value(row, "prompt_risk_score")
    activation_score = score_value(row, "activation_probability")
    embedding_score = score_value(row, "embedding_prior_score")
    corpus_label = normalized_corpus_label(row)

    if bucket == "stability_severe_fn_slice":
        return (1.0 - ramp_score) + activation_score + embedding_score
    if bucket == "stability_hard_benign_fp_slice":
        return ramp_score + prompt_score + activation_score
    if bucket == "stability_ramp_fn_slice":
        return (1.0 - ramp_score) + activation_score
    if bucket == "stability_ramp_fp_slice":
        return ramp_score + prompt_score
    if bucket == "fusion_prompt_disagreement":
        return abs(ramp_score - prompt_score) + activation_score + embedding_score
    if bucket == "undercovered_domain":
        label_bonus = 0.25 if corpus_label == "unsafe" else 0.0
        return label_bonus + 1.0 - abs(ramp_score - threshold)
    if bucket == "uncertain_margin":
        return 1.0 - abs(ramp_score - threshold)
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
    stability: dict[str, Any],
    reviewed_source_ids: set[str],
    undercovered_domains: set[str],
    threshold: float,
    uncertainty_window: float,
    max_rows: int,
    max_per_bucket: int,
    max_per_stratum: int,
    max_error_slices: int,
) -> list[dict[str, Any]]:
    slice_map = stability_slice_map(stability, max_error_slices=max_error_slices)
    bucketed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_bucket_ids: set[tuple[str, str]] = set()
    for row in rows:
        row_id = str(row.get("id"))
        if row_id in reviewed_source_ids:
            continue
        if (
            row.get("prompt_risk_score") is None
            or row.get("embedding_prior_score") is None
            or row.get("activation_probability") is None
        ):
            continue
        ramp_score = approximate_ramp_score(row, stability)
        for bucket in candidate_buckets(
            row,
            stability=stability,
            slice_map=slice_map,
            undercovered_domains=undercovered_domains,
            threshold=threshold,
            uncertainty_window=uncertainty_window,
        ):
            key = (bucket, row_id)
            if key in seen_bucket_ids:
                continue
            seen_bucket_ids.add(key)
            candidate = dict(row)
            candidate["selection_bucket"] = bucket
            candidate["selection_priority"] = bucket_priority(
                row,
                bucket,
                ramp_score=ramp_score,
                threshold=threshold,
            )
            candidate["ramp_fusion_score"] = ramp_score
            bucketed[bucket].append(candidate)

    grouped: dict[tuple[str, str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for bucket in BUCKET_ORDER:
        rows_for_bucket = sorted(
            bucketed[bucket],
            key=lambda item: float(item.get("selection_priority", 0.0)),
            reverse=True,
        )[:max_per_bucket]
        for row in rows_for_bucket:
            grouped[stratum_key(row)].append(row)

    selected = []
    selected_ids = set()
    stratum_counts: Counter[tuple[str, str, str]] = Counter()
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
        "review_id": f"prompt_review_v0_3_{index:04d}",
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
        "ramp_fusion_score": row.get("ramp_fusion_score"),
        "activation_probability": row.get("activation_probability"),
        "embedding_prior_score": row.get("embedding_prior_score"),
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
        "ramp_fusion_score",
        "activation_probability",
        "embedding_prior_score",
        "source_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_summary(
    rows: list[dict[str, Any]],
    *,
    excluded_reviewed_rows: int,
    undercovered_domains: set[str],
) -> dict[str, Any]:
    return {
        "artifact_id": "ramp_prompt_review_batch_v0.3",
        "num_rows": len(rows),
        "excluded_existing_reviewed_rows": excluded_reviewed_rows,
        "undercovered_domains": sorted(undercovered_domains),
        "review_label_options": list(REVIEW_LABEL_OPTIONS),
        "by_selection_bucket": dict(Counter(str(row.get("selection_bucket")) for row in rows)),
        "by_source": dict(Counter(str(row.get("source")) for row in rows)),
        "by_domain": dict(Counter(str(row.get("domain")) for row in rows)),
        "instructions": (
            "This v0.3 batch is driven by repeated split-stability error slices. Prioritize "
            "stable severe misses, stable hard benign false positives, and undercovered domains."
        ),
    }


def main() -> None:
    args = parse_args()
    stability = load_stability(Path(args.stability_artifact))
    review_path = Path(args.review_csv)
    existing_reviewed_ids = reviewed_ids(review_path)
    domain_counts = reviewed_domain_counts(review_path)
    undercovered_domains = {
        domain
        for domain, count in domain_counts.items()
        if count < args.min_domain_binary_reviewed
    }
    rows = select_rows(
        load_jsonl(Path(args.feature_table)),
        stability=stability,
        reviewed_source_ids=existing_reviewed_ids,
        undercovered_domains=undercovered_domains,
        threshold=args.threshold,
        uncertainty_window=args.uncertainty_window,
        max_rows=args.max_rows,
        max_per_bucket=args.max_per_bucket,
        max_per_stratum=args.max_per_stratum,
        max_error_slices=args.max_error_slices,
    )
    review_rows = [review_row(row, index + 1) for index, row in enumerate(rows)]

    write_jsonl(Path(args.output_jsonl), review_rows)
    print(f"wrote v0.3 review JSONL to {args.output_jsonl}")
    if args.output_csv:
        write_csv(Path(args.output_csv), review_rows)
        print(f"wrote v0.3 review CSV to {args.output_csv}")

    summary = build_summary(
        review_rows,
        excluded_reviewed_rows=len(existing_reviewed_ids),
        undercovered_domains=undercovered_domains,
    )
    if args.summary_output:
        output_path = Path(args.summary_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"wrote v0.3 review summary to {args.summary_output}")

    print(f"review_rows={summary['num_rows']}")
    print(f"by_selection_bucket={summary['by_selection_bucket']}")


if __name__ == "__main__":
    main()
