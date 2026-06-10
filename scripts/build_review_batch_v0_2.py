#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from evaluate_ramp_harness import HIGH_SEVERITY_DOMAINS, is_hard_benign
from evaluate_reviewed_cumulative_signals import load_reviewed_rows
from evaluate_severity_floors import (
    DEFAULT_HIGH_SEVERITY_CATEGORIES,
    floor_event,
    load_calibration,
    normalize_category,
    raw_fusion_score,
    severity_categories,
)

REVIEW_LABEL_OPTIONS = (
    "safe",
    "unsafe",
    "controversial",
    "ambiguous_or_context_needed",
    "bad_benchmark_label",
)

BUCKET_ORDER = (
    "severe_miss_candidate",
    "hard_benign_fp_candidate",
    "controversial_high_severity",
    "ramp_uncertain",
    "qwen_safe_corpus_unsafe",
    "domain_coverage",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a harness-targeted v0.2 prompt review batch."
    )
    parser.add_argument(
        "--feature-table",
        required=True,
        help="Full joined prompt/embedding/activation feature table JSONL.",
    )
    parser.add_argument(
        "--review-csv",
        default=None,
        help="Existing reviewed CSV. Reviewed source_ids are excluded.",
    )
    parser.add_argument(
        "--calibration-artifact",
        required=True,
        help="Reviewed fusion calibration artifact JSON.",
    )
    parser.add_argument(
        "--calibration-objective",
        default="selected_by_best_f1",
        help="Calibration objective key to use for RAMP fusion scoring.",
    )
    parser.add_argument("--output-jsonl", required=True, help="Reviewer batch JSONL.")
    parser.add_argument("--output-csv", default=None, help="Optional reviewer batch CSV.")
    parser.add_argument("--summary-output", default=None, help="Optional summary JSON.")
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--max-per-bucket", type=int, default=125)
    parser.add_argument("--max-per-stratum", type=int, default=20)
    parser.add_argument("--floor", type=float, default=0.72)
    parser.add_argument("--uncertainty-window", type=float, default=0.08)
    parser.add_argument(
        "--high-severity-category",
        action="append",
        default=None,
        help="High-severity prompt category. Repeat to override defaults.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def reviewed_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return set(load_reviewed_rows(path))


def normalized_corpus_label(row: dict[str, Any]) -> str:
    value = str(row.get("label", "")).strip().lower()
    if value in {"unsafe", "harmful", "1", "true"}:
        return "unsafe"
    if value in {"safe", "benign", "0", "false"}:
        return "safe"
    return "unknown"


def normalized_prompt_label(row: dict[str, Any]) -> str:
    value = str(row.get("prompt_label", "")).strip().lower()
    if value in {"safe", "unsafe", "controversial"}:
        return value
    return "unknown"


def is_high_severity(
    row: dict[str, Any],
    high_severity_categories: set[str],
) -> bool:
    if severity_categories(row) & high_severity_categories:
        return True
    return str(row.get("domain", "")).strip().lower() in HIGH_SEVERITY_DOMAINS


def score_row(
    row: dict[str, Any],
    calibration: dict[str, Any],
    *,
    floor: float,
    high_severity_categories: set[str],
) -> dict[str, Any]:
    raw_score = raw_fusion_score(row, calibration)
    event = floor_event(
        row,
        raw_score,
        floor=floor,
        high_severity_categories=high_severity_categories,
    )
    return {
        **row,
        "calibrated_weighted_score": raw_score,
        "ramp_fusion_score": float(event["final_score"]),
        "severity_floor_applied": bool(event["floor_applied"]),
        "severity_floor_categories": event["matched_categories"],
    }


def bucket_priority(
    row: dict[str, Any],
    bucket: str,
    *,
    threshold: float,
) -> float:
    prompt_score = float(row.get("prompt_risk_score") or 0.0)
    embedding_score = float(row.get("embedding_prior_score") or 0.0)
    activation_score = float(row.get("activation_probability") or 0.0)
    ramp_score = float(row.get("ramp_fusion_score") or 0.0)
    corpus_label = normalized_corpus_label(row)
    prompt_label = normalized_prompt_label(row)

    if bucket == "severe_miss_candidate":
        return (1.0 - ramp_score) + (1.0 - prompt_score) + activation_score
    if bucket == "hard_benign_fp_candidate":
        return ramp_score + prompt_score + activation_score
    if bucket == "controversial_high_severity":
        return abs(ramp_score - threshold) * -1 + activation_score + embedding_score
    if bucket == "ramp_uncertain":
        return 1.0 - abs(ramp_score - threshold)
    if bucket == "qwen_safe_corpus_unsafe":
        return (1.0 - prompt_score) + activation_score + embedding_score
    if bucket == "domain_coverage":
        label_bonus = 0.5 if corpus_label == "unsafe" and prompt_label != "unsafe" else 0.0
        return label_bonus + abs(ramp_score - threshold) * -1
    return 0.0


def candidate_buckets(
    row: dict[str, Any],
    *,
    threshold: float,
    uncertainty_window: float,
    high_severity_categories: set[str],
) -> list[str]:
    corpus_label = normalized_corpus_label(row)
    prompt_label = normalized_prompt_label(row)
    ramp_score = float(row.get("ramp_fusion_score") or 0.0)
    buckets = []

    if (
        corpus_label == "unsafe"
        and ramp_score < threshold
        and is_high_severity(row, high_severity_categories)
    ):
        buckets.append("severe_miss_candidate")

    if corpus_label == "safe" and ramp_score >= threshold and is_hard_benign(row):
        buckets.append("hard_benign_fp_candidate")

    if prompt_label == "controversial" and is_high_severity(row, high_severity_categories):
        buckets.append("controversial_high_severity")

    if abs(ramp_score - threshold) <= uncertainty_window:
        buckets.append("ramp_uncertain")

    if corpus_label == "unsafe" and prompt_label == "safe":
        buckets.append("qwen_safe_corpus_unsafe")

    if is_high_severity(row, high_severity_categories) or str(row.get("domain")):
        buckets.append("domain_coverage")

    return buckets


def stratum_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("selection_bucket", "unknown")),
        str(row.get("source", "unknown")),
        str(row.get("domain", "unknown")),
    )


def select_review_rows(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    uncertainty_window: float,
    high_severity_categories: set[str],
    max_rows: int,
    max_per_bucket: int,
    max_per_stratum: int,
) -> list[dict[str, Any]]:
    bucketed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_bucket_ids: set[tuple[str, str]] = set()
    for row in rows:
        row_id = str(row.get("id"))
        for bucket in candidate_buckets(
            row,
            threshold=threshold,
            uncertainty_window=uncertainty_window,
            high_severity_categories=high_severity_categories,
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
                threshold=threshold,
            )
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
        "review_id": f"prompt_review_v0_2_{index:04d}",
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
        "qwen_confidence": row.get("prompt_confidence"),
        "qwen_harm_category": row.get("prompt_harm_category"),
        "selection_bucket": row.get("selection_bucket"),
        "selection_priority": row.get("selection_priority"),
        "source": row.get("source"),
        "domain": row.get("domain"),
        "subcluster_role": row.get("subcluster_role"),
        "subcluster_id": row.get("subcluster_id"),
        "embedding_prior_score": row.get("embedding_prior_score"),
        "activation_probability": row.get("activation_probability"),
        "calibrated_weighted_score": row.get("calibrated_weighted_score"),
        "ramp_fusion_score": row.get("ramp_fusion_score"),
        "severity_floor_applied": row.get("severity_floor_applied"),
        "severity_floor_categories": row.get("severity_floor_categories"),
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
        "calibrated_weighted_score",
        "activation_probability",
        "embedding_prior_score",
        "source_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_summary(rows: list[dict[str, Any]], *, excluded_reviewed_rows: int) -> dict[str, Any]:
    return {
        "artifact_id": "ramp_prompt_review_batch_v0.2",
        "num_rows": len(rows),
        "excluded_existing_reviewed_rows": excluded_reviewed_rows,
        "review_label_options": list(REVIEW_LABEL_OPTIONS),
        "by_selection_bucket": dict(Counter(str(row.get("selection_bucket")) for row in rows)),
        "by_source": dict(Counter(str(row.get("source")) for row in rows)),
        "by_domain": dict(Counter(str(row.get("domain")) for row in rows)),
        "instructions": (
            "This v0.2 batch is targeted from the RAMP evaluation harness. Prioritize "
            "reviewed_label and use reviewer_notes for why borderline benign examples are safe "
            "or why high-severity misses are unsafe."
        ),
    }


def main() -> None:
    args = parse_args()
    calibration = load_calibration(
        Path(args.calibration_artifact),
        args.calibration_objective,
    )
    high_severity_categories = {
        normalize_category(category)
        for category in (args.high_severity_category or DEFAULT_HIGH_SEVERITY_CATEGORIES)
    }
    existing_reviewed_ids = reviewed_ids(Path(args.review_csv) if args.review_csv else None)
    rows = [
        score_row(
            row,
            calibration,
            floor=args.floor,
            high_severity_categories=high_severity_categories,
        )
        for row in load_jsonl(Path(args.feature_table))
        if str(row.get("id")) not in existing_reviewed_ids
        and row.get("prompt_risk_score") is not None
        and row.get("embedding_prior_score") is not None
        and row.get("activation_probability") is not None
    ]
    selected = select_review_rows(
        rows,
        threshold=float(calibration["threshold"]),
        uncertainty_window=args.uncertainty_window,
        high_severity_categories=high_severity_categories,
        max_rows=args.max_rows,
        max_per_bucket=args.max_per_bucket,
        max_per_stratum=args.max_per_stratum,
    )
    review_rows = [review_row(row, index + 1) for index, row in enumerate(selected)]

    write_jsonl(Path(args.output_jsonl), review_rows)
    print(f"wrote v0.2 review JSONL to {args.output_jsonl}")

    if args.output_csv:
        write_csv(Path(args.output_csv), review_rows)
        print(f"wrote v0.2 review CSV to {args.output_csv}")

    summary = build_summary(
        review_rows,
        excluded_reviewed_rows=len(existing_reviewed_ids),
    )
    if args.summary_output:
        output_path = Path(args.summary_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"wrote v0.2 review summary to {args.summary_output}")

    print(f"review_rows={summary['num_rows']}")
    print(f"by_selection_bucket={summary['by_selection_bucket']}")


if __name__ == "__main__":
    main()
