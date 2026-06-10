#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

REVIEW_LABEL_OPTIONS = (
    "safe",
    "unsafe",
    "controversial",
    "ambiguous_or_context_needed",
    "bad_benchmark_label",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reviewer-ready prompt-label audit batch."
    )
    parser.add_argument("--candidates", required=True, help="Audit candidate JSONL.")
    parser.add_argument("--output-jsonl", required=True, help="Reviewer batch JSONL.")
    parser.add_argument("--output-csv", default=None, help="Optional reviewer batch CSV.")
    parser.add_argument("--summary-output", default=None, help="Optional summary JSON.")
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument(
        "--max-per-stratum",
        type=int,
        default=50,
        help="Maximum rows per audit_bucket/source/domain stratum.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stratum_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("audit_bucket", "unknown")),
        str(row.get("source", "unknown")),
        str(row.get("domain", "unknown")),
    )


def select_round_robin(
    rows: list[dict[str, Any]],
    *,
    max_rows: int,
    max_per_stratum: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for row in sorted(rows, key=lambda item: float(item.get("audit_priority", 0.0)), reverse=True):
        grouped[stratum_key(row)].append(row)

    selected = []
    stratum_counts: Counter[tuple[str, str, str]] = Counter()
    active_keys = deque(sorted(grouped))
    while active_keys and len(selected) < max_rows:
        key = active_keys.popleft()
        group = grouped[key]
        if not group or stratum_counts[key] >= max_per_stratum:
            continue
        selected.append(group.popleft())
        stratum_counts[key] += 1
        if group and stratum_counts[key] < max_per_stratum:
            active_keys.append(key)
    return selected


def review_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    prompt_text = str(row.get("span_text", ""))
    return {
        "review_id": f"prompt_review_v0_1_{index:04d}",
        "source_id": row.get("id"),
        "review_status": "unreviewed",
        "reviewed_label": "",
        "review_label_options": list(REVIEW_LABEL_OPTIONS),
        "label_issue_type": "",
        "reviewer_notes": "",
        "reviewed_by": "",
        "reviewed_at": "",
        "prompt_text": prompt_text,
        "corpus_label": row.get("label"),
        "qwen_label": row.get("prompt_label"),
        "qwen_risk_score": row.get("prompt_risk_score"),
        "qwen_confidence": row.get("prompt_confidence"),
        "audit_bucket": row.get("audit_bucket"),
        "audit_priority": row.get("audit_priority"),
        "source": row.get("source"),
        "domain": row.get("domain"),
        "subcluster_role": row.get("subcluster_role"),
        "subcluster_id": row.get("subcluster_id"),
        "embedding_margin": row.get("embedding_margin"),
        "activation_probability": row.get("activation_probability"),
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
        "audit_bucket",
        "source",
        "domain",
        "subcluster_id",
        "source_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_rows": len(rows),
        "review_label_options": list(REVIEW_LABEL_OPTIONS),
        "by_audit_bucket": dict(Counter(str(row.get("audit_bucket")) for row in rows)),
        "by_source": dict(Counter(str(row.get("source")) for row in rows)),
        "by_domain": dict(Counter(str(row.get("domain")) for row in rows)),
        "instructions": (
            "Fill reviewed_label with one review_label_options value. Use label_issue_type "
            "for benchmark/corpus issues such as idiom_false_positive, definition_request, "
            "context_missing, or true_policy_violation."
        ),
    }


def main() -> None:
    args = parse_args()
    candidates = load_jsonl(Path(args.candidates))
    selected = select_round_robin(
        candidates,
        max_rows=args.max_rows,
        max_per_stratum=args.max_per_stratum,
    )
    review_rows = [review_row(row, index + 1) for index, row in enumerate(selected)]

    write_jsonl(Path(args.output_jsonl), review_rows)
    print(f"wrote review JSONL to {args.output_jsonl}")

    if args.output_csv:
        write_csv(Path(args.output_csv), review_rows)
        print(f"wrote review CSV to {args.output_csv}")

    summary = build_summary(review_rows)
    if args.summary_output:
        output_path = Path(args.summary_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"wrote review summary to {args.summary_output}")

    print(f"review_rows={summary['num_rows']}")


if __name__ == "__main__":
    main()
