#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit corpus labels against prompt-classifier outputs."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--feature-table",
        help="Joined feature table JSONL from evaluate_cumulative_internal_signals.py.",
    )
    input_group.add_argument(
        "--corpus",
        help="Corpus JSONL. Requires --prompt-scores.",
    )
    parser.add_argument(
        "--prompt-scores",
        default=None,
        help="Prompt score JSONL. Required when --corpus is used.",
    )
    parser.add_argument("--output-json", required=True, help="Output audit report JSON.")
    parser.add_argument("--output-md", default=None, help="Optional Markdown summary.")
    parser.add_argument(
        "--suspect-rows",
        default=None,
        help="Optional JSONL of all rows that need review.",
    )
    parser.add_argument(
        "--audit-candidates",
        default=None,
        help="Optional stratified JSONL candidate set for manual review.",
    )
    parser.add_argument(
        "--max-candidates-per-bucket",
        type=int,
        default=100,
        help="Maximum candidates per audit bucket/source/domain stratum.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in load_jsonl(path):
        row_id = row.get("id")
        if row_id is not None:
            rows[str(row_id)] = row
    return rows


def normalized_corpus_label(label: Any) -> str:
    value = str(label).strip().lower()
    if value in {"unsafe", "harmful", "1", "true"}:
        return "unsafe"
    if value in {"safe", "benign", "0", "false"}:
        return "safe"
    return "unknown"


def normalized_prompt_label(label: Any) -> str:
    value = str(label).strip().lower()
    if value == "unsafe":
        return "unsafe"
    if value == "safe":
        return "safe"
    if value == "controversial":
        return "controversial"
    if value in {"parse_error", "", "none", "null"}:
        return "unknown"
    return value


def disagreement_bucket(row: dict[str, Any]) -> str:
    corpus_label = normalized_corpus_label(row.get("label"))
    prompt_label = normalized_prompt_label(row.get("prompt_label"))
    if corpus_label == "unknown" or prompt_label == "unknown":
        return "missing_or_unknown_label"
    if prompt_label == "controversial":
        return f"corpus_{corpus_label}_classifier_controversial"
    if corpus_label == prompt_label:
        return f"corpus_{corpus_label}_classifier_agree"
    return f"corpus_{corpus_label}_classifier_{prompt_label}"


def review_priority(row: dict[str, Any]) -> float:
    bucket = disagreement_bucket(row)
    priority = 0.0
    if bucket in {"corpus_safe_classifier_unsafe", "corpus_unsafe_classifier_safe"}:
        priority += 10.0
    elif "controversial" in bucket:
        priority += 4.0
    elif bucket == "missing_or_unknown_label":
        priority += 3.0

    prompt_score = row.get("prompt_risk_score")
    if prompt_score is not None:
        corpus_label = normalized_corpus_label(row.get("label"))
        score = float(prompt_score)
        if corpus_label == "safe":
            priority += score
        elif corpus_label == "unsafe":
            priority += 1.0 - score

    activation_probability = row.get("activation_probability")
    if activation_probability is not None:
        corpus_label = normalized_corpus_label(row.get("label"))
        probability = float(activation_probability)
        if corpus_label == "safe":
            priority += probability
        elif corpus_label == "unsafe":
            priority += 1.0 - probability

    embedding_margin = row.get("embedding_margin")
    if embedding_margin is not None:
        margin = float(embedding_margin)
        corpus_label = normalized_corpus_label(row.get("label"))
        if corpus_label == "safe" and margin > 0:
            priority += min(1.0, margin)
        elif corpus_label == "unsafe" and margin < 0:
            priority += min(1.0, abs(margin))

    return priority


def join_corpus_and_prompt_scores(
    corpus_path: Path,
    prompt_scores_path: Path,
) -> list[dict[str, Any]]:
    corpus_rows = load_jsonl_by_id(corpus_path)
    prompt_rows = load_jsonl_by_id(prompt_scores_path)
    joined_ids = sorted(set(corpus_rows) & set(prompt_rows))
    rows = []
    for row_id in joined_ids:
        corpus = corpus_rows[row_id]
        prompt = prompt_rows[row_id]
        rows.append(
            {
                "id": row_id,
                "label": corpus.get("label"),
                "source": corpus.get("source"),
                "domain": corpus.get("domain"),
                "subcluster_role": corpus.get("subcluster_role"),
                "subcluster_id": corpus.get("subcluster_id"),
                "span_text": corpus.get("span_text", prompt.get("span_text")),
                "prompt_risk_score": prompt.get("prompt_risk_score"),
                "prompt_confidence": prompt.get("prompt_confidence"),
                "prompt_label": prompt.get("prompt_label"),
                "prompt_harm_category": prompt.get("prompt_harm_category"),
                "prompt_classifier_version": prompt.get("prompt_classifier_version"),
            }
        )
    return rows


def summarize_counter(counter: Counter[tuple[str, ...] | str]) -> dict[str, int]:
    return {
        " | ".join(key) if isinstance(key, tuple) else str(key): count
        for key, count in counter.most_common()
    }


def build_report(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    label_cross_tab: Counter[tuple[str, str]] = Counter()
    bucket_counts: Counter[str] = Counter()
    bucket_by_source: Counter[tuple[str, str]] = Counter()
    bucket_by_domain: Counter[tuple[str, str]] = Counter()
    prompt_score_by_bucket: dict[str, list[float]] = defaultdict(list)
    suspect_rows = []

    for row in rows:
        corpus_label = normalized_corpus_label(row.get("label"))
        prompt_label = normalized_prompt_label(row.get("prompt_label"))
        bucket = disagreement_bucket(row)
        source = str(row.get("source", "unknown"))
        domain = str(row.get("domain", "unknown"))

        label_cross_tab[(corpus_label, prompt_label)] += 1
        bucket_counts[bucket] += 1
        bucket_by_source[(bucket, source)] += 1
        bucket_by_domain[(bucket, domain)] += 1
        if row.get("prompt_risk_score") is not None:
            prompt_score_by_bucket[bucket].append(float(row["prompt_risk_score"]))

        if not bucket.endswith("_classifier_agree"):
            suspect = dict(row)
            suspect["audit_bucket"] = bucket
            suspect["audit_priority"] = review_priority(row)
            suspect["review_status"] = "needs_review"
            suspect_rows.append(suspect)

    suspect_rows.sort(key=lambda row: float(row["audit_priority"]), reverse=True)

    report = {
        "num_rows": len(rows),
        "num_suspect_rows": len(suspect_rows),
        "label_cross_tab": summarize_counter(label_cross_tab),
        "audit_buckets": summarize_counter(bucket_counts),
        "audit_bucket_by_source": summarize_counter(bucket_by_source),
        "audit_bucket_by_domain": summarize_counter(bucket_by_domain),
        "prompt_score_by_bucket": {
            bucket: distribution(values)
            for bucket, values in sorted(prompt_score_by_bucket.items())
        },
        "top_suspect_rows": [compact_row(row) for row in suspect_rows[:25]],
        "interpretation": (
            "This audit measures disagreement between corpus labels and prompt-classifier labels. "
            "It should be used to select rows for manual review, not to declare either side "
            "correct by default."
        ),
    }
    return report, suspect_rows


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0.0, "p50": 0.0, "max": 0.0}
    sorted_values = sorted(values)
    return {
        "count": len(values),
        "min": sorted_values[0],
        "p50": sorted_values[len(sorted_values) // 2],
        "max": sorted_values[-1],
    }


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "id",
        "audit_bucket",
        "audit_priority",
        "label",
        "prompt_label",
        "prompt_risk_score",
        "source",
        "domain",
        "subcluster_id",
        "span_text",
    ]
    return {key: row.get(key) for key in keys}


def stratified_candidates(
    suspect_rows: list[dict[str, Any]],
    max_candidates_per_bucket: int,
) -> list[dict[str, Any]]:
    selected = []
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in suspect_rows:
        key = (
            str(row.get("audit_bucket", "unknown")),
            str(row.get("source", "unknown")),
            str(row.get("domain", "unknown")),
        )
        if counts[key] >= max_candidates_per_bucket:
            continue
        counts[key] += 1
        selected.append(row)
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, separators=(",", ":")) + "\n")


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Prompt Classifier Label Audit",
        "",
        report["interpretation"],
        "",
        f"Rows: {report['num_rows']}",
        f"Suspect rows: {report['num_suspect_rows']}",
        "",
        "## Label Cross-Tab",
        "",
        "| Corpus label | Prompt label | Rows |",
        "| --- | --- | ---: |",
    ]
    for key, count in report["label_cross_tab"].items():
        corpus_label, prompt_label = key.split(" | ", 1)
        lines.append(f"| `{corpus_label}` | `{prompt_label}` | {count} |")
    lines.extend(["", "## Audit Buckets", "", "| Bucket | Rows |", "| --- | ---: |"])
    for bucket, count in report["audit_buckets"].items():
        lines.append(f"| `{bucket}` | {count} |")
    lines.extend(
        [
            "",
            "## Top Suspect Rows",
            "",
            "| Bucket | Source | Domain | Corpus | Prompt | Score | Text |",
            "| --- | --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for row in report["top_suspect_rows"]:
        text = str(row.get("span_text", "")).replace("|", "\\|")[:140]
        score = row.get("prompt_risk_score")
        score_text = "" if score is None else f"{float(score):.2f}"
        lines.append(
            f"| `{row.get('audit_bucket')}` | `{row.get('source')}` | "
            f"`{row.get('domain')}` | `{row.get('label')}` | `{row.get('prompt_label')}` | "
            f"{score_text} | {text} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.corpus and not args.prompt_scores:
        raise ValueError("--prompt-scores is required when --corpus is used")

    if args.feature_table:
        rows = load_jsonl(Path(args.feature_table))
        input_paths = {"feature_table": args.feature_table}
    else:
        rows = join_corpus_and_prompt_scores(Path(args.corpus), Path(args.prompt_scores))
        input_paths = {"corpus": args.corpus, "prompt_scores": args.prompt_scores}

    report, suspect_rows = build_report(rows)
    report["input_paths"] = input_paths

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote prompt-label audit report to {output_json}")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown_report(report) + "\n", encoding="utf-8")
        print(f"wrote prompt-label audit markdown to {output_md}")

    if args.suspect_rows:
        write_jsonl(Path(args.suspect_rows), suspect_rows)
        print(f"wrote suspect rows to {args.suspect_rows}")

    if args.audit_candidates:
        candidates = stratified_candidates(
            suspect_rows,
            max_candidates_per_bucket=args.max_candidates_per_bucket,
        )
        write_jsonl(Path(args.audit_candidates), candidates)
        print(f"wrote audit candidates to {args.audit_candidates}")

    print(f"rows={report['num_rows']} suspect_rows={report['num_suspect_rows']}")


if __name__ == "__main__":
    main()
