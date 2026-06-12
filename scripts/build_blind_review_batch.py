#!/usr/bin/env python
"""Build a blind review batch by stratified random sampling from the benchmark corpus.

Blind requirements (docs/fragility-study-design.md):
- random sample stratified by (source, domain) to match the corpus distribution,
  excluding every row id that appears in any prior review CSV
- selection is deterministic given --seed and is committed (manifest with ids and
  checksum) BEFORE labeling begins
- the reviewer CSV contains prompt text and source metadata only: no model scores,
  no bucket assignments, no severity hints
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a blind review batch from the benchmark corpus feature table."
    )
    parser.add_argument("--feature-table", required=True)
    parser.add_argument(
        "--exclude-review-csv",
        action="append",
        default=[],
        help="Prior review CSV whose source_id rows are excluded. Repeatable.",
    )
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--seed", default="ramp_blind_batch_v0_1")
    parser.add_argument("--batch-id", default="ramp_blind_review_batch_v0_1")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-manifest", required=True)
    return parser.parse_args()


def load_corpus_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id") is None or not str(row.get("span_text", "")).strip():
                continue
            rows.append(
                {
                    "id": str(row["id"]),
                    "span_text": str(row["span_text"]),
                    "source": str(row.get("source", "unknown")),
                    "domain": str(row.get("domain", "unknown")),
                }
            )
    return rows


def load_excluded_ids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8", newline="") as input_file:
            for row in csv.DictReader(input_file):
                row_id = (row.get("source_id") or row.get("id") or "").strip()
                if row_id:
                    excluded.add(row_id)
    return excluded


def sample_key(seed: str, row_id: str) -> str:
    return hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest()


def allocate_cells(
    cell_sizes: dict[tuple[str, str], int],
    *,
    max_rows: int,
) -> dict[tuple[str, str], int]:
    total = sum(cell_sizes.values())
    if total <= max_rows:
        return dict(cell_sizes)
    exact = {cell: max_rows * size / total for cell, size in cell_sizes.items()}
    allocation = {cell: int(value) for cell, value in exact.items()}
    remaining = max_rows - sum(allocation.values())
    remainders = sorted(
        cell_sizes,
        key=lambda cell: (exact[cell] - allocation[cell], str(cell)),
        reverse=True,
    )
    for cell in remainders:
        if remaining <= 0:
            break
        if allocation[cell] < cell_sizes[cell]:
            allocation[cell] += 1
            remaining -= 1
    return allocation


def sample_batch(
    rows: list[dict[str, Any]],
    *,
    excluded_ids: set[str],
    max_rows: int,
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = [row for row in rows if row["id"] not in excluded_ids]
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        cells[(row["source"], row["domain"])].append(row)
    allocation = allocate_cells(
        {cell: len(cell_rows) for cell, cell_rows in cells.items()},
        max_rows=max_rows,
    )
    sampled: list[dict[str, Any]] = []
    for cell in sorted(cells):
        cell_rows = sorted(cells[cell], key=lambda row: sample_key(seed, row["id"]))
        sampled.extend(cell_rows[: allocation[cell]])
    sampled.sort(key=lambda row: sample_key(seed, row["id"]))
    summary = {
        "corpus_rows": len(rows),
        "excluded_ids": len(excluded_ids),
        "eligible_rows": len(eligible),
        "sampled_rows": len(sampled),
        "allocation": {
            f"{source}/{domain}": allocation[(source, domain)]
            for source, domain in sorted(allocation)
        },
    }
    return sampled, summary


def main() -> None:
    args = parse_args()
    rows = load_corpus_rows(Path(args.feature_table))
    excluded_ids = load_excluded_ids([Path(path) for path in args.exclude_review_csv])
    sampled, summary = sample_batch(
        rows,
        excluded_ids=excluded_ids,
        max_rows=args.max_rows,
        seed=args.seed,
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "review_id",
        "source_id",
        "source",
        "prompt_text",
        "review_status",
        "reviewed_label",
        "label_issue_type",
        "notes",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(sampled, start=1):
            writer.writerow(
                {
                    "review_id": f"{args.batch_id}_{index:04d}",
                    "source_id": row["id"],
                    "source": row["source"],
                    "prompt_text": row["span_text"],
                    "review_status": "",
                    "reviewed_label": "",
                    "label_issue_type": "",
                    "notes": "",
                }
            )

    sampled_ids = [row["id"] for row in sampled]
    manifest = {
        "batch_id": args.batch_id,
        "seed": args.seed,
        "max_rows": args.max_rows,
        "feature_table": args.feature_table,
        "exclude_review_csvs": list(args.exclude_review_csv),
        "summary": summary,
        "sampled_ids_sha256": hashlib.sha256("\n".join(sampled_ids).encode()).hexdigest(),
        "sampled": [
            {"id": row["id"], "source": row["source"], "domain": row["domain"]} for row in sampled
        ],
        "blinding": (
            "Reviewer CSV contains prompt text and source metadata only; no model "
            "scores, buckets, domains, or severity hints. Selection is deterministic "
            "from the seed and committed before labeling."
        ),
    }
    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"wrote blind review CSV to {output_csv}")
    print(f"wrote blind batch manifest to {output_manifest}")
    print(
        f"sampled {summary['sampled_rows']} of {summary['eligible_rows']} eligible rows "
        f"({summary['excluded_ids']} previously reviewed ids excluded)"
    )


if __name__ == "__main__":
    main()
