#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TAXONOMY_FIELDS = (
    "domain",
    "subcluster_role",
    "subcluster_id",
    "label",
    "harm_severity",
    "actionability",
    "intent_confidence",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply explicit taxonomy correction overlays to JSONL rows by id."
    )
    parser.add_argument("--input", required=True, help="Input JSONL with id fields.")
    parser.add_argument("--corrections", required=True, help="Taxonomy corrections JSON.")
    parser.add_argument("--output", required=True, help="Corrected output JSONL.")
    parser.add_argument("--summary-output", default=None, help="Optional summary JSON.")
    return parser.parse_args()


def load_corrections(path: Path) -> dict[str, dict[str, Any]]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    corrections: dict[str, dict[str, Any]] = {}
    for correction in artifact.get("corrections", []):
        source_id = str(correction["source_id"])
        corrections[source_id] = correction
    return corrections


def apply_correction(row: dict[str, Any], correction: dict[str, Any]) -> dict[str, Any]:
    corrected = correction["corrected"]
    output = dict(row)
    previous = {field: output.get(field) for field in TAXONOMY_FIELDS if field in output}
    for field in TAXONOMY_FIELDS:
        if field in corrected:
            output[field] = corrected[field]
    output["taxonomy_correction_applied"] = True
    output["taxonomy_correction_id"] = correction["source_id"]
    output["taxonomy_original"] = correction.get("original", previous)
    output["taxonomy_correction_reason"] = correction.get("reason")
    return output


def main() -> None:
    args = parse_args()
    corrections = load_corrections(Path(args.corrections))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    corrected_rows = 0

    with Path(args.input).open(encoding="utf-8") as input_file, output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            total_rows += 1
            row_id = str(row.get("id"))
            if row_id in corrections:
                row = apply_correction(row, corrections[row_id])
                corrected_rows += 1
            output_file.write(json.dumps(row, separators=(",", ":")) + "\n")

    summary = {
        "input": args.input,
        "output": args.output,
        "corrections": args.corrections,
        "total_rows": total_rows,
        "corrected_rows": corrected_rows,
        "corrected_ids": sorted(corrections),
    }
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"wrote {total_rows} rows to {output_path}")
    print(f"corrected_rows={corrected_rows}")


if __name__ == "__main__":
    main()
