#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a corrected external prompt-baseline corpus from an existing RAMP "
            "benchmark corpus while preserving row ids for already-generated scores."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--include-source",
        action="append",
        default=None,
        help="Optional source to include. Repeat to restrict the output.",
    )
    return parser.parse_args()


def load_wildguardmix_labels() -> dict[str, str]:
    from datasets import load_dataset

    rows = load_dataset("bogdanminko/wildguardmix-cleaned", split="train")
    labels = {}
    for idx, row in enumerate(rows):
        value = str(row.get("prompt_harm_label", "")).strip().lower()
        if value in {"harmful", "unsafe"}:
            labels[f"train:{idx}"] = "unsafe"
        elif value in {"unharmful", "harmless", "safe"}:
            labels[f"train:{idx}"] = "safe"
    return labels


def load_beavertails_labels() -> dict[str, str]:
    from datasets import load_dataset

    labels = {}
    for split in ("30k_train", "train", "test"):
        try:
            rows = load_dataset("PKU-Alignment/BeaverTails", split=split)
        except Exception:
            continue
        for idx, row in enumerate(rows):
            value = row.get("is_safe")
            if value is True:
                labels[f"{split}:{idx}"] = "safe"
            elif value is False:
                labels[f"{split}:{idx}"] = "unsafe"
        if labels:
            return labels
    return labels


def corrected_label(
    row: dict[str, Any],
    *,
    wildguardmix_labels: dict[str, str],
    beavertails_labels: dict[str, str],
) -> str | None:
    source = str(row.get("source", "")).strip().lower()
    source_record_id = str(row.get("source_record_id", "")).strip()
    if source == "wildguardmix":
        return wildguardmix_labels.get(source_record_id)
    if source == "beavertails":
        return beavertails_labels.get(source_record_id)
    if source == "harmbench":
        return "unsafe"
    if source == "do_not_answer":
        return "safe"
    label = str(row.get("label", "")).strip().lower()
    if label in {"safe", "unsafe"}:
        return label
    return None


def main() -> None:
    args = parse_args()
    include_sources = (
        {source.strip().lower() for source in args.include_source}
        if args.include_source
        else None
    )
    wildguardmix_labels = load_wildguardmix_labels()
    beavertails_labels = load_beavertails_labels()
    output_rows = []
    with Path(args.input).open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            source = str(row.get("source", "")).strip().lower()
            if include_sources is not None and source not in include_sources:
                continue
            label = corrected_label(
                row,
                wildguardmix_labels=wildguardmix_labels,
                beavertails_labels=beavertails_labels,
            )
            if label is None:
                continue
            corrected = dict(row)
            corrected["label"] = label
            corrected["external_label_source"] = f"{source}:native_prompt_label"
            output_rows.append(corrected)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as output_file:
        for row in output_rows:
            output_file.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"wrote {len(output_rows)} corrected external rows to {output}")


if __name__ == "__main__":
    main()
