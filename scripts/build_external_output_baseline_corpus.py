#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a native response-labeled external output-classifier corpus."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--include-source",
        action="append",
        choices=["wildguardmix", "beavertails"],
        default=None,
        help="Source to include. Repeat to restrict output.",
    )
    parser.add_argument("--max-per-source", type=int, default=10_000_000)
    return parser.parse_args()


def clean(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def wildguardmix_response_label(value: Any) -> str | None:
    label = str(value or "").strip().lower()
    if label in {"harmful", "unsafe"}:
        return "unsafe"
    if label in {"unharmful", "harmless", "safe"}:
        return "safe"
    return None


def add_wildguardmix(rows: list[dict[str, Any]], max_per_source: int) -> None:
    from datasets import load_dataset

    dataset = load_dataset("bogdanminko/wildguardmix-cleaned", split="train")
    before = len(rows)
    for idx, row in enumerate(dataset):
        if idx >= max_per_source:
            break
        prompt_text = clean(row.get("prompt"))
        output_text = clean(row.get("response"))
        label = wildguardmix_response_label(row.get("response_harm_label"))
        if not output_text or label is None:
            continue
        rows.append(
            {
                "id": f"external_output_wildguardmix_{idx:07d}",
                "source_id": f"train:{idx}",
                "eval_id": f"external_output_wildguardmix_{idx:07d}",
                "source": "wildguardmix",
                "prompt_text": prompt_text,
                "output_text": output_text,
                "output_reviewed_label": label,
                "label": label,
                "response_refusal_label": row.get("response_refusal_label"),
                "native_response_label": row.get("response_harm_label"),
                "native_prompt_label": row.get("prompt_harm_label"),
                "source_record_hash": sha(prompt_text + "\n\n" + output_text),
                "external_label_source": "wildguardmix:response_harm_label",
            }
        )
    print(f"wildguardmix: added {len(rows) - before}")


def add_beavertails(rows: list[dict[str, Any]], max_per_source: int) -> None:
    from datasets import load_dataset

    dataset = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
    before = len(rows)
    for idx, row in enumerate(dataset):
        if idx >= max_per_source:
            break
        prompt_text = clean(row.get("prompt"))
        output_text = clean(row.get("response"))
        is_safe = row.get("is_safe")
        if not output_text or is_safe is None:
            continue
        label = "safe" if is_safe is True else "unsafe"
        rows.append(
            {
                "id": f"external_output_beavertails_{idx:07d}",
                "source_id": f"30k_train:{idx}",
                "eval_id": f"external_output_beavertails_{idx:07d}",
                "source": "beavertails",
                "prompt_text": prompt_text,
                "output_text": output_text,
                "output_reviewed_label": label,
                "label": label,
                "native_category": row.get("category"),
                "source_record_hash": sha(prompt_text + "\n\n" + output_text),
                "external_label_source": "beavertails:is_safe",
            }
        )
    print(f"beavertails: added {len(rows) - before}")


def main() -> None:
    args = parse_args()
    include_sources = set(args.include_source or ["wildguardmix", "beavertails"])
    rows: list[dict[str, Any]] = []
    if "wildguardmix" in include_sources:
        add_wildguardmix(rows, args.max_per_source)
    if "beavertails" in include_sources:
        add_beavertails(rows, args.max_per_source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"wrote {len(rows)} external output rows to {output}")


if __name__ == "__main__":
    main()
