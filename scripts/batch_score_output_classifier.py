#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from enum import Enum
from pathlib import Path
from typing import Any

from ramp.features import FeatureInput, KeywordOutputRiskFeature, Qwen3GuardOutputRiskFeature
from ramp.features.qwen3guard_prompt_risk import (
    parse_qwen3guard_output,
    qwen3guard_risk_score,
)
from ramp.risk_state import RiskState
from ramp.schemas.provenance import RuntimeProvenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-score generated responses with an output classifier."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Prompt/response eval JSONL or CSV with prompt_text and output_text.",
    )
    parser.add_argument("--output", required=True, help="Output-score JSONL.")
    parser.add_argument(
        "--provider",
        choices=["qwen3guard", "keyword"],
        default="qwen3guard",
        help="Output classifier provider.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Qwen3Guard model id or local path. Defaults to Qwen3Guard 0.6B.",
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map for Qwen3Guard.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def encode(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [encode(item) for item in value]
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        return {encode(key): encode(item) for key, item in value.items()}
    return value


def existing_eval_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row.get("eval_id") or row.get("id")
            if row_id is not None:
                ids.add(str(row_id))
    return ids


def load_records(path: Path, max_records: int | None) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as input_file:
            rows = list(csv.DictReader(input_file))
    else:
        rows = []
        with path.open(encoding="utf-8") as input_file:
            for line in input_file:
                if line.strip():
                    rows.append(json.loads(line))
    return rows[:max_records] if max_records is not None else rows


def make_feature(provider: str, model: str | None):
    if provider == "keyword":
        return KeywordOutputRiskFeature()
    return Qwen3GuardOutputRiskFeature(
        model_id=model or "Qwen/Qwen3Guard-Gen-0.6B",
    )


def output_text(record: dict[str, Any]) -> str:
    value = record.get("output_text")
    return value if isinstance(value, str) else ""


def prompt_text(record: dict[str, Any]) -> str:
    for key in ("prompt_text", "span_text", "prompt", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def source_id(record: dict[str, Any]) -> str | None:
    value = record.get("source_id") or record.get("id")
    return str(value) if value is not None else None


def eval_id(record: dict[str, Any]) -> str:
    value = record.get("eval_id") or source_id(record) or "unknown"
    return str(value)


class BatchedQwen3GuardOutputScorer:
    def __init__(
        self,
        *,
        model_id: str,
        max_new_tokens: int,
        device_map: str,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self._tokenizer = None
        self._model = None

    @property
    def version(self) -> str:
        return f"qwen3guard_output:{self.model_id}"

    def score(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._ensure_model_loaded()
        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:
            raise RuntimeError("Qwen3Guard model failed to load")

        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": output_text(record)}],
                tokenize=False,
            )
            for record in records
        ]
        model_inputs = tokenizer(texts, padding=True, return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        outputs = []
        input_lengths = model_inputs.input_ids.shape[1]
        for record, generated in zip(records, generated_ids, strict=True):
            raw_output = tokenizer.decode(
                generated[input_lengths:].tolist(),
                skip_special_tokens=True,
            )
            parsed = parse_qwen3guard_output(raw_output)
            risk_score, confidence = qwen3guard_risk_score(parsed.label, parsed.categories)
            harm_category = next(
                (category for category in parsed.categories if category != "None"),
                None,
            )
            outputs.append(
                {
                    "id": source_id(record),
                    "source_id": source_id(record),
                    "eval_id": eval_id(record),
                    "reviewed_label": record.get("reviewed_label"),
                    "output_reviewed_label": record.get("output_reviewed_label"),
                    "source": record.get("source"),
                    "domain": record.get("domain"),
                    "subcluster_role": record.get("subcluster_role"),
                    "subcluster_id": record.get("subcluster_id"),
                    "prompt_text": prompt_text(record),
                    "output_text": output_text(record),
                    "output_risk_score": risk_score,
                    "output_confidence": confidence,
                    "output_label": parsed.label or "parse_error",
                    "output_harm_category": harm_category,
                    "output_classifier_version": self.version,
                    "output_classifier_metadata": {
                        "backend": "qwen3guard",
                        "model_id": self.model_id,
                        "raw_output": parsed.raw_output,
                        "safety_label": parsed.label,
                        "categories": list(parsed.categories),
                        "scored_text": "output",
                    },
                }
            )
        return outputs

    def _ensure_model_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map=self.device_map,
        )


def score_record(record: dict[str, Any], feature) -> dict[str, Any]:
    row_id = eval_id(record)
    state = RiskState(
        request_id=f"req_output_batch_{row_id}",
        session_id="sess_output_batch",
        provenance=RuntimeProvenance(
            request_id=f"req_output_batch_{row_id}",
            session_id="sess_output_batch",
        ),
    )
    result = feature.extract(
        FeatureInput(prompt=prompt_text(record), output=output_text(record)),
        state,
    )
    return {
        "id": source_id(record),
        "source_id": source_id(record),
        "eval_id": eval_id(record),
        "reviewed_label": record.get("reviewed_label"),
        "output_reviewed_label": record.get("output_reviewed_label"),
        "source": record.get("source"),
        "domain": record.get("domain"),
        "subcluster_role": record.get("subcluster_role"),
        "subcluster_id": record.get("subcluster_id"),
        "prompt_text": prompt_text(record),
        "output_text": output_text(record),
        "output_risk_score": result.risk_score,
        "output_confidence": result.confidence,
        "output_label": result.label,
        "output_harm_category": result.harm_category,
        "output_classifier_version": result.version,
        "output_classifier_metadata": encode(result.metadata),
    }


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = existing_eval_ids(output_path) if args.resume else set()
    records = load_records(Path(args.input), args.max_records)

    qwen_scorer = (
        BatchedQwen3GuardOutputScorer(
            model_id=args.model or "Qwen/Qwen3Guard-Gen-0.6B",
            max_new_tokens=args.max_new_tokens,
            device_map=args.device_map,
        )
        if args.provider == "qwen3guard"
        else None
    )
    feature = make_feature(args.provider, args.model) if qwen_scorer is None else None

    written = 0
    skipped_existing = 0
    skipped_missing_output = 0
    batch: list[dict[str, Any]] = []
    mode = "a" if args.resume else "w"

    def flush_batch(output_file) -> None:
        nonlocal written
        if not batch:
            return
        scored_rows = (
            qwen_scorer.score(batch)
            if qwen_scorer is not None
            else [score_record(record, feature) for record in batch]
        )
        for scored in scored_rows:
            output_file.write(json.dumps(scored, separators=(",", ":")) + "\n")
            written += 1
            if args.progress_every and written % args.progress_every == 0:
                print(
                    "wrote "
                    f"{written} output scores; "
                    f"skipped_existing={skipped_existing}; "
                    f"skipped_missing_output={skipped_missing_output}"
                )
        batch.clear()

    with output_path.open(mode, encoding="utf-8") as output_file:
        for record in records:
            if eval_id(record) in seen:
                skipped_existing += 1
                continue
            if not output_text(record).strip():
                skipped_missing_output += 1
                continue
            batch.append(record)
            if len(batch) >= max(1, args.batch_size):
                flush_batch(output_file)
        flush_batch(output_file)

    print(
        f"wrote {written} output scores to {output_path}; "
        f"skipped_existing={skipped_existing}; "
        f"skipped_missing_output={skipped_missing_output}"
    )


if __name__ == "__main__":
    main()
