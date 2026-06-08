#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from enum import Enum
from pathlib import Path
from typing import Any

from ramp.features import FeatureInput, KeywordPromptRiskFeature, Qwen3GuardPromptRiskFeature
from ramp.features.qwen3guard_prompt_risk import (
    parse_qwen3guard_output,
    qwen3guard_risk_score,
)
from ramp.risk_state import RiskState
from ramp.schemas.provenance import RuntimeProvenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-score corpus rows with a prompt classifier."
    )
    parser.add_argument("--corpus", required=True, help="Corpus JSONL with span_text/prompt rows.")
    parser.add_argument("--output", required=True, help="Output prompt-score JSONL.")
    parser.add_argument(
        "--provider",
        choices=["qwen3guard", "keyword"],
        default="qwen3guard",
        help="Prompt classifier provider.",
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


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row.get("id")
            if row_id is not None:
                ids.add(str(row_id))
    return ids


def make_feature(provider: str, model: str | None):
    if provider == "keyword":
        return KeywordPromptRiskFeature()
    return Qwen3GuardPromptRiskFeature(
        model_id=model or "Qwen/Qwen3Guard-Gen-0.6B",
    )


class BatchedQwen3GuardScorer:
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
        return f"qwen3guard:{self.model_id}"

    def score(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._ensure_model_loaded()
        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:
            raise RuntimeError("Qwen3Guard model failed to load")

        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text(record)}],
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
                    "id": record.get("id"),
                    "label": record.get("label"),
                    "source": record.get("source"),
                    "domain": record.get("domain"),
                    "subcluster_role": record.get("subcluster_role"),
                    "subcluster_id": record.get("subcluster_id"),
                    "span_text": prompt_text(record),
                    "prompt_risk_score": risk_score,
                    "prompt_confidence": confidence,
                    "prompt_label": parsed.label or "parse_error",
                    "prompt_harm_category": harm_category,
                    "prompt_classifier_version": self.version,
                    "prompt_classifier_metadata": {
                        "backend": "qwen3guard",
                        "model_id": self.model_id,
                        "raw_output": parsed.raw_output,
                        "safety_label": parsed.label,
                        "categories": list(parsed.categories),
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


def prompt_text(record: dict[str, Any]) -> str:
    for key in ("span_text", "prompt", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def score_record(record: dict[str, Any], feature) -> dict[str, Any]:
    row_id = str(record.get("id", "unknown"))
    state = RiskState(
        request_id=f"req_prompt_batch_{row_id}",
        session_id="sess_prompt_batch",
        provenance=RuntimeProvenance(
            request_id=f"req_prompt_batch_{row_id}",
            session_id="sess_prompt_batch",
        ),
    )
    result = feature.extract(FeatureInput(prompt=prompt_text(record)), state)
    return {
        "id": record.get("id"),
        "label": record.get("label"),
        "source": record.get("source"),
        "domain": record.get("domain"),
        "subcluster_role": record.get("subcluster_role"),
        "subcluster_id": record.get("subcluster_id"),
        "span_text": prompt_text(record),
        "prompt_risk_score": result.risk_score,
        "prompt_confidence": result.confidence,
        "prompt_label": result.label,
        "prompt_harm_category": result.harm_category,
        "prompt_classifier_version": result.version,
        "prompt_classifier_metadata": encode(result.metadata),
    }


def main() -> None:
    args = parse_args()
    corpus_path = Path(args.corpus)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = existing_ids(output_path) if args.resume else set()
    mode = "a" if args.resume else "w"
    qwen_scorer = (
        BatchedQwen3GuardScorer(
            model_id=args.model or "Qwen/Qwen3Guard-Gen-0.6B",
            max_new_tokens=args.max_new_tokens,
            device_map=args.device_map,
        )
        if args.provider == "qwen3guard"
        else None
    )
    feature = make_feature(args.provider, args.model) if qwen_scorer is None else None

    written = 0
    skipped = 0
    batch: list[dict[str, Any]] = []

    def flush_batch(output_file) -> None:
        nonlocal written
        if not batch:
            return
        if qwen_scorer is not None:
            scored_rows = qwen_scorer.score(batch)
        else:
            scored_rows = [score_record(record, feature) for record in batch]
        for scored in scored_rows:
            output_file.write(json.dumps(scored, separators=(",", ":")) + "\n")
            written += 1
            if args.progress_every and written % args.progress_every == 0:
                print(f"wrote {written} prompt scores; skipped {skipped}")
        batch.clear()

    with corpus_path.open(encoding="utf-8") as input_file, output_path.open(
        mode,
        encoding="utf-8",
    ) as output_file:
        for idx, line in enumerate(input_file):
            if args.max_records is not None and idx >= args.max_records:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            row_id = str(record.get("id", ""))
            if row_id in seen:
                skipped += 1
                continue
            batch.append(record)
            if len(batch) >= max(1, args.batch_size):
                flush_batch(output_file)
        flush_batch(output_file)

    print(f"wrote {written} prompt scores to {output_path}; skipped {skipped}")


if __name__ == "__main__":
    main()
