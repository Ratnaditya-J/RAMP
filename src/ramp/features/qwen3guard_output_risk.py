from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

from ramp.features.base import FeatureExtractor, FeatureInput
from ramp.features.qwen3guard_prompt_risk import (
    DEFAULT_QWEN3GUARD_MODEL,
    QWEN3GUARD_SCORE_MAPPING_VERSION,
    parse_qwen3guard_output,
    qwen3guard_risk_score,
)
from ramp.risk_state import RiskState
from ramp.schemas.feature_result import FeatureResult, FeatureStage


class Qwen3GuardOutputRiskFeature(FeatureExtractor):
    """Output-risk feature backed by the same Qwen3Guard interface as prompt risk."""

    stage = FeatureStage.OUTPUT_RISK

    def __init__(
        self,
        model_id: str = DEFAULT_QWEN3GUARD_MODEL,
        *,
        max_new_tokens: int = 128,
        completion_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._completion_fn = completion_fn
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    @classmethod
    def from_env(cls) -> Qwen3GuardOutputRiskFeature:
        return cls(
            model_id=os.getenv(
                "RAMP_OUTPUT_RISK_MODEL",
                os.getenv("RAMP_PROMPT_RISK_MODEL", DEFAULT_QWEN3GUARD_MODEL),
            )
        )

    def extract(self, feature_input: FeatureInput, state: RiskState) -> FeatureResult:
        output_text = feature_input.output or ""
        start = time.perf_counter()
        raw_output = self._complete(output_text)
        parsed = parse_qwen3guard_output(raw_output)
        risk_score, confidence = qwen3guard_risk_score(parsed.label, parsed.categories)
        latency_ms = round((time.perf_counter() - start) * 1000)
        harm_category = next(
            (category for category in parsed.categories if category != "None"),
            None,
        )
        return FeatureResult(
            stage=self.stage,
            risk_score=risk_score,
            confidence=confidence,
            label=parsed.label or "parse_error",
            harm_category=harm_category,
            latency_ms=latency_ms,
            cost_tier=3,
            version=f"qwen3guard_output:{self.model_id}",
            metadata={
                "backend": "qwen3guard",
                "model_id": self.model_id,
                "raw_output": parsed.raw_output,
                "safety_label": parsed.label,
                "categories": list(parsed.categories),
                "risk_score_mapping_version": QWEN3GUARD_SCORE_MAPPING_VERSION,
                "scored_text": "output",
            },
        )

    def _complete(self, output_text: str) -> str:
        if self._completion_fn is not None:
            return self._completion_fn(output_text)

        self._ensure_model_loaded()
        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:
            raise RuntimeError("Qwen3Guard model failed to load")

        messages = [{"role": "user", "content": output_text}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        return tokenizer.decode(output_ids, skip_special_tokens=True)

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3GuardOutputRiskFeature requires the qwen optional dependencies. "
                "Install them with `uv pip install -e '.[qwen]'` or `pip install -e '.[qwen]'`."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto",
        )
