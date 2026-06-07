from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ramp.features.base import FeatureExtractor, FeatureInput
from ramp.risk_state import RiskState
from ramp.schemas.feature_result import FeatureResult, FeatureStage

DEFAULT_QWEN3GUARD_MODEL = "Qwen/Qwen3Guard-Gen-0.6B"
RESEARCH_QWEN3GUARD_MODEL = "Qwen/Qwen3Guard-Gen-4B"
QWEN3GUARD_SCORE_MAPPING_VERSION = "qwen3guard_prompt_risk_mapping_v1"

SAFETY_LABEL_PATTERN = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)
CATEGORY_PATTERN = re.compile(
    r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|"
    r"Personally Identifiable Information|Suicide & Self-Harm|Unethical Acts|"
    r"Politically Sensitive Topics|Copyright Violation|Jailbreak|None)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Qwen3GuardResult:
    label: str | None
    categories: tuple[str, ...]
    raw_output: str


def parse_qwen3guard_output(content: str) -> Qwen3GuardResult:
    label_match = SAFETY_LABEL_PATTERN.search(content)
    label = _normalize_label(label_match.group(1)) if label_match else None
    categories = tuple(
        dict.fromkeys(
            _normalize_category(match.group(1)) for match in CATEGORY_PATTERN.finditer(content)
        )
    )
    return Qwen3GuardResult(label=label, categories=categories, raw_output=content.strip())


def qwen3guard_risk_score(label: str | None, categories: tuple[str, ...]) -> tuple[float, float]:
    """Map Qwen's generated label to a RAMP feature risk score and parser confidence."""

    if label == "Safe":
        return 0.08, 0.86
    if label == "Controversial":
        return 0.58, 0.76
    if label == "Unsafe":
        return 0.92, 0.88

    if categories and categories != ("None",):
        return 0.64, 0.35
    return 0.50, 0.20


class Qwen3GuardPromptRiskFeature(FeatureExtractor):
    """Prompt-risk feature backed by Qwen3Guard-Gen.

    The model is loaded lazily so tests and non-Qwen pipelines do not pay startup cost.
    """

    stage = FeatureStage.PROMPT_RISK

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
    def from_env(cls) -> Qwen3GuardPromptRiskFeature:
        return cls(model_id=os.getenv("RAMP_PROMPT_RISK_MODEL", DEFAULT_QWEN3GUARD_MODEL))

    def extract(self, feature_input: FeatureInput, state: RiskState) -> FeatureResult:
        start = time.perf_counter()
        raw_output = self._complete(feature_input.prompt)
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
            cost_tier=1,
            version=f"qwen3guard:{self.model_id}",
            metadata={
                "backend": "qwen3guard",
                "model_id": self.model_id,
                "raw_output": parsed.raw_output,
                "safety_label": parsed.label,
                "categories": list(parsed.categories),
                "risk_score_mapping_version": QWEN3GUARD_SCORE_MAPPING_VERSION,
            },
        )

    def _complete(self, prompt: str) -> str:
        if self._completion_fn is not None:
            return self._completion_fn(prompt)

        self._ensure_model_loaded()
        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:
            raise RuntimeError("Qwen3Guard model failed to load")

        messages = [{"role": "user", "content": prompt}]
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
                "Qwen3GuardPromptRiskFeature requires the qwen optional dependencies. "
                "Install them with `uv pip install -e '.[qwen]'` or `pip install -e '.[qwen]'`."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto",
        )


def _normalize_label(label: str) -> str:
    return label.strip().lower().capitalize()


def _normalize_category(category: str) -> str:
    normalized = category.strip()
    if normalized.lower() == "personally identifiable information":
        return "PII"
    if normalized.lower() == "pii":
        return "PII"
    if normalized.lower() == "none":
        return "None"
    return normalized
