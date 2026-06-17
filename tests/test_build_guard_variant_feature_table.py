from __future__ import annotations

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "build_guard_variant_feature_table",
    Path(__file__).resolve().parent.parent / "scripts" / "build_guard_variant_feature_table.py",
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)


def test_build_variant_swaps_prompt_score_and_keeps_baseline() -> None:
    feature_rows = [
        {
            "id": "a",
            "prompt_risk_score": 0.2,
            "embedding_prior_score": 0.3,
            "activation_probability": 0.4,
        },
        {
            "id": "b",
            "prompt_risk_score": 0.9,
            "embedding_prior_score": 0.1,
            "activation_probability": 0.5,
        },
        {
            "id": "c",
            "prompt_risk_score": 0.5,
            "embedding_prior_score": 0.5,
            "activation_probability": 0.5,
        },
    ]
    guard_scores = {
        "a": {"source_id": "a", "guard_status": "scored", "guard_score": 0.75, "guard_model": "g"},
        "b": {"source_id": "b", "guard_status": "error", "guard_score": None, "guard_model": "g"},
    }
    variant, summary = mod.build_variant(feature_rows, guard_scores)
    # Only 'a' is emitted: 'b' errored, 'c' has no guard score.
    assert summary["variant_rows"] == 1
    row = variant[0]
    assert row["id"] == "a"
    assert row["prompt_risk_score"] == 0.75
    assert row["prompt_risk_score_baseline"] == 0.2
    # Internal signals untouched.
    assert row["embedding_prior_score"] == 0.3
    assert row["activation_probability"] == 0.4


def test_load_guard_scores_filters_unscored(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    path.write_text(
        '{"source_id": "a", "guard_status": "scored", "guard_score": 0.6}\n'
        '{"source_id": "b", "guard_status": "scored", "guard_score": null}\n'
        '{"source_id": "c", "guard_status": "parse_error", "guard_score": null}\n',
        encoding="utf-8",
    )
    scores = mod.load_guard_scores(path)
    assert set(scores) == {"a"}
