from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "score_prompts_external_guard",
    Path(__file__).resolve().parent.parent / "scripts" / "score_prompts_external_guard.py",
)
guard = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(guard)


def test_parse_probability_clean() -> None:
    parsed = guard.parse_probability('{"unsafe_probability": 0.83, "label": "unsafe"}')
    assert parsed["guard_score"] == 0.83
    assert parsed["guard_label"] == "unsafe"
    assert parsed["parse_error"] is False


def test_parse_probability_clamps_and_infers_label() -> None:
    parsed = guard.parse_probability('{"unsafe_probability": 1.4}')
    assert parsed["guard_score"] == 1.0
    assert parsed["guard_label"] == "unsafe"


def test_parse_probability_rejects_missing_field() -> None:
    assert guard.parse_probability('{"label": "safe"}')["parse_error"] is True
    assert guard.parse_probability("no json here")["parse_error"] is True


def test_parse_llamaguard_safe_and_unsafe() -> None:
    safe = guard.parse_llamaguard("safe")
    assert safe["guard_score"] == 0.0 and safe["guard_label"] == "safe"
    unsafe = guard.parse_llamaguard("unsafe\nS2")
    assert unsafe["guard_score"] == 1.0 and unsafe["guard_label"] == "unsafe"
    assert unsafe["categories"] == ["s2"]


def test_parse_llamaguard_rejects_junk() -> None:
    assert guard.parse_llamaguard("I cannot help")["parse_error"] is True


def test_collect_eval_ids_unions_review_csvs_and_uses_span_text(tmp_path: Path) -> None:
    feature_table = tmp_path / "features.jsonl"
    feature_table.write_text(
        "\n".join(
            json.dumps({"id": f"row-{i}", "span_text": f"prompt {i}", "source": "s"})
            for i in range(5)
        )
        + "\n",
        encoding="utf-8",
    )
    review = tmp_path / "review.csv"
    with review.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["source_id"])
        writer.writeheader()
        writer.writerows([{"source_id": "row-1"}, {"source_id": "row-3"}, {"source_id": "missing"}])
    prompts = guard.collect_eval_ids(feature_table, [review])
    assert prompts == {"row-1": "prompt 1", "row-3": "prompt 3"}


def test_load_done_ids_only_scored(tmp_path: Path) -> None:
    jsonl = tmp_path / "scores.jsonl"
    jsonl.write_text(
        '{"source_id": "a", "guard_status": "scored"}\n'
        '{"source_id": "b", "guard_status": "error"}\n',
        encoding="utf-8",
    )
    assert guard.load_done_ids(jsonl) == {"a"}
