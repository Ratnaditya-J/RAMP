from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "label_blind_batch_llm_judge",
    Path(__file__).resolve().parent.parent / "scripts" / "label_blind_batch_llm_judge.py",
)
judge = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(judge)


def test_parse_judge_content_accepts_clean_json() -> None:
    parsed = judge.parse_judge_content('{"label": "unsafe", "confidence": 0.9, "rationale": "x"}')
    assert parsed["label"] == "unsafe"
    assert parsed["confidence"] == 0.9
    assert parsed["parse_error"] is False


def test_parse_judge_content_extracts_json_from_prose() -> None:
    reply = (
        'Here is my answer:\n{"label": "safe", "confidence": 0.7, "rationale": "benign"}\nThanks!'
    )
    parsed = judge.parse_judge_content(reply)
    assert parsed["label"] == "safe"
    assert parsed["confidence"] == 0.7


def test_parse_judge_content_rejects_unknown_label() -> None:
    parsed = judge.parse_judge_content('{"label": "kinda_bad", "confidence": 0.5}')
    assert parsed["label"] is None
    assert parsed["parse_error"] is True


def test_parse_judge_content_clamps_confidence() -> None:
    parsed = judge.parse_judge_content('{"label": "safe", "confidence": 5}')
    assert parsed["confidence"] == 1.0


def test_parse_judge_content_handles_garbage() -> None:
    parsed = judge.parse_judge_content("the model refused to answer")
    assert parsed["parse_error"] is True


def test_write_csv_marks_only_labeled_rows_reviewed(tmp_path: Path) -> None:
    records = [
        {
            "review_id": "b1",
            "source_id": "row-1",
            "source": "wildguardmix",
            "prompt_text": "p1",
            "reviewed_label": "unsafe",
            "judge_model": "anthropic/claude-opus-4.1",
            "judge_confidence": 0.9,
            "judge_rationale": "harmful",
            "judge_status": "labeled",
        },
        {
            "review_id": "b2",
            "source_id": "row-2",
            "source": "beavertails",
            "prompt_text": "p2",
            "reviewed_label": "",
            "judge_status": "error",
        },
    ]
    output = tmp_path / "labeled.csv"
    judge.write_csv(records, output)
    with output.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    assert rows[0]["review_status"] == "reviewed"
    assert rows[0]["reviewed_label"] == "unsafe"
    # An errored row must NOT be marked reviewed, so the ladder will not consume it.
    assert rows[1]["review_status"] == ""


def test_load_done_ids_only_counts_labeled(tmp_path: Path) -> None:
    jsonl = tmp_path / "out.jsonl"
    jsonl.write_text(
        '{"source_id": "a", "judge_status": "labeled"}\n'
        '{"source_id": "b", "judge_status": "error"}\n',
        encoding="utf-8",
    )
    done = judge.load_done_ids(jsonl)
    assert done == {"a"}
