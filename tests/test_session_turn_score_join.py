from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_flatten_and_join_session_turn_scores(tmp_path: Path) -> None:
    corpus = tmp_path / "sessions.jsonl"
    turns = tmp_path / "turns.jsonl"
    scores = tmp_path / "scores.jsonl"
    scored_sessions = tmp_path / "scored_sessions.jsonl"
    session = {
        "session_id": "s1",
        "source": "synthetic",
        "source_record_id": "r1",
        "session_label": "unsafe",
        "session_risk_type": "persistence",
        "turns": [
            {"turn_id": "s1_t1", "turn_index": 0, "role": "user", "text": "hello"},
            {"turn_id": "s1_t2", "turn_index": 1, "role": "agent", "text": "bad answer"},
        ],
    }
    corpus.write_text(json.dumps(session) + "\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/flatten_session_turns.py",
            "--session-corpus",
            str(corpus),
            "--output",
            str(turns),
        ],
        check=True,
    )
    turn_rows = [json.loads(line) for line in turns.read_text().splitlines()]
    assert [row["id"] for row in turn_rows] == ["s1_t1", "s1_t2"]

    scores.write_text(
        json.dumps({"id": "s1_t2", "prompt_risk_score": 0.91, "prompt_label": "Unsafe"})
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/join_session_turn_scores.py",
            "--session-corpus",
            str(corpus),
            "--turn-scores",
            str(scores),
            "--output",
            str(scored_sessions),
        ],
        check=True,
    )
    joined = json.loads(scored_sessions.read_text().splitlines()[0])
    assert joined["turns"][1]["risk_score"] == 0.91
    assert joined["turns"][1]["classifier_label"] == "Unsafe"
