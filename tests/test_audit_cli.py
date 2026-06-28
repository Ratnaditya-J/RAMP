"""Tests for the ramp-audit CLI (ramp.audit.cli): error handling + happy path.

The CLI is a user-facing surface, so a missing file, malformed JSON, or a
structurally invalid bundle must produce a clean one-line ``[ramp-audit] error:``
on stderr and a non-zero exit -- never a Python traceback (fix #6).
"""
from __future__ import annotations

import json

import pytest

from ramp.audit.bundle import LadderBundle
from ramp.audit.cli import main


def _good_bundle_dict() -> dict:
    cell = {
        "available": True,
        "candidate": {"auc": 0.8, "f1": 0.75, "recall": 0.9, "fpr": 0.1},
        "baseline": {"auc": 0.7, "f1": 0.65, "recall": 0.9, "fpr": 0.1},
    }
    return LadderBundle(
        signal="activation",
        signal_description="rigged",
        target_model="m",
        n=100,
        rungs={"naive": cell, "split": cell, "crossfit": cell},
    ).to_dict()


def test_cli_missing_bundle_errors_cleanly(tmp_path, capsys) -> None:
    missing = tmp_path / "nope.json"
    rc = main(["audit", "--bundle", str(missing)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "[ramp-audit] error:" in captured.err
    assert "not found" in captured.err
    # no traceback leaked to either stream
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def test_cli_bad_json_errors_cleanly(tmp_path, capsys) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not json ")
    rc = main(["audit", "--bundle", str(bad)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "[ramp-audit] error:" in captured.err
    assert "not valid JSON" in captured.err
    assert "Traceback" not in captured.err


def test_cli_invalid_bundle_errors_cleanly(tmp_path, capsys) -> None:
    # Structurally invalid: a string AUROC -> validate() raises ValueError.
    d = _good_bundle_dict()
    d["rungs"]["naive"]["candidate"]["auc"] = "not_a_number"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(d))
    rc = main(["audit", "--bundle", str(path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "[ramp-audit] error:" in captured.err
    assert "finite number" in captured.err
    assert "Traceback" not in captured.err


def test_cli_happy_path_writes_card(tmp_path, capsys) -> None:
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(_good_bundle_dict()))
    out = tmp_path / "card.json"
    rc = main(["audit", "--bundle", str(path), "--out", str(out)])
    captured = capsys.readouterr()
    assert rc == 0
    assert out.exists()
    card = json.loads(out.read_text())
    assert card["axis"] == "evaluation_robustness"
    # naive/split/crossfit all survive, OOD absent -> in_distribution_only, capped.
    assert card["verdict"] == "in_distribution_only"
    assert "[ramp-audit] verdict:" in captured.out


def test_cli_selftest_subcommand_passes(capsys) -> None:
    rc = main(["audit-selftest"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "selftest passed" in captured.out


def test_cli_no_command_prints_help(capsys) -> None:
    rc = main([])
    captured = capsys.readouterr()
    assert rc == 0
    assert "ramp" in captured.out.lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
