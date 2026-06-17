from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "compare_label_agreement",
    Path(__file__).resolve().parent.parent / "scripts" / "compare_label_agreement.py",
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=["source_id", "review_status", "reviewed_label"]
        )
        writer.writeheader()
        writer.writerows(rows)


def test_load_binary_labels_filters_nonbinary_and_unreviewed(tmp_path: Path) -> None:
    path = tmp_path / "a.csv"
    _write(
        path,
        [
            {"source_id": "1", "review_status": "reviewed", "reviewed_label": "unsafe"},
            {"source_id": "2", "review_status": "reviewed", "reviewed_label": "safe"},
            {"source_id": "3", "review_status": "reviewed", "reviewed_label": "controversial"},
            {"source_id": "4", "review_status": "", "reviewed_label": "unsafe"},
        ],
    )
    labels = mod.load_binary_labels(path)
    assert labels == {"1": 1, "2": 0}


def test_cohen_kappa_perfect_agreement() -> None:
    a = {"1": 1, "2": 0, "3": 1, "4": 0}
    result = mod.cohen_kappa(a, dict(a))
    assert result["n"] == 4
    assert result["raw_agreement"] == 1.0
    assert result["kappa"] == 1.0


def test_cohen_kappa_partial_and_confusion() -> None:
    a = {"1": 1, "2": 1, "3": 0, "4": 0}
    b = {"1": 1, "2": 0, "3": 0, "4": 0}  # disagree on "2"
    result = mod.cohen_kappa(a, b)
    assert result["n"] == 4
    assert result["raw_agreement"] == 0.75
    assert result["confusion"] == {"a0_b0": 2, "a0_b1": 0, "a1_b0": 1, "a1_b1": 1}
    assert result["kappa"] is not None and result["kappa"] < 1.0


def test_cohen_kappa_no_overlap() -> None:
    result = mod.cohen_kappa({"1": 1}, {"2": 0})
    assert result["n"] == 0
    assert result["kappa"] is None
