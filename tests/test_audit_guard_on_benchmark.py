from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

# audit_guard_on_benchmark imports a sibling script, so scripts/ must be importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

SPEC = importlib.util.spec_from_file_location(
    "audit_guard_on_benchmark",
    Path(__file__).resolve().parent.parent / "scripts" / "audit_guard_on_benchmark.py",
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)


def test_load_corpus_filters_nonbinary(tmp_path: Path) -> None:
    path = tmp_path / "corpus.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source_id", "source", "label", "prompt_text"])
        w.writeheader()
        w.writerows(
            [
                {"source_id": "a", "source": "rel1", "label": "unsafe", "prompt_text": "x"},
                {"source_id": "b", "source": "rel1", "label": "safe", "prompt_text": "y"},
                {"source_id": "c", "source": "rel2", "label": "controversial", "prompt_text": "z"},
            ]
        )
    corpus = mod.load_corpus(path)
    assert corpus == {"a": (1, "rel1"), "b": (0, "rel1")}


def test_build_card_shape_and_no_sieve_import() -> None:
    import sys

    report = {
        "guard_kind": "continuous",
        "threshold": 0.5,
        "target_fpr": 0.1,
        "overall": {"n": 200, "auc": 0.8, "f1": 0.7},
        "per_source": {"rel1": {}, "rel2": {}},
        "verdict": {"verdict": "shift_fragile", "reasons": ["recall collapses"]},
        "inputs": {"x": 1},
    }
    card = mod.build_card(report, guard_name="g", benchmark="toxicchat")
    assert card["axis"] == "evaluation_robustness"
    assert card["verdict"] == "shift_fragile"
    assert card["scope"]["axis_specific"]["label_provenance"] == "benchmark_native"
    assert card["config_hash"].startswith("sha256:")
    assert not any("sieve" in name.lower() for name in sys.modules)
