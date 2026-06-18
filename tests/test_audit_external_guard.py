from __future__ import annotations

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "audit_external_guard",
    Path(__file__).resolve().parent.parent / "scripts" / "audit_external_guard.py",
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)


def test_auc_matches_known_value() -> None:
    # perfectly separable -> AUC 1.0
    assert mod.auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    # reversed -> 0.0
    assert mod.auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0
    # single class -> None
    assert mod.auc([1, 1, 1], [0.1, 0.2, 0.3]) is None


def test_op_metrics_threshold() -> None:
    m = mod.op_metrics([1, 1, 0, 0], [0.9, 0.6, 0.4, 0.1], 0.5)
    assert m["tp"] == 2 and m["fp"] == 0 and m["tn"] == 2 and m["fn"] == 0
    assert m["recall"] == 1.0 and m["fpr"] == 0.0 and m["f1"] == 1.0


def test_best_threshold_at_fpr_respects_budget() -> None:
    labels = [1, 1, 0, 0, 0]
    scores = [0.9, 0.7, 0.6, 0.3, 0.1]
    thr = mod.best_threshold_at_fpr(labels, scores, target_fpr=0.0)
    # at 0% FPR budget, threshold must exclude the 0.6 negative
    m = mod.op_metrics(labels, scores, thr)
    assert m["fpr"] == 0.0


def test_derive_verdict_flags_eval_inflation() -> None:
    adaptive = {"f1": 0.90}
    blind = {"f1": 0.70}
    out = mod.derive_verdict(adaptive, blind, {})
    assert out["verdict"] == "eval_inflated"
    assert any("adaptive to blind" in r for r in out["reasons"])


def test_derive_verdict_flags_shift_and_both() -> None:
    shift = {
        "beavertails": {
            "calibration_fpr": 0.10,
            "heldout_fpr": 0.40,
            "calibration_recall": 0.8,
            "heldout_recall": 0.8,
        }
    }
    only_shift = mod.derive_verdict({"f1": 0.9}, {"f1": 0.89}, shift)
    assert only_shift["verdict"] == "shift_fragile"
    both = mod.derive_verdict({"f1": 0.9}, {"f1": 0.7}, shift)
    assert both["verdict"] == "eval_and_shift_fragile"


def test_derive_verdict_flags_recall_collapse_under_shift() -> None:
    # FPR is fine, but recall collapses on the held-out source -> still shift_fragile.
    shift = {
        "wildguardmix": {
            "calibration_fpr": 0.02,
            "heldout_fpr": 0.03,
            "calibration_recall": 0.91,
            "heldout_recall": 0.57,
        }
    }
    out = mod.derive_verdict({"f1": 0.9}, {"f1": 0.89}, shift)
    assert out["verdict"] == "shift_fragile"
    assert any("recall collapses" in r for r in out["reasons"])


def test_derive_verdict_robust_when_stable() -> None:
    out = mod.derive_verdict({"f1": 0.9}, {"f1": 0.89}, {})
    assert out["verdict"] == "robust"


def test_build_card_conforms_and_no_sieve_import() -> None:
    import sys

    report = {
        "guard_kind": "binary",
        "threshold": 0.5,
        "target_fpr": 0.1,
        "adaptive": {"n": 100, "f1": 0.9},
        "blind": {"n": 50, "f1": 0.7},
        "per_source_blind": {"wildguardmix": {}, "beavertails": {}},
        "verdict": {"verdict": "eval_inflated", "reasons": ["x"]},
        "inputs": {"a": 1},
    }
    card = mod.build_card(
        report, guard_name="llama-guard-4-12b", label_provenance="framing_inclusive_judge"
    )
    assert card["axis"] == "evaluation_robustness"
    assert card["verdict"] == "eval_inflated"
    assert card["config_hash"].startswith("sha256:")
    assert not any("sieve" in name.lower() for name in sys.modules)
