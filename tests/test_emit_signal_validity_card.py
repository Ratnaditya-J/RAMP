from __future__ import annotations

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "emit_signal_validity_card",
    Path(__file__).resolve().parent.parent / "scripts" / "emit_signal_validity_card.py",
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)


def _cells(naive, split, crossfit, blind, shifted):
    def cell(v, sig=None):
        c = {"verdict": v}
        if sig is not None:
            c["auc_significant"] = sig
        return c

    return {
        "naive": cell(naive),
        "split": cell(split),
        "crossfit": cell(crossfit),
        "blind": cell(*blind) if isinstance(blind, tuple) else cell(blind),
        "shifted": cell(*shifted) if isinstance(shifted, tuple) else cell(shifted),
    }


def _verdict(cells, provenance="human", kappa=None):
    return mod.derive_robustness_verdict(
        cells, blind_label_provenance=provenance, inter_judge_kappa=kappa
    )


def test_no_value_when_fails_everywhere() -> None:
    out = _verdict(_cells("fails", "fails", "fails", "fails", "fails"))
    assert out["verdict"] == "no_value"


def test_leak_inflated_when_only_leaky_rungs_pass() -> None:
    # survives naive/split, but crossfit only mixed -> leakage-driven.
    out = _verdict(_cells("survives", "survives", "mixed", ("mixed", True), "fails"))
    assert out["verdict"] == "leak_inflated"


def test_in_distribution_only_when_crossfit_passes_but_ood_weak() -> None:
    out = _verdict(
        _cells("survives", "survives", "survives", ("mixed", False), ("survives", False))
    )
    assert out["verdict"] == "in_distribution_only"
    assert any("distribution shift" in c for c in out["claims"]["disallowed"])


def test_distribution_robust_requires_significant_ood() -> None:
    out = _verdict(
        _cells("survives", "survives", "survives", ("survives", True), ("survives", True))
    )
    assert out["verdict"] == "distribution_robust"


def test_asymmetry_missing_blind_caps_claim() -> None:
    cells = _cells("survives", "survives", "survives", "pending", ("survives", True))
    out = _verdict(cells)
    # blind unavailable -> cannot earn distribution_robust; capped + insufficient_protocol.
    assert out["verdict"] == "in_distribution_only"
    assert out["status"] == mod.INSUFFICIENT_PROTOCOL


def test_silver_labels_forbid_human_validation_claim() -> None:
    cells = _cells("survives", "survives", "survives", ("survives", True), ("survives", True))
    out = _verdict(cells, provenance="silver_llm", kappa=0.887)
    assert out["verdict"] == "distribution_robust"
    assert any("human-labeled blind data" in c for c in out["claims"]["disallowed"])
    assert any("silver labels" in c for c in out["claims"]["residual"])
    assert any("0.887" in c for c in out["claims"]["residual"])


def test_build_card_shape_and_no_sieve_import() -> None:
    import sys

    report = {
        "num_binary_eval_rows": 448,
        "artifact_id": "ramp_signal_survival_ladder_v0.1",
        "survival_rule": "rule",
        "num_splits": 30,
        "calibration_folds": 5,
        "review_csv": "a.csv",
        "survival_table": {
            "activation": _cells("survives", "survives", "mixed", ("mixed", True), "fails"),
        },
    }
    card = mod.build_card(
        "activation",
        report["survival_table"]["activation"],
        report=report,
        scope_extra={
            "target_model": "m",
            "front_door": "qwen",
            "probe_kind": "linear",
            "preregistration": None,
            "signal_descriptions": {},
        },
        blind_label_provenance="silver_llm",
        inter_judge_kappa=0.887,
        report_path="r.json",
        rerun_command="cmd",
    )
    assert card["axis"] == "evaluation_robustness"
    assert card["verdict"] == "leak_inflated"
    assert card["config_hash"].startswith("sha256:")
    assert card["inputs_hash"].startswith("sha256:")
    # Standalone guarantee: no SIEVE module imported by the emitter.
    assert not any("sieve" in name.lower() for name in sys.modules)
