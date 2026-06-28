"""Tests for the pure robustness-audit engine (ramp.audit.engine).

Covers every rigged selftest scenario plus the per-cell survival rule and the
anti-gaming asymmetry, asserting the engine returns EXACTLY the ground-truth
verdicts. These mirror SIEVE's selftest-as-tests pattern.
"""
from __future__ import annotations

import pytest

from ramp.audit import audit, run_selftest
from ramp.audit.engine import (
    INSUFFICIENT_PROTOCOL,
    cell_verdict,
    derive_robustness_verdict,
)
from ramp.audit.selftest import SCENARIOS, check_scenario

# --------------------------------------------------------------------------- #
# the rigged ground-truth scenarios (the heart of the validity guarantee)
# --------------------------------------------------------------------------- #


def test_run_selftest_all_green() -> None:
    assert run_selftest(verbose=False) is True


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario_returns_rigged_verdict(name: str) -> None:
    ok, detail = check_scenario(name)
    assert ok, f"{name}: {detail}"


@pytest.mark.parametrize(
    "name,expected_verdict,expected_status",
    [(n, v, s) for n, (_g, v, s) in SCENARIOS.items()],
)
def test_scenario_verdict_and_status(name: str, expected_verdict, expected_status) -> None:
    generator = SCENARIOS[name][0]
    result = audit(generator())
    assert result["verdict"] == expected_verdict
    assert result["status"] == expected_status


# --------------------------------------------------------------------------- #
# per-cell survival rule (ported survival_verdict)
# --------------------------------------------------------------------------- #


def test_cell_survives_requires_both_deltas_positive() -> None:
    cell = cell_verdict({"auc": 0.8, "f1": 0.7}, {"auc": 0.7, "f1": 0.6})
    assert cell["verdict"] == "survives"
    assert cell["auc_delta"] == pytest.approx(0.1)
    assert cell["f1_delta"] == pytest.approx(0.1)


def test_cell_mixed_when_exactly_one_improves() -> None:
    cell = cell_verdict({"auc": 0.8, "f1": 0.5}, {"auc": 0.7, "f1": 0.6})
    assert cell["verdict"] == "mixed"


def test_cell_fails_when_neither_improves() -> None:
    cell = cell_verdict({"auc": 0.6, "f1": 0.5}, {"auc": 0.7, "f1": 0.6})
    assert cell["verdict"] == "fails"


def test_cell_incomplete_when_a_delta_is_none() -> None:
    cell = cell_verdict({"auc": None, "f1": 0.7}, {"auc": 0.7, "f1": 0.6})
    assert cell["verdict"] == "incomplete"
    assert cell["auc_delta"] is None


def test_cell_surfaces_bootstrap_significance() -> None:
    boot = {
        "auc": {"ci95": [0.01, 0.05], "significant": True},
        "f1": {"ci95": [-0.01, 0.01], "significant": False},
    }
    cell = cell_verdict({"auc": 0.8, "f1": 0.7}, {"auc": 0.7, "f1": 0.6}, boot)
    assert cell["auc_significant"] is True
    assert cell["f1_significant"] is False
    assert cell["auc_ci95"] == [0.01, 0.05]


# --------------------------------------------------------------------------- #
# cross-rung verdict + anti-gaming asymmetry (ported derive_robustness_verdict)
# --------------------------------------------------------------------------- #


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


def _verdict(cells, provenance="human", kappa=None, human_audit_kappa=None):
    return derive_robustness_verdict(
        cells,
        blind_label_provenance=provenance,
        inter_judge_kappa=kappa,
        human_audit_kappa=human_audit_kappa,
    )


def test_no_value_when_fails_everywhere() -> None:
    out = _verdict(_cells("fails", "fails", "fails", "fails", "fails"))
    assert out["verdict"] == "no_value"


def test_leak_inflated_when_only_leaky_rungs_pass() -> None:
    out = _verdict(_cells("survives", "survives", "mixed", ("mixed", True), "fails"))
    assert out["verdict"] == "leak_inflated"


def test_in_distribution_only_when_ood_not_significant() -> None:
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


def test_missing_blind_caps_claim_and_flags_insufficient_protocol() -> None:
    cells = _cells("survives", "survives", "survives", "pending", ("survives", True))
    # "pending" is not in {survives,mixed,fails} -> rung unavailable.
    out = _verdict(cells)
    assert out["verdict"] == "in_distribution_only"
    assert out["status"] == INSUFFICIENT_PROTOCOL


def test_missing_naive_refuses_verdict() -> None:
    cells = _cells("not_run", "survives", "survives", ("survives", True), ("survives", True))
    out = _verdict(cells)
    assert out["verdict"] is None
    assert out["status"] == INSUFFICIENT_PROTOCOL


# --------------------------------------------------------------------------- #
# crossfit-absent edge: negative verdicts are provisional, not settled (fix #3)
# --------------------------------------------------------------------------- #


def test_no_value_with_crossfit_absent_is_insufficient_protocol() -> None:
    # fails leaky and crossfit was never run -> no_value, but provisional.
    cells = _cells("fails", "fails", "not_run", "not_run", "not_run")
    out = _verdict(cells)
    assert out["verdict"] == "no_value"
    assert out["status"] == INSUFFICIENT_PROTOCOL
    assert any("crossfit rung not available" in r for r in out["reasons"])


def test_leak_inflated_with_crossfit_absent_is_insufficient_protocol() -> None:
    # survives leaky, crossfit never run -> leak_inflated, but provisional.
    cells = _cells("survives", "survives", "not_run", "not_run", "not_run")
    out = _verdict(cells)
    assert out["verdict"] == "leak_inflated"
    assert out["status"] == INSUFFICIENT_PROTOCOL
    assert any("crossfit rung not available" in r for r in out["reasons"])


def test_no_value_with_crossfit_present_stays_ok() -> None:
    # crossfit ran and failed -> a settled no_value; status must stay ok.
    cells = _cells("fails", "fails", "fails", "fails", "fails")
    out = _verdict(cells)
    assert out["verdict"] == "no_value"
    assert out["status"] == "ok"


def test_leak_inflated_with_crossfit_present_stays_ok() -> None:
    # crossfit ran and failed while leaky rungs survived -> settled leak_inflated.
    cells = _cells("survives", "survives", "fails", "fails", "fails")
    out = _verdict(cells)
    assert out["verdict"] == "leak_inflated"
    assert out["status"] == "ok"


def test_silver_labels_forbid_human_validation_claim() -> None:
    cells = _cells("survives", "survives", "survives", ("survives", True), ("survives", True))
    out = _verdict(cells, provenance="silver_llm", kappa=0.887)
    assert out["verdict"] == "distribution_robust"
    assert any("human-labeled blind data" in c for c in out["claims"]["disallowed"])
    assert any("silver labels" in c for c in out["claims"]["residual"])
    assert any("0.887" in c for c in out["claims"]["residual"])


def test_human_audit_kappa_updates_residual_from_pending_to_audited() -> None:
    cells = _cells("survives", "survives", "survives", ("survives", True), ("survives", False))
    pending = _verdict(cells, provenance="silver_llm", kappa=0.81)
    audited = _verdict(cells, provenance="silver_llm", kappa=0.81, human_audit_kappa=0.57)
    assert any("human audit pending" in c for c in pending["claims"]["residual"])
    assert not any("human audit pending" in c for c in audited["claims"]["residual"])
    assert any("human-audited at kappa=0.570" in c for c in audited["claims"]["residual"])
