"""Tests for the ladder bundle (ramp.audit.bundle): validation, round-trip, ingest.

Includes the load-bearing fidelity check: a bundle built from a real
survival-ladder report, audited by the package, returns the SAME card the
study's own emitter (scripts/emit_signal_validity_card.py) produces — verdict,
status, claims, AND hashes. If this drifts, the auditor is no longer reporting
what the study found.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from ramp.audit import audit, build_card, render_markdown
from ramp.audit.bundle import OOD_RUNGS, RUNG_ORDER, LadderBundle
from ramp.audit.engine import survival_table

_REPO = Path(__file__).resolve().parent.parent
_REPORTS = sorted((_REPO / ".artifacts" / "prompt_label_audit").glob(
    "ramp_signal_survival_ladder*.json"
))


def _load_original_emitter():
    spec = importlib.util.spec_from_file_location(
        "emit_signal_validity_card_orig",
        _REPO / "scripts" / "emit_signal_validity_card.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


def _good_bundle(**overrides) -> LadderBundle:
    cell = {
        "available": True,
        "candidate": {"auc": 0.8, "f1": 0.75, "recall": 0.9, "fpr": 0.1},
        "baseline": {"auc": 0.7, "f1": 0.65, "recall": 0.9, "fpr": 0.1},
    }
    kwargs = dict(
        signal="activation",
        signal_description="rigged",
        target_model="m",
        n=100,
        rungs={"naive": cell, "split": cell, "crossfit": cell},
    )
    kwargs.update(overrides)
    return LadderBundle(**kwargs)


def test_valid_bundle_passes_validation() -> None:
    _good_bundle().validate()  # no raise


def test_validate_rejects_unknown_rung() -> None:
    b = _good_bundle(rungs={"bogus": {"available": False}})
    with pytest.raises(ValueError, match="unknown rung"):
        b.validate()


def test_validate_rejects_bad_provenance() -> None:
    b = _good_bundle(blind_label_provenance="made_up")
    with pytest.raises(ValueError, match="blind_label_provenance"):
        b.validate()


def test_validate_rejects_available_rung_without_metrics() -> None:
    b = _good_bundle(rungs={"naive": {"available": True}})
    with pytest.raises(ValueError, match="candidate"):
        b.validate()


def test_validate_rejects_rung_without_available_flag() -> None:
    b = _good_bundle(rungs={"naive": {"candidate": {}, "baseline": {}}})
    with pytest.raises(ValueError, match="available"):
        b.validate()


# --------------------------------------------------------------------------- #
# validation: hardened against garbage that used to crash the engine (fix #1)
# --------------------------------------------------------------------------- #


def _avail(candidate, baseline, bootstrap=None):
    cell = {"available": True, "candidate": candidate, "baseline": baseline}
    if bootstrap is not None:
        cell["bootstrap"] = bootstrap
    return cell


def test_validate_rejects_string_metric() -> None:
    b = _good_bundle(
        rungs={"naive": _avail({"auc": "bad", "f1": 0.7}, {"auc": 0.7, "f1": 0.6})}
    )
    with pytest.raises(ValueError, match="finite number"):
        b.validate()


def test_validate_rejects_nan_metric() -> None:
    b = _good_bundle(
        rungs={"naive": _avail({"auc": float("nan"), "f1": 0.7}, {"auc": 0.7, "f1": 0.6})}
    )
    with pytest.raises(ValueError, match="finite number"):
        b.validate()


def test_validate_rejects_inf_metric() -> None:
    b = _good_bundle(
        rungs={"naive": _avail({"auc": float("inf"), "f1": 0.7}, {"auc": 0.7, "f1": 0.6})}
    )
    with pytest.raises(ValueError, match="finite number"):
        b.validate()


def test_validate_rejects_bool_metric() -> None:
    # bool is an int subclass; an AUROC of True is corruption, not a 1.0 metric.
    b = _good_bundle(
        rungs={"naive": _avail({"auc": True, "f1": 0.7}, {"auc": 0.7, "f1": 0.6})}
    )
    with pytest.raises(ValueError, match="finite number"):
        b.validate()


def test_validate_rejects_empty_metrics_dict() -> None:
    b = _good_bundle(rungs={"naive": _avail({}, {"auc": 0.7, "f1": 0.6})})
    with pytest.raises(ValueError, match="numeric 'auc'"):
        b.validate()


def test_validate_rejects_missing_f1() -> None:
    b = _good_bundle(rungs={"naive": _avail({"auc": 0.8}, {"auc": 0.7, "f1": 0.6})})
    with pytest.raises(ValueError, match="numeric 'f1'"):
        b.validate()


def test_validate_rejects_string_optional_metric() -> None:
    b = _good_bundle(
        rungs={
            "naive": _avail(
                {"auc": 0.8, "f1": 0.7, "recall": "x"}, {"auc": 0.7, "f1": 0.6}
            )
        }
    )
    with pytest.raises(ValueError, match="recall"):
        b.validate()


def test_validate_accepts_none_optional_metric() -> None:
    # recall/fpr may legitimately be absent or None; only present non-numerics fail.
    b = _good_bundle(
        rungs={
            "naive": _avail(
                {"auc": 0.8, "f1": 0.7, "recall": None, "fpr": None},
                {"auc": 0.7, "f1": 0.6},
            )
        }
    )
    b.validate()  # no raise


def test_validate_rejects_bootstrap_not_dict() -> None:
    b = _good_bundle(
        rungs={
            "blind": _avail(
                {"auc": 0.8, "f1": 0.7},
                {"auc": 0.7, "f1": 0.6},
                bootstrap={"auc": 5, "f1": {"ci95": [0.0, 1.0], "significant": False}},
            )
        }
    )
    with pytest.raises(ValueError, match="bootstrap 'auc' must be a dict"):
        b.validate()


def test_validate_rejects_bootstrap_significant_not_bool() -> None:
    b = _good_bundle(
        rungs={
            "blind": _avail(
                {"auc": 0.8, "f1": 0.7},
                {"auc": 0.7, "f1": 0.6},
                bootstrap={
                    "auc": {"ci95": [0.01, 0.05], "significant": 1},
                    "f1": {"ci95": [-0.01, 0.01], "significant": False},
                },
            )
        }
    )
    with pytest.raises(ValueError, match="significant' must be a bool"):
        b.validate()


def test_validate_rejects_bootstrap_ci_wrong_length() -> None:
    b = _good_bundle(
        rungs={
            "blind": _avail(
                {"auc": 0.8, "f1": 0.7},
                {"auc": 0.7, "f1": 0.6},
                bootstrap={
                    "auc": {"ci95": [0.01], "significant": True},
                    "f1": {"ci95": [-0.01, 0.01], "significant": False},
                },
            )
        }
    )
    with pytest.raises(ValueError, match="2-element"):
        b.validate()


def test_validate_rejects_bootstrap_ci_nan() -> None:
    b = _good_bundle(
        rungs={
            "blind": _avail(
                {"auc": 0.8, "f1": 0.7},
                {"auc": 0.7, "f1": 0.6},
                bootstrap={
                    "auc": {"ci95": [float("nan"), 0.05], "significant": True},
                    "f1": {"ci95": [-0.01, 0.01], "significant": False},
                },
            )
        }
    )
    with pytest.raises(ValueError, match="finite numbers"):
        b.validate()


def test_validate_rejects_bad_kappa_type() -> None:
    b = _good_bundle(inter_judge_kappa="x")
    with pytest.raises(ValueError, match="inter_judge_kappa"):
        b.validate()


def test_validate_rejects_nan_kappa() -> None:
    b = _good_bundle(human_audit_kappa=float("nan"))
    with pytest.raises(ValueError, match="human_audit_kappa"):
        b.validate()


def test_validate_rejects_bool_n() -> None:
    b = _good_bundle(n=True)
    with pytest.raises(ValueError, match="non-bool"):
        b.validate()


def test_validate_rejects_negative_n() -> None:
    b = _good_bundle(n=-1)
    with pytest.raises(ValueError, match="non-negative"):
        b.validate()


def test_validate_rejects_available_not_bool() -> None:
    b = _good_bundle(
        rungs={"naive": {"available": 1, "candidate": {}, "baseline": {}}}
    )
    with pytest.raises(ValueError, match="'available' must be a bool"):
        b.validate()


def test_engine_no_longer_crashes_on_garbage() -> None:
    """The headline bug: garbage metrics used to pass validate() and crash audit().

    Now audit() (which calls validate first) raises a clean ValueError instead of
    a TypeError from subtracting a string.
    """
    from ramp.audit import audit

    b = _good_bundle(
        rungs={"naive": _avail({"auc": "bad", "f1": "bad"}, {"auc": 0.7, "f1": 0.6})}
    )
    with pytest.raises(ValueError):
        audit(b)


# --------------------------------------------------------------------------- #
# anti-gaming: recorded significance must agree with the recorded CI (fix #4)
# --------------------------------------------------------------------------- #


def test_validate_rejects_significant_true_but_ci_includes_zero() -> None:
    b = _good_bundle(
        rungs={
            "blind": _avail(
                {"auc": 0.8, "f1": 0.7},
                {"auc": 0.7, "f1": 0.6},
                bootstrap={
                    # claims significant, but CI spans zero -> gamed; reject.
                    "auc": {"ci95": [-0.02, 0.03], "significant": True},
                    "f1": {"ci95": [-0.01, 0.01], "significant": False},
                },
            )
        }
    )
    with pytest.raises(ValueError, match="disagrees with ci95"):
        b.validate()


def test_validate_rejects_significant_false_but_ci_excludes_zero() -> None:
    b = _good_bundle(
        rungs={
            "blind": _avail(
                {"auc": 0.8, "f1": 0.7},
                {"auc": 0.7, "f1": 0.6},
                bootstrap={
                    # CI clears zero (lo>0) but flagged not-significant -> reject.
                    "auc": {"ci95": [0.01, 0.05], "significant": False},
                    "f1": {"ci95": [-0.01, 0.01], "significant": False},
                },
            )
        }
    )
    with pytest.raises(ValueError, match="disagrees with ci95"):
        b.validate()


def test_validate_accepts_consistent_significant_negative_ci() -> None:
    # significance can come from a CI entirely below zero (hi < 0); accept it.
    b = _good_bundle(
        rungs={
            "blind": _avail(
                {"auc": 0.8, "f1": 0.7},
                {"auc": 0.7, "f1": 0.6},
                bootstrap={
                    "auc": {"ci95": [-0.05, -0.01], "significant": True},
                    "f1": {"ci95": [-0.01, 0.01], "significant": False},
                },
            )
        }
    )
    b.validate()  # no raise


# --------------------------------------------------------------------------- #
# round-trip
# --------------------------------------------------------------------------- #


def test_to_dict_from_dict_round_trip() -> None:
    b = _good_bundle(
        blind_label_provenance="silver_llm",
        inter_judge_kappa=0.887,
        front_door="qwen3guard_0.6b",
        probe_kind="linear",
    )
    restored = LadderBundle.from_dict(b.to_dict())
    assert restored == b


def test_save_load_round_trip(tmp_path) -> None:
    b = _good_bundle(front_door="fd", probe_kind="mlp")
    path = tmp_path / "bundle.json"
    b.save(path)
    assert LadderBundle.load(path) == b


# --------------------------------------------------------------------------- #
# from_survival_report ingest + survival cell reconstruction
# --------------------------------------------------------------------------- #


def test_from_survival_report_unknown_signal_raises() -> None:
    with pytest.raises(ValueError, match="unknown signal"):
        LadderBundle.from_survival_report({"rungs": {}}, "not_a_signal")


def test_from_survival_report_marks_absent_rungs_unavailable() -> None:
    report = {
        "num_binary_eval_rows": 10,
        "rungs": {
            "naive": {"status": "completed", "aggregate_holdout_metrics": {
                "prompt_activation_calibrated": {
                    "auc": {"mean": 0.8}, "f1": {"mean": 0.7},
                    "recall": {"mean": 0.9}, "false_positive_rate": {"mean": 0.1},
                },
                "prompt_only_calibrated": {
                    "auc": {"mean": 0.7}, "f1": {"mean": 0.6},
                    "recall": {"mean": 0.9}, "false_positive_rate": {"mean": 0.1},
                },
            }},
            "blind": {"status": "pending"},
        },
    }
    b = LadderBundle.from_survival_report(report, "activation")
    assert b.rungs["naive"]["available"] is True
    assert b.rungs["blind"]["available"] is False
    cells = survival_table(b)
    assert cells["naive"]["verdict"] == "survives"
    assert cells["blind"]["verdict"] == "pending"


@pytest.mark.skipif(not _REPORTS, reason="no survival-ladder report artifacts present")
@pytest.mark.parametrize("report_path", _REPORTS, ids=lambda p: p.name)
@pytest.mark.parametrize("signal", ["embedding", "activation", "full_fusion"])
def test_card_matches_original_emitter_on_real_reports(report_path, signal) -> None:
    """Load-bearing fidelity: the package reproduces the study emitter's card exactly."""
    orig = _load_original_emitter()
    report = json.loads(report_path.read_text())
    table = report.get("survival_table", {})
    if signal not in table:
        pytest.skip(f"{signal} not in {report_path.name}")

    # signal descriptions used by the original emitter's defaults
    probe_kind = "linear"
    signal_descriptions = {
        "embedding": "GPT-OSS input-embedding centroid proximity",
        "activation": f"GPT-OSS layer-19 {probe_kind} probe",
        "full_fusion": "prompt + embedding + activation calibrated fusion",
    }
    scope_extra = {
        "target_model": "openai/gpt-oss-20b",
        "front_door": "qwen3guard_0.6b",
        "probe_kind": probe_kind,
        "blind_label_rubric": None,
        "preregistration": None,
        "signal_descriptions": signal_descriptions,
    }
    orig_card = orig.build_card(
        signal,
        table[signal],
        report=report,
        scope_extra=scope_extra,
        blind_label_provenance="silver_llm",
        inter_judge_kappa=0.887,
        report_path=str(report_path),
        rerun_command=None,
    )

    bundle = LadderBundle.from_survival_report(
        report,
        signal,
        target_model="openai/gpt-oss-20b",
        front_door="qwen3guard_0.6b",
        probe_kind=probe_kind,
        signal_description=signal_descriptions[signal],
        blind_label_provenance="silver_llm",
        inter_judge_kappa=0.887,
    )
    new_card = build_card(bundle, audit(bundle))

    assert new_card["verdict"] == orig_card["verdict"]
    assert new_card["status"] == orig_card["status"]
    assert new_card["allowed_claims"] == orig_card["allowed_claims"]
    assert new_card["disallowed_claims"] == orig_card["disallowed_claims"]
    assert new_card["residual_risks"] == orig_card["residual_risks"]
    assert new_card["diagnostics"]["per_rung"] == orig_card["diagnostics"]["per_rung"]
    # hashes match -> config/inputs canonicalization is identical to the study's
    assert new_card["config_hash"] == orig_card["config_hash"]
    assert new_card["inputs_hash"] == orig_card["inputs_hash"]


def test_render_markdown_contains_verdict_and_hashes() -> None:
    b = _good_bundle(front_door="qwen3guard_0.6b", probe_kind="linear")
    card = build_card(b, audit(b))
    md = render_markdown(card)
    assert "Signal Validity Card" in md
    assert "config_hash:" in md
    assert "inputs_hash:" in md


# --------------------------------------------------------------------------- #
# from_raw_scores: re-derive metrics + significance from raw evidence (fix #5)
# --------------------------------------------------------------------------- #


def _raw_rung(labels, cand, base, cand_t, base_t):
    return {
        "labels": labels,
        "candidate_scores": cand,
        "baseline_scores": base,
        "candidate_threshold": cand_t,
        "baseline_threshold": base_t,
    }


def test_from_raw_scores_matches_hand_built_equivalent() -> None:
    """from_raw_scores -> audit equals a bundle whose cells we computed by hand.

    Pins that from_raw_scores re-derives exactly the AUROC/F1 (via stats) and OOD
    bootstrap (via the SEEDED stats.paired_bootstrap) that a hand-built bundle with
    the identical numbers produces — same verdict, status, and survival table.
    """
    import numpy as np

    from ramp.audit import stats

    # A fully deterministic dataset (no RNG): candidate perfectly separates the
    # classes; the baseline genuinely misranks ~a quarter of rows, so it is strictly
    # worse on AUROC and F1. 40 rows so the seeded bootstrap CI excludes zero.
    n_each = 20
    labels = [0] * n_each + [1] * n_each
    # candidate: negatives in [0.0, 0.475], positives in [0.525, 1.0] -> AUROC 1.0
    cand = [round(i / (2 * n_each), 4) for i in range(n_each)] + [
        round(0.525 + i / (2 * n_each), 4) for i in range(n_each)
    ]
    # baseline: same ramp, but every 4th label is flipped above/below 0.5 so the
    # baseline misranks those rows -> AUROC < 1 and F1 < candidate's.
    base = []
    for i in range(n_each):  # negatives: every 4th scores high (false positive)
        base.append(0.8 if i % 4 == 0 else round(0.05 + i / (2 * n_each), 4))
    for i in range(n_each):  # positives: every 4th scores low (false negative)
        base.append(0.1 if i % 4 == 0 else round(0.55 + i / (2 * n_each), 4))
    base = [round(x, 4) for x in base]
    cand_t, base_t = 0.5, 0.5
    seed, resamples = "fixture-seed", 1000

    raw = {r: _raw_rung(labels, cand, base, cand_t, base_t) for r in RUNG_ORDER}
    from_raw = LadderBundle.from_raw_scores(
        "activation", raw, seed=seed, num_resamples=resamples, target_model="m"
    )

    # Hand-build the identical bundle: same metric helpers, same seeded bootstrap.
    npl = np.asarray(labels, dtype=np.int64)
    npc = np.asarray(cand, dtype=np.float64)
    npb = np.asarray(base, dtype=np.float64)

    def hand_cell(rung):
        cell = {
            "available": True,
            "candidate": {
                "auc": stats._fast_auc(np, npl, npc),
                "f1": stats._f1_at(np, npl, npc, cand_t),
                "recall": None,
                "fpr": None,
            },
            "baseline": {
                "auc": stats._fast_auc(np, npl, npb),
                "f1": stats._f1_at(np, npl, npb, base_t),
                "recall": None,
                "fpr": None,
            },
        }
        if rung in OOD_RUNGS:
            boot = stats.paired_bootstrap(
                np, labels, base, base_t, [("candidate", cand, cand_t)],
                num_resamples=resamples, seed=f"{seed}:{rung}",
            )["candidate"]
            cell["bootstrap"] = {
                "auc": {
                    "delta": boot["auc"]["delta"],
                    "ci95": boot["auc"]["ci95"],
                    "significant": bool(boot["auc"]["significant"]),
                },
                "f1": {
                    "delta": boot["f1"]["delta"],
                    "ci95": boot["f1"]["ci95"],
                    "significant": bool(boot["f1"]["significant"]),
                },
            }
        return cell

    hand = LadderBundle(
        signal="activation",
        signal_description="activation",
        target_model="m",
        n=len(labels),
        rungs={r: hand_cell(r) for r in RUNG_ORDER},
    )

    # The two bundles' rungs are byte-identical, hence so are the audits.
    assert from_raw.rungs == hand.rungs
    assert audit(from_raw) == audit(hand)
    assert audit(from_raw)["verdict"] == "distribution_robust"


def test_from_raw_scores_only_marks_present_rungs_available() -> None:
    labels = [0, 0, 1, 1]
    cand = [0.1, 0.2, 0.8, 0.9]  # AUROC 1.0, F1 1.0
    base = [0.6, 0.2, 0.7, 0.3]  # AUROC 0.75, F1 0.5 -> candidate survives
    raw = {"naive": _raw_rung(labels, cand, base, 0.5, 0.5)}
    b = LadderBundle.from_raw_scores("activation", raw, num_resamples=100)
    assert b.rungs["naive"]["available"] is True
    assert "split" not in b.rungs  # absent rung stays absent
    cells = survival_table(b)
    assert cells["naive"]["verdict"] == "survives"
    assert cells["split"]["verdict"] == "not_run"


def test_from_raw_scores_is_reproducible_under_seed() -> None:
    labels = [0, 0, 0, 1, 1, 1]
    cand = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    base = [0.2, 0.4, 0.3, 0.5, 0.6, 0.7]
    raw = {"blind": _raw_rung(labels, cand, base, 0.5, 0.5)}
    a = LadderBundle.from_raw_scores("activation", raw, seed="s", num_resamples=200)
    b = LadderBundle.from_raw_scores("activation", raw, seed="s", num_resamples=200)
    assert a.to_dict() == b.to_dict()


def test_from_raw_scores_unknown_signal_raises() -> None:
    with pytest.raises(ValueError, match="unknown signal"):
        LadderBundle.from_raw_scores("not_a_signal", {})
