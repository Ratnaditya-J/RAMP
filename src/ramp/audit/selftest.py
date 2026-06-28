"""Rigged ground-truth self-test: the RAMP robustness auditor auditing itself.

Each generator builds a ``LadderBundle`` whose correct SVC verdict is known *by
construction*, and ``run_selftest`` asserts the engine returns exactly that
verdict (and, where it matters, the exact status / disallowed claim). A validity
auditor whose own verdicts have not been checked against known ground truth
would be an embarrassment; this is the antidote, and the direct sibling of
``sieve selftest`` (``~/sieve-audit``, ``synth.py`` / ``sieve selftest``).

Scenarios (by construction):

- ``no_value``: candidate worse than baseline at naive -> fails naive.
- ``leak_inflated``: survives naive + split, fails crossfit.
- ``in_distribution_only``: survives crossfit; blind & shifted present but NOT
  robustly-significant -> capped.
- ``distribution_robust``: survives crossfit; blind & shifted both survive AND
  are AUROC-significant.
- ``insufficient_protocol_a``: naive rung absent -> verdict None, status
  insufficient_protocol.
- ``insufficient_protocol_b``: survives crossfit, blind/shifted absent ->
  in_distribution_only, status insufficient_protocol.
- ``silver_label_asymmetry``: a distribution_robust signal whose blind labels
  are not human -> "claim validation against human-labeled blind data" must be
  a disallowed claim.

Discriminating scenarios that pin the exact rules against mutation:

- ``zero_delta_crossfit_not_robust``: crossfit has an exactly-zero AUROC delta ->
  mixed, not survives (pins the strict ``> 0``: a 0 must NOT survive).
- ``leaky_naive_only`` / ``leaky_split_only``: exactly one leaky rung survives ->
  leak_inflated (pins ``passes_leaky = survives(naive) OR survives(split)`` over
  both rungs, against an ``AND`` mutation).
- ``ood_significant_but_not_surviving``: blind is significant but only mixed ->
  not robust (pins ``_robustly_passes_ood = survives AND significant``).
- ``blind_robust_shifted_not`` / ``shifted_robust_blind_not``: exactly one OOD
  rung robustly passes -> capped (pins that BOTH OOD rungs are required).
- ``human_audited_silver``: distribution_robust, silver labels human-audited at
  kappa=0.62 -> the residual reports the audit (substantial), not 'pending'.

The four canonical single-verdict scenarios additionally assert the FULL claims
triple (allowed/disallowed/residual), not just one substring.

The engine is PURE, so these assertions are deterministic.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .bundle import LadderBundle
from .engine import INSUFFICIENT_PROTOCOL, audit

# --------------------------------------------------------------------------- #
# cell builders
# --------------------------------------------------------------------------- #


def _cell(
    *,
    auc_c: float,
    f1_c: float,
    auc_b: float,
    f1_b: float,
    recall_c: float = 0.9,
    recall_b: float = 0.9,
    fpr_c: float = 0.1,
    fpr_b: float = 0.1,
    significant: bool | None = None,
) -> dict[str, Any]:
    """An available rung cell with explicit candidate vs baseline metrics.

    ``significant`` (when not None) attaches an OOD paired-bootstrap block whose
    AUROC CI excludes zero iff ``significant`` is True (the CI bounds are rigged
    to match, so ``cell_verdict`` reads ``auc_significant`` straight through).
    """
    cell: dict[str, Any] = {
        "available": True,
        "candidate": {"auc": auc_c, "f1": f1_c, "recall": recall_c, "fpr": fpr_c},
        "baseline": {"auc": auc_b, "f1": f1_b, "recall": recall_b, "fpr": fpr_b},
    }
    if significant is not None:
        delta = auc_c - auc_b
        ci95 = [0.01, 0.05] if significant else [-0.02, 0.03]
        cell["bootstrap"] = {
            "auc": {"delta": delta, "ci95": ci95, "significant": significant},
            "f1": {"delta": f1_c - f1_b, "ci95": [-0.01, 0.01], "significant": False},
        }
    return cell


def _survives_cell(significant: bool | None = None) -> dict[str, Any]:
    """Candidate beats baseline on BOTH AUROC and F1 -> survives."""
    return _cell(auc_c=0.80, f1_c=0.75, auc_b=0.70, f1_b=0.65, significant=significant)


def _fails_cell(significant: bool | None = None) -> dict[str, Any]:
    """Candidate worse than baseline on both -> fails."""
    return _cell(auc_c=0.60, f1_c=0.55, auc_b=0.70, f1_b=0.65, significant=significant)


def _mixed_cell(significant: bool | None = None) -> dict[str, Any]:
    """AUROC improves, F1 does NOT -> exactly one improvement -> mixed.

    Used to pin two rules at once: (a) ``mixed`` is one improvement, not two, and
    (b) a ``mixed`` cell that is bootstrap-significant must STILL fail
    ``_robustly_passes_ood`` (which requires ``survives``, not mere significance).
    """
    return _cell(auc_c=0.80, f1_c=0.60, auc_b=0.70, f1_b=0.65, significant=significant)


def _zero_delta_auc_cell() -> dict[str, Any]:
    """AUROC delta EXACTLY zero, F1 improves -> one strict improvement -> mixed.

    Pins ``cell_verdict``'s strict ``> 0``: under the correct rule this is ``mixed``
    (the zero-delta metric is NOT an improvement); a ``>= 0`` mutation would miscount
    it as a second improvement and read ``survives``.
    """
    return _cell(auc_c=0.70, f1_c=0.75, auc_b=0.70, f1_b=0.65)


def _unavailable_cell(status: str = "not_run") -> dict[str, Any]:
    return {"available": False, "status": status}


def _scope(signal: str, **overrides: Any) -> dict[str, Any]:
    base = dict(
        signal=signal,
        signal_description=f"rigged {signal} signal (synthetic)",
        target_model="synthetic/ground-truth",
        n=448,
        front_door="qwen3guard_0.6b",
        probe_kind="linear",
        config={"survival_rule": "rigged", "num_splits": 30, "calibration_folds": 5},
        inputs={"review_csv": "synthetic.csv"},
    )
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# the rigged scenarios
# --------------------------------------------------------------------------- #


def scenario_no_value() -> LadderBundle:
    """Candidate worse than baseline at naive: no value under any rung."""
    return LadderBundle(
        **_scope("activation"),
        rungs={
            "naive": _fails_cell(),
            "split": _fails_cell(),
            "crossfit": _fails_cell(),
            "blind": _fails_cell(significant=False),
            "shifted": _fails_cell(significant=False),
        },
    )


def scenario_leak_inflated() -> LadderBundle:
    """Survives the leaky rungs (naive + split) but fails leakage-free crossfit."""
    return LadderBundle(
        **_scope("activation"),
        rungs={
            "naive": _survives_cell(),
            "split": _survives_cell(),
            "crossfit": _fails_cell(),
            "blind": _fails_cell(significant=False),
            "shifted": _fails_cell(significant=False),
        },
    )


def scenario_in_distribution_only() -> LadderBundle:
    """Survives crossfit; blind & shifted present but not robustly significant."""
    return LadderBundle(
        **_scope("activation"),
        rungs={
            "naive": _survives_cell(),
            "split": _survives_cell(),
            "crossfit": _survives_cell(),
            # present + survives, but AUROC CI includes zero -> not robust.
            "blind": _survives_cell(significant=False),
            "shifted": _survives_cell(significant=False),
        },
    )


def scenario_distribution_robust() -> LadderBundle:
    """Survives crossfit AND robustly passes blind AND shifted (significant AUROC)."""
    return LadderBundle(
        **_scope("activation"),
        rungs={
            "naive": _survives_cell(),
            "split": _survives_cell(),
            "crossfit": _survives_cell(),
            "blind": _survives_cell(significant=True),
            "shifted": _survives_cell(significant=True),
        },
    )


def scenario_insufficient_protocol_a() -> LadderBundle:
    """Naive rung absent: no in-distribution floor -> verdict None, refused."""
    return LadderBundle(
        **_scope("activation"),
        rungs={
            # naive omitted entirely (absent rung).
            "split": _survives_cell(),
            "crossfit": _survives_cell(),
            "blind": _survives_cell(significant=True),
            "shifted": _survives_cell(significant=True),
        },
    )


def scenario_insufficient_protocol_b() -> LadderBundle:
    """Survives crossfit but blind/shifted absent: capped at in_distribution_only."""
    return LadderBundle(
        **_scope("activation"),
        rungs={
            "naive": _survives_cell(),
            "split": _survives_cell(),
            "crossfit": _survives_cell(),
            "blind": _unavailable_cell("pending"),
            "shifted": _unavailable_cell("not_run"),
        },
    )


def scenario_silver_label_asymmetry() -> LadderBundle:
    """distribution_robust but blind labels are silver -> human-validation claim forbidden."""
    return LadderBundle(
        **_scope(
            "activation",
            blind_label_provenance="silver_llm",
            inter_judge_kappa=0.887,
        ),
        rungs={
            "naive": _survives_cell(),
            "split": _survives_cell(),
            "crossfit": _survives_cell(),
            "blind": _survives_cell(significant=True),
            "shifted": _survives_cell(significant=True),
        },
    )


# --- discriminating scenarios that pin the exact survival/robustness rules --- #


def scenario_zero_delta_crossfit_not_robust() -> LadderBundle:
    """Crossfit has an EXACTLY-zero AUROC delta -> mixed, not survives (pins strict >0).

    Leaky rungs survive, so the in-distribution floor is leak_inflated. Under the
    correct strict ``> 0`` rule crossfit is ``mixed`` and does not survive; a
    ``>= 0`` mutation would read it as ``survives`` and wrongly upgrade the verdict
    to in_distribution_only.
    """
    return LadderBundle(
        **_scope("activation"),
        rungs={
            "naive": _survives_cell(),
            "split": _survives_cell(),
            "crossfit": _zero_delta_auc_cell(),  # mixed under >0, survives under >=0
            "blind": _fails_cell(significant=False),
            "shifted": _fails_cell(significant=False),
        },
    )


def scenario_leaky_naive_only() -> LadderBundle:
    """naive survives, split FAILS, crossfit fails -> leak_inflated (pins passes_leaky OR).

    With the correct ``survives(naive) OR survives(split)`` rule this is
    leak_inflated; an ``AND`` mutation would demand both leaky rungs and collapse
    this to no_value.
    """
    return LadderBundle(
        **_scope("activation"),
        rungs={
            "naive": _survives_cell(),
            "split": _fails_cell(),
            "crossfit": _fails_cell(),
            "blind": _fails_cell(significant=False),
            "shifted": _fails_cell(significant=False),
        },
    )


def scenario_leaky_split_only() -> LadderBundle:
    """naive FAILS, split survives, crossfit fails -> leak_inflated (reverse of above).

    The mirror of ``scenario_leaky_naive_only``: together they pin that the OR
    ranges over BOTH leaky rungs, not just one hardcoded rung.
    """
    return LadderBundle(
        **_scope("activation"),
        rungs={
            "naive": _fails_cell(),
            "split": _survives_cell(),
            "crossfit": _fails_cell(),
            "blind": _fails_cell(significant=False),
            "shifted": _fails_cell(significant=False),
        },
    )


def scenario_ood_significant_but_not_surviving() -> LadderBundle:
    """blind is significant but only MIXED -> not robust (pins survives in _robustly_passes_ood).

    Crossfit survives and shifted robustly passes, but blind is ``mixed`` with a
    significant AUROC CI. Under the correct rule (``survives AND significant``) blind
    is NOT robust, so the verdict is capped at in_distribution_only; a mutation that
    drops the ``survives`` conjunct would read blind as robust and (with shifted)
    wrongly award distribution_robust.
    """
    return LadderBundle(
        **_scope("activation"),
        rungs={
            "naive": _survives_cell(),
            "split": _survives_cell(),
            "crossfit": _survives_cell(),
            "blind": _mixed_cell(significant=True),  # significant but not surviving
            "shifted": _survives_cell(significant=True),
        },
    )


def scenario_blind_robust_shifted_not() -> LadderBundle:
    """blind robust, shifted NOT robust -> capped (pins shifted is required for the top tier).

    A mutation that drops the ``shifted`` conjunct from the OOD check would award
    distribution_robust off blind alone.
    """
    return LadderBundle(
        **_scope("activation"),
        rungs={
            "naive": _survives_cell(),
            "split": _survives_cell(),
            "crossfit": _survives_cell(),
            "blind": _survives_cell(significant=True),
            "shifted": _survives_cell(significant=False),  # survives but not significant
        },
    )


def scenario_shifted_robust_blind_not() -> LadderBundle:
    """shifted robust, blind NOT robust -> capped (pins blind is required for the top tier).

    The mirror of ``scenario_blind_robust_shifted_not``: together they pin that BOTH
    OOD rungs must robustly pass, not either one alone.
    """
    return LadderBundle(
        **_scope("activation"),
        rungs={
            "naive": _survives_cell(),
            "split": _survives_cell(),
            "crossfit": _survives_cell(),
            "blind": _survives_cell(significant=False),  # survives but not significant
            "shifted": _survives_cell(significant=True),
        },
    )


def scenario_human_audited_silver() -> LadderBundle:
    """distribution_robust, silver labels human-audited at kappa=0.62 (substantial).

    Exercises ``human_audit_kappa`` and the ``silver_llm_inter_judge_validated``
    provenance, pinning the audited residual wording (vs the 'human audit pending'
    wording of the un-audited silver case).
    """
    return LadderBundle(
        **_scope(
            "activation",
            blind_label_provenance="silver_llm_inter_judge_validated",
            inter_judge_kappa=0.901,
            human_audit_kappa=0.62,
        ),
        rungs={
            "naive": _survives_cell(),
            "split": _survives_cell(),
            "crossfit": _survives_cell(),
            "blind": _survives_cell(significant=True),
            "shifted": _survives_cell(significant=True),
        },
    )


# --------------------------------------------------------------------------- #
# expectations + runner
# --------------------------------------------------------------------------- #

# Each entry: scenario name -> (generator, expected_verdict, expected_status).
# expected_verdict is the SVC verdict (or None for the refused case).
SCENARIOS: dict[str, tuple[Callable[[], LadderBundle], str | None, str]] = {
    "no_value": (scenario_no_value, "no_value", "ok"),
    "leak_inflated": (scenario_leak_inflated, "leak_inflated", "ok"),
    "in_distribution_only": (scenario_in_distribution_only, "in_distribution_only", "ok"),
    "distribution_robust": (scenario_distribution_robust, "distribution_robust", "ok"),
    "insufficient_protocol_a": (
        scenario_insufficient_protocol_a,
        None,
        INSUFFICIENT_PROTOCOL,
    ),
    "insufficient_protocol_b": (
        scenario_insufficient_protocol_b,
        "in_distribution_only",
        INSUFFICIENT_PROTOCOL,
    ),
    "silver_label_asymmetry": (
        scenario_silver_label_asymmetry,
        "distribution_robust",
        "ok",
    ),
    # --- discriminating scenarios (pin the exact rules against mutation) --- #
    "zero_delta_crossfit_not_robust": (
        scenario_zero_delta_crossfit_not_robust,
        "leak_inflated",
        "ok",
    ),
    "leaky_naive_only": (scenario_leaky_naive_only, "leak_inflated", "ok"),
    "leaky_split_only": (scenario_leaky_split_only, "leak_inflated", "ok"),
    "ood_significant_but_not_surviving": (
        scenario_ood_significant_but_not_surviving,
        "in_distribution_only",
        "ok",
    ),
    "blind_robust_shifted_not": (
        scenario_blind_robust_shifted_not,
        "in_distribution_only",
        "ok",
    ),
    "shifted_robust_blind_not": (
        scenario_shifted_robust_blind_not,
        "in_distribution_only",
        "ok",
    ),
    "human_audited_silver": (scenario_human_audited_silver, "distribution_robust", "ok"),
}

# The disallowed claim the silver-label asymmetry case must surface.
_SILVER_DISALLOWED = "claim validation against human-labeled blind data"

# Full expected claims triple for the four canonical, human-label, single-verdict
# scenarios. Asserting the WHOLE triple (not one substring) pins the claim text so a
# mutation that silently drops or reorders a licensed/forbidden claim is caught.
_EXPECTED_CLAIMS: dict[str, dict[str, list[str]]] = {
    "no_value": {
        "allowed": [
            "the signal does not beat the prompt-classifier baseline under any rung"
        ],
        "disallowed": ["claim the signal adds safety value"],
        "residual": [],
    },
    "leak_inflated": {
        "allowed": [
            "the signal's apparent value appears only under leakage-prone evaluation"
        ],
        "disallowed": [
            "claim the signal earns runtime weight",
            "report naive/split numbers as the signal's value",
        ],
        "residual": ["leaky-rung metrics overstate the signal's value"],
    },
    "in_distribution_only": {
        "allowed": [
            "the signal adds value under leakage-free in-distribution evaluation"
        ],
        "disallowed": [
            "claim the signal generalizes under distribution shift",
            "deploy an in-distribution-tuned threshold without shift testing",
        ],
        "residual": [
            "value does not robustly pass BOTH out-of-distribution rungs "
            "(blind random sample and held-out source); see per-rung diagnostics"
        ],
    },
    "distribution_robust": {
        "allowed": [
            "the signal adds value under leakage-free, blind, and shifted evaluation"
        ],
        "disallowed": [
            "claim causal sufficiency (out of scope for this axis; see SIEVE)"
        ],
        "residual": [],
    },
}

# The audited-silver residual wording the human_audited_silver case must surface
# (kappa=0.62 -> Landis-Koch "substantial").
_HUMAN_AUDITED_RESIDUAL = "human-audited at kappa=0.620 (substantial"


def check_scenario(name: str) -> tuple[bool, str]:
    """Run one rigged scenario; return (ok, detail) with a clear diff on mismatch."""
    generator, expected_verdict, expected_status = SCENARIOS[name]
    bundle = generator()
    result = audit(bundle)
    got_verdict = result["verdict"]
    got_status = result["status"]
    claims = result["claims"]

    problems: list[str] = []
    if got_verdict != expected_verdict:
        problems.append(f"verdict: expected {expected_verdict!r}, got {got_verdict!r}")
    if got_status != expected_status:
        problems.append(f"status: expected {expected_status!r}, got {got_status!r}")

    # Canonical single-verdict, human-label scenarios pin the FULL claims triple
    # (allowed/disallowed/residual), not a single substring.
    if name in _EXPECTED_CLAIMS:
        expected = _EXPECTED_CLAIMS[name]
        for bucket in ("allowed", "disallowed", "residual"):
            if claims[bucket] != expected[bucket]:
                problems.append(
                    f"{bucket} claims: expected {expected[bucket]!r}, got {claims[bucket]!r}"
                )

    if name == "silver_label_asymmetry":
        if not any(_SILVER_DISALLOWED in c for c in claims["disallowed"]):
            problems.append(
                f"missing disallowed claim {_SILVER_DISALLOWED!r}; got {claims['disallowed']!r}"
            )

    if name == "human_audited_silver":
        # Silver provenance still forbids the human-validated phrasing...
        if not any(_SILVER_DISALLOWED in c for c in claims["disallowed"]):
            problems.append(
                f"missing disallowed claim {_SILVER_DISALLOWED!r}; got {claims['disallowed']!r}"
            )
        # ...and the residual must report the human audit (not 'pending').
        if any("human audit pending" in c for c in claims["residual"]):
            problems.append(
                f"residual still says 'human audit pending'; got {claims['residual']!r}"
            )
        if not any(_HUMAN_AUDITED_RESIDUAL in c for c in claims["residual"]):
            problems.append(
                f"missing audited residual {_HUMAN_AUDITED_RESIDUAL!r}; "
                f"got {claims['residual']!r}"
            )

    if problems:
        return False, "; ".join(problems)
    detail = f"verdict={got_verdict!r} status={got_status!r}"
    return True, detail


def run_selftest(verbose: bool = True) -> bool:
    """Run all rigged scenarios; raise AssertionError with a diff on any mismatch.

    Returns True when every scenario returns exactly its rigged verdict. Mirrors
    SIEVE's selftest semantics: a failure names the scenario, the expected, and
    the got.
    """
    failures: list[tuple[str, str]] = []
    for name in SCENARIOS:
        ok, detail = check_scenario(name)
        if verbose:
            status = "PASS" if ok else "FAIL"
            print(f"[ramp-audit] {status}  {name:26s} -> {detail}")
        if not ok:
            failures.append((name, detail))
    if failures:
        diff = "\n".join(f"  {name}: {detail}" for name, detail in failures)
        raise AssertionError(
            f"ramp-audit selftest FAILED ({len(failures)}/{len(SCENARIOS)} scenarios):\n{diff}"
        )
    if verbose:
        print(
            f"[ramp-audit] selftest passed: {len(SCENARIOS)}/{len(SCENARIOS)} "
            "rigged scenarios returned the rigged verdict"
        )
    return True
