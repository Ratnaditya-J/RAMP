"""The ladder bundle: the serialized evidence a RAMP robustness audit consumes.

This mirrors SIEVE's ``EvidenceBundle`` (``~/sieve-audit``): the core auditor
never touches a model. A study script runs the signal-survival ladder (probe
training, calibration, evaluation under each rung) and serializes *what
happened* here. The core then audits the **evidence**, so the verdict logic is
reproducible, GPU-free, and signal-agnostic — and anyone can re-run the audit
from the bundle without the 60 study scripts.

A ``LadderBundle`` is the evidence for ONE signal under test (e.g. the layer-19
activation probe). It carries:

- scope: which signal, on which model, with what n and label provenance;
- config: the survival rule and protocol parameters (for the config hash);
- inputs: input provenance (for the inputs hash);
- rungs: per-rung candidate vs prompt-only-baseline metrics, plus the paired
  bootstrap for the out-of-distribution rungs (``blind``, ``shifted``).

Each rung is ``{available, candidate:{auc,f1,recall,fpr}, baseline:{...},
bootstrap?:{auc:{delta,ci95,significant}, f1:{...}}}``. An absent rung, or one
with ``available=False``, is "not run" and (per the anti-gaming asymmetry) can
only cap a claim, never earn one.

Serialization is a single small JSON file, so bundles can be committed,
diffed, and hashed.

Three construction paths, with different provenance guarantees:

- ``from_dict`` / ``from_survival_report`` build a bundle from metrics ALREADY
  recorded by a study run; the audit is then a deterministic transform of those
  recorded numbers (including the recorded bootstrap significance) — it does not
  recompute anything.
- ``from_raw_scores`` is the honest end-to-end path: it takes raw per-row labels
  and scores and RE-DERIVES the rung metrics (``stats._fast_auc`` / ``_f1_at``)
  and the OOD significance (``stats.paired_bootstrap``, seeded), so the auditor
  can stand behind the significance it reports rather than trusting a recorded
  flag.

Note: ``config`` and ``inputs`` are free-form, but to interoperate with the study
emitter's hashes (so a card built either way has the same ``config_hash`` /
``inputs_hash``) they must carry the canonical keys — ``config``:
``survival_rule``/``num_splits``/``calibration_folds``; ``inputs``:
``review_csv``/``blind_review_csv``/``feature_table``/``activation``.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import stats  # pure (hashlib+typing only); numpy is passed in, never imported here

# The five rungs, in protocol order (increasingly honest evaluation).
RUNG_ORDER = ("naive", "split", "crossfit", "blind", "shifted")

# Out-of-distribution rungs: single-evaluation, so significance comes from the
# paired bootstrap rather than a stdev across splits.
OOD_RUNGS = ("blind", "shifted")

# Valid blind-label provenances (mirrors emit_signal_validity_card.py choices).
BLIND_LABEL_PROVENANCES = ("human", "silver_llm", "silver_llm_inter_judge_validated")

# Metric keys carried per arm in a rung cell.
_METRIC_KEYS = ("auc", "f1", "recall", "fpr")

# Mapping from the survival-ladder report's metric names to the bundle's metric
# keys (the report uses "false_positive_rate"; the bundle uses "fpr").
_REPORT_METRIC_MAP = {
    "auc": "auc",
    "f1": "f1",
    "recall": "recall",
    "fpr": "false_positive_rate",
}

# The survival-ladder report keys candidate metrics by condition, not signal.
# This maps a signal under test to its calibrated-combination condition name and
# is the same mapping as evaluate_signal_survival_ladder.SIGNAL_CONDITIONS.
SIGNAL_CONDITIONS = {
    "prompt": "prompt_only_calibrated",
    "embedding": "prompt_embedding_calibrated",
    "activation": "prompt_activation_calibrated",
    "full_fusion": "prompt_embedding_activation_calibrated",
}

# The prompt-only condition is always the baseline a candidate is measured against.
BASELINE_CONDITION = "prompt_only_calibrated"

# Metric keys that, when present in an available rung arm, must be finite numerics.
# auc & f1 are required (the survival rule reads them); recall & fpr are optional but,
# if present, must also be numeric (not string/NaN).
_REQUIRED_METRIC_KEYS = ("auc", "f1")
_OPTIONAL_METRIC_KEYS = ("recall", "fpr")


def _is_finite_number(value: Any) -> bool:
    """True iff ``value`` is a real int|float (not bool) and finite (not NaN/inf)."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _check_metrics(rung_name: str, arm: str, metrics: dict[str, Any]) -> None:
    """Reject an arm whose metrics are not finite numbers (string/NaN/missing).

    ``auc`` and ``f1`` are required and must be finite numerics; ``recall``/``fpr``
    may be absent but, when present, must also be finite numerics. A bare ``{}`` is
    rejected because the engine would read ``None`` deltas as ``incomplete`` rather
    than detecting the corruption.
    """
    for key in _REQUIRED_METRIC_KEYS:
        if key not in metrics:
            raise ValueError(
                f"available rung {rung_name!r} {arm!r} must carry a numeric {key!r}"
            )
        if not _is_finite_number(metrics[key]):
            raise ValueError(
                f"available rung {rung_name!r} {arm!r} {key!r} must be a finite number; "
                f"got {metrics[key]!r}"
            )
    for key in _OPTIONAL_METRIC_KEYS:
        if key in metrics and metrics[key] is not None and not _is_finite_number(metrics[key]):
            raise ValueError(
                f"available rung {rung_name!r} {arm!r} {key!r} must be a finite number or "
                f"absent; got {metrics[key]!r}"
            )


def _check_bootstrap_entry(rung_name: str, metric: str, entry: Any) -> None:
    """Reject a malformed bootstrap entry for one OOD metric.

    Requires a dict with a bool ``significant`` and a 2-element ``ci95`` of finite
    numerics, and (anti-gaming, fix #4) requires ``significant`` to agree with the
    CI: ``significant == (lo > 0 or hi < 0)``. A bundle that claims significance the
    CI does not support — or hides significance the CI does support — is rejected so
    the auditor cannot be gamed by an inconsistent recorded bootstrap.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"rung {rung_name!r} bootstrap {metric!r} must be a dict")
    significant = entry.get("significant")
    if not isinstance(significant, bool):
        raise ValueError(
            f"rung {rung_name!r} bootstrap {metric!r} 'significant' must be a bool; "
            f"got {significant!r}"
        )
    ci95 = entry.get("ci95")
    if not isinstance(ci95, (list, tuple)) or len(ci95) != 2:
        raise ValueError(
            f"rung {rung_name!r} bootstrap {metric!r} 'ci95' must be a 2-element list; "
            f"got {ci95!r}"
        )
    if not _is_finite_number(ci95[0]) or not _is_finite_number(ci95[1]):
        raise ValueError(
            f"rung {rung_name!r} bootstrap {metric!r} 'ci95' bounds must be finite numbers; "
            f"got {ci95!r}"
        )
    ci_excludes_zero = ci95[0] > 0 or ci95[1] < 0
    if significant != ci_excludes_zero:
        raise ValueError(
            f"rung {rung_name!r} bootstrap {metric!r} 'significant'={significant} disagrees "
            f"with ci95={list(ci95)} (CI excludes zero: {ci_excludes_zero}); refusing a "
            "self-inconsistent bootstrap"
        )


@dataclass
class LadderBundle:
    """Everything a RAMP robustness audit needs for ONE signal, recorded by a study run."""

    # --- scope (copied into the validity card verbatim) ---
    signal: str                      # signal under test, e.g. "activation"
    signal_description: str          # human-readable, e.g. "GPT-OSS layer-19 linear probe"
    target_model: str                # e.g. "openai/gpt-oss-20b"
    n: int                           # number of binary eval rows

    # --- label provenance (drives the silver-label calibration of claims) ---
    blind_label_provenance: str = "human"      # one of BLIND_LABEL_PROVENANCES
    inter_judge_kappa: float | None = None     # inter-judge agreement on silver labels
    human_audit_kappa: float | None = None     # agreement of a human audit of silver labels

    # --- reproducibility inputs ---
    # config feeds the config hash; inputs feeds the inputs hash. Free-form, but
    # the canonical keys mirror the survival-ladder report fields.
    config: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)

    # --- evidence: per-rung candidate vs baseline metrics ---
    # rungs[rung_name] = {
    #   "available": bool,
    #   "candidate": {"auc","f1","recall","fpr"},   # may hold None metrics
    #   "baseline":  {"auc","f1","recall","fpr"},
    #   "bootstrap": {"auc": {...}, "f1": {...}},    # OOD rungs only
    # }
    rungs: dict[str, dict[str, Any]] = field(default_factory=dict)

    # --- scope extras that parameterize the card (no RAMP-specific hardcoding) ---
    front_door: str | None = None              # e.g. "qwen3guard_0.6b"
    probe_kind: str | None = None              # e.g. "linear" | "mlp"
    blind_label_rubric: str | None = None
    preregistration: dict[str, Any] | None = None
    protocol_version: str = "ramp_signal_survival_ladder_v0.1"
    rerun_command: str | None = None

    bundle_version: str = "0.1"

    # ---- validation ----

    def validate(self) -> None:
        """Reject structurally invalid bundles before the engine reads them.

        Conservative and total: a malformed rung cell, a non-numeric metric, a
        corrupt or self-inconsistent bootstrap, a bad provenance/kappa/n type, or
        an unknown rung name is a hard ``ValueError`` here so the engine never
        silently mis-reads evidence or crashes on garbage. (The anti-gaming
        *asymmetry* — missing rungs capping the claim — lives in the engine; this
        rejects corrupt input and self-inconsistent significance claims.)
        """
        if not self.signal:
            raise ValueError("bundle must name the signal under test")
        # n must be a real (non-bool) int >= 0; bool is an int subclass, exclude it.
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 0:
            raise ValueError(f"n must be a non-negative (non-bool) integer; got {self.n!r}")
        if self.blind_label_provenance not in BLIND_LABEL_PROVENANCES:
            raise ValueError(
                f"blind_label_provenance must be one of {BLIND_LABEL_PROVENANCES}; "
                f"got {self.blind_label_provenance!r}"
            )
        for field_name in ("inter_judge_kappa", "human_audit_kappa"):
            value = getattr(self, field_name)
            if value is not None and not _is_finite_number(value):
                raise ValueError(
                    f"{field_name} must be None or a finite number; got {value!r}"
                )
        for rung_name, cell in self.rungs.items():
            if rung_name not in RUNG_ORDER:
                raise ValueError(f"unknown rung {rung_name!r}; expected one of {RUNG_ORDER}")
            if not isinstance(cell, dict):
                raise ValueError(f"rung {rung_name!r} must be a dict")
            if "available" not in cell:
                raise ValueError(f"rung {rung_name!r} must declare 'available'")
            # 'available' must be a real bool, not a truthy/falsy proxy (1, "yes", []).
            if not isinstance(cell["available"], bool):
                raise ValueError(
                    f"rung {rung_name!r} 'available' must be a bool; got {cell['available']!r}"
                )
            if not cell["available"]:
                continue
            for arm in ("candidate", "baseline"):
                arm_metrics = cell.get(arm)
                if not isinstance(arm_metrics, dict):
                    raise ValueError(
                        f"available rung {rung_name!r} must carry a {arm!r} metrics dict"
                    )
                _check_metrics(rung_name, arm, arm_metrics)
            bootstrap = cell.get("bootstrap")
            if bootstrap is not None:
                if not isinstance(bootstrap, dict):
                    raise ValueError(f"rung {rung_name!r} 'bootstrap' must be a dict")
                for metric in ("auc", "f1"):
                    if metric not in bootstrap:
                        raise ValueError(
                            f"rung {rung_name!r} bootstrap must carry an {metric!r} entry"
                        )
                    _check_bootstrap_entry(rung_name, metric, bootstrap[metric])

    # ---- (de)serialization ----

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LadderBundle:
        return cls(
            signal=d["signal"],
            signal_description=d.get("signal_description", d["signal"]),
            target_model=d.get("target_model", ""),
            n=int(d.get("n", 0)),
            blind_label_provenance=d.get("blind_label_provenance", "human"),
            inter_judge_kappa=d.get("inter_judge_kappa"),
            human_audit_kappa=d.get("human_audit_kappa"),
            config=dict(d.get("config", {})),
            inputs=dict(d.get("inputs", {})),
            rungs=dict(d.get("rungs", {})),
            front_door=d.get("front_door"),
            probe_kind=d.get("probe_kind"),
            blind_label_rubric=d.get("blind_label_rubric"),
            preregistration=d.get("preregistration"),
            protocol_version=d.get("protocol_version", "ramp_signal_survival_ladder_v0.1"),
            rerun_command=d.get("rerun_command"),
            bundle_version=d.get("bundle_version", "0.1"),
        )

    @classmethod
    def load(cls, path: str | Path) -> LadderBundle:
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ---- ingest an existing survival-ladder report ----

    @classmethod
    def from_survival_report(
        cls,
        report: dict[str, Any],
        signal: str,
        *,
        signal_description: str | None = None,
        target_model: str | None = None,
        blind_label_provenance: str = "human",
        inter_judge_kappa: float | None = None,
        human_audit_kappa: float | None = None,
        front_door: str | None = None,
        probe_kind: str | None = None,
        blind_label_rubric: str | None = None,
        preregistration: dict[str, Any] | None = None,
        rerun_command: str | None = None,
    ) -> LadderBundle:
        """Build a bundle for ``signal`` from a survival-ladder report JSON.

        Reads the report produced by ``scripts/evaluate_signal_survival_ladder.py``:
        per-rung candidate metrics come from ``rungs[rung].aggregate_holdout_metrics``
        (or ``row_weighted_holdout_metrics`` when present, e.g. the shifted rung —
        matching ``rung_mean``), and the OOD bootstrap comes from
        ``rungs[rung].bootstrap[condition]``. This lets existing study outputs be
        carded by the tool without re-running the ladder.

        This path copies the metrics and bootstrap significance the study ALREADY
        recorded; the audit is deterministic from those recorded numbers and does
        not recompute AUROC/F1 or the bootstrap. To re-derive significance from raw
        evidence, use :meth:`from_raw_scores`.
        """
        if signal not in SIGNAL_CONDITIONS:
            raise ValueError(
                f"unknown signal {signal!r}; expected one of {tuple(SIGNAL_CONDITIONS)}"
            )
        condition = SIGNAL_CONDITIONS[signal]
        rungs_report = report.get("rungs", {})
        rungs: dict[str, dict[str, Any]] = {}
        for rung_name in RUNG_ORDER:
            rung = rungs_report.get(rung_name, {"status": "not_run"})
            if rung.get("status") != "completed":
                rungs[rung_name] = {"available": False, "status": rung.get("status", "not_run")}
                continue
            cell: dict[str, Any] = {
                "available": True,
                "candidate": _metrics_from_rung(rung, condition),
                "baseline": _metrics_from_rung(rung, BASELINE_CONDITION),
            }
            bootstrap = rung.get("bootstrap", {}).get(condition)
            if bootstrap is not None:
                cell["bootstrap"] = {
                    "auc": {
                        "delta": bootstrap["auc"].get("delta"),
                        "ci95": bootstrap["auc"].get("ci95", [None, None]),
                        "significant": bool(bootstrap["auc"].get("significant", False)),
                    },
                    "f1": {
                        "delta": bootstrap["f1"].get("delta"),
                        "ci95": bootstrap["f1"].get("ci95", [None, None]),
                        "significant": bool(bootstrap["f1"].get("significant", False)),
                    },
                }
            rungs[rung_name] = cell

        return cls(
            signal=signal,
            signal_description=signal_description or signal,
            target_model=target_model or "",
            n=int(report.get("num_binary_eval_rows", 0)),
            blind_label_provenance=blind_label_provenance,
            inter_judge_kappa=inter_judge_kappa,
            human_audit_kappa=human_audit_kappa,
            config={
                "survival_rule": report.get("survival_rule"),
                "num_splits": report.get("num_splits"),
                "calibration_folds": report.get("calibration_folds"),
            },
            inputs={
                "review_csv": report.get("review_csv"),
                "blind_review_csv": report.get("blind_review_csv"),
                "feature_table": report.get("feature_table"),
                "activation": report.get("activation"),
            },
            rungs=rungs,
            front_door=front_door,
            probe_kind=probe_kind,
            blind_label_rubric=blind_label_rubric,
            preregistration=preregistration,
            protocol_version=report.get("artifact_id", "ramp_signal_survival_ladder_v0.1"),
            rerun_command=rerun_command,
        )

    # ---- build a bundle by re-deriving metrics from RAW per-row evidence ----

    @classmethod
    def from_raw_scores(
        cls,
        signal: str,
        raw_rungs: dict[str, dict[str, Any]],
        *,
        signal_description: str | None = None,
        target_model: str | None = None,
        seed: str = "ramp_audit_from_raw_scores",
        num_resamples: int = 2000,
        blind_label_provenance: str = "human",
        inter_judge_kappa: float | None = None,
        human_audit_kappa: float | None = None,
        config: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        front_door: str | None = None,
        probe_kind: str | None = None,
        blind_label_rubric: str | None = None,
        preregistration: dict[str, Any] | None = None,
        rerun_command: str | None = None,
    ) -> LadderBundle:
        """Build a bundle by RE-DERIVING rung metrics from raw per-row labels + scores.

        Unlike :meth:`from_survival_report` / :meth:`from_dict` (which copy metrics a
        study already recorded), this is the honest end-to-end path: it computes each
        rung's AUROC/F1 with ``stats._fast_auc`` / ``stats._f1_at`` and, for the
        out-of-distribution rungs, re-derives significance with the SEEDED
        ``stats.paired_bootstrap`` — so the auditor stands behind the significance it
        reports rather than trusting a recorded flag.

        ``raw_rungs[rung]`` is a dict with::

            {
              "labels": [0/1, ...],                 # binary ground truth, one per row
              "candidate_scores": [float, ...],     # the signal-under-test's scores
              "baseline_scores": [float, ...],      # prompt-only baseline's scores
              "candidate_threshold": float,         # threshold for candidate F1
              "baseline_threshold": float,          # threshold for baseline F1
            }

        Only rungs present in ``raw_rungs`` are marked available. For the OOD rungs
        (``blind``, ``shifted``) a paired bootstrap is run and its 95% CI / significance
        attached; ``num_resamples`` and ``seed`` make that reproducible. ``recall``/
        ``fpr`` are left ``None`` (the survival rule reads only AUROC and F1).
        """
        if signal not in SIGNAL_CONDITIONS:
            raise ValueError(
                f"unknown signal {signal!r}; expected one of {tuple(SIGNAL_CONDITIONS)}"
            )
        # numpy is imported lazily, inside this raw-evidence path only, so the core
        # auditor's import graph stays numpy-free until someone re-derives from raw.
        import numpy as np

        rungs: dict[str, dict[str, Any]] = {}
        n_rows = 0
        for rung_name in RUNG_ORDER:
            raw = raw_rungs.get(rung_name)
            if raw is None:
                continue
            labels = np.asarray(raw["labels"], dtype=np.int64)
            cand_scores = np.asarray(raw["candidate_scores"], dtype=np.float64)
            base_scores = np.asarray(raw["baseline_scores"], dtype=np.float64)
            cand_threshold = raw["candidate_threshold"]
            base_threshold = raw["baseline_threshold"]
            n_rows = max(n_rows, int(labels.size))

            cell: dict[str, Any] = {
                "available": True,
                "candidate": {
                    "auc": stats._fast_auc(np, labels, cand_scores),
                    "f1": stats._f1_at(np, labels, cand_scores, cand_threshold),
                    "recall": None,
                    "fpr": None,
                },
                "baseline": {
                    "auc": stats._fast_auc(np, labels, base_scores),
                    "f1": stats._f1_at(np, labels, base_scores, base_threshold),
                    "recall": None,
                    "fpr": None,
                },
            }
            if rung_name in OOD_RUNGS:
                boot = stats.paired_bootstrap(
                    np,
                    raw["labels"],
                    raw["baseline_scores"],
                    base_threshold,
                    [("candidate", raw["candidate_scores"], cand_threshold)],
                    num_resamples=num_resamples,
                    seed=f"{seed}:{rung_name}",
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
            rungs[rung_name] = cell

        bundle = cls(
            signal=signal,
            signal_description=signal_description or signal,
            target_model=target_model or "",
            n=n_rows,
            blind_label_provenance=blind_label_provenance,
            inter_judge_kappa=inter_judge_kappa,
            human_audit_kappa=human_audit_kappa,
            config=dict(config) if config else {},
            inputs=dict(inputs) if inputs else {},
            rungs=rungs,
            front_door=front_door,
            probe_kind=probe_kind,
            blind_label_rubric=blind_label_rubric,
            preregistration=preregistration,
            rerun_command=rerun_command,
        )
        bundle.validate()
        return bundle


def _metrics_from_rung(rung: dict[str, Any], condition: str) -> dict[str, float | None]:
    """Extract {auc,f1,recall,fpr} for one condition from a completed rung.

    Mirrors ``evaluate_signal_survival_ladder.rung_mean``: prefer the
    row-weighted holdout metrics when the rung reports them (shifted), else the
    mean over the aggregate holdout metrics.
    """
    row_weighted = rung.get("row_weighted_holdout_metrics")
    out: dict[str, float | None] = {}
    for key in _METRIC_KEYS:
        report_metric = _REPORT_METRIC_MAP[key]
        if row_weighted is not None:
            out[key] = row_weighted.get(condition, {}).get(report_metric)
        else:
            summary = rung.get("aggregate_holdout_metrics", {}).get(condition, {}).get(
                report_metric
            )
            out[key] = summary.get("mean") if summary else None
    return out
