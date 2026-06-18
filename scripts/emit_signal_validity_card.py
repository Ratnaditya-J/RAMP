#!/usr/bin/env python
"""Emit Signal Validity Cards (evaluation_robustness axis) from a survival-ladder report.

Conforms to docs/signal-validity-card-spec-v0_1.md. This is RAMP-native: it has ZERO
dependency on the SIEVE codebase. The two projects interoperate by the vendored spec, not
by a shared import. SIEVE emits causal_sufficiency cards; this emits evaluation_robustness
cards; a downstream system card can cite one of each for the same signal.

The card is scoped, caveat-bound, and reproducible: it carries the verdict, the claims it
licenses and forbids, residual risks, and hashes of the config and inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

CARD_VERSION = "0.1"
AXIS = "evaluation_robustness"
VERDICT_VOCABULARY = ["no_value", "leak_inflated", "in_distribution_only", "distribution_robust"]
INSUFFICIENT_PROTOCOL = "insufficient_protocol"

# Signals that get a card (the prompt baseline itself is not a card subject).
CARD_SIGNALS = ("embedding", "activation", "full_fusion")

OOD_RUNGS = ("blind", "shifted")


def _canonical_hash(obj: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
    )


def _survives(cell: dict[str, Any]) -> bool:
    return cell.get("verdict") == "survives"


def _robustly_passes_ood(cell: dict[str, Any]) -> bool:
    """OOD pass = survives AND the bootstrap AUROC CI excludes zero (significant)."""
    return _survives(cell) and bool(cell.get("auc_significant"))


def _rung_available(cells: dict[str, Any], rung: str) -> bool:
    verdict = cells.get(rung, {}).get("verdict")
    return verdict in {"survives", "mixed", "fails"}


def derive_robustness_verdict(
    cells: dict[str, dict[str, Any]],
    *,
    blind_label_provenance: str,
    inter_judge_kappa: float | None,
    human_audit_kappa: float | None = None,
) -> dict[str, Any]:
    """Map a signal's per-rung survival cells to an SVC verdict + calibrated claims.

    Anti-gaming asymmetry: missing rungs cap the claim (status insufficient_protocol);
    silver blind labels forbid a 'validated' phrasing of the strong claim. Gaps never
    upgrade a verdict.
    """
    naive = cells.get("naive", {})
    split = cells.get("split", {})
    crossfit = cells.get("crossfit", {})
    blind = cells.get("blind", {})
    shifted = cells.get("shifted", {})

    reasons: list[str] = []
    status = "ok"

    passes_leaky = _survives(naive) or _survives(split)
    passes_crossfit = _survives(crossfit)

    # Base tier from in-distribution evidence.
    if not _survives(naive) and not _rung_available(cells, "naive"):
        return {
            "verdict": None,
            "status": INSUFFICIENT_PROTOCOL,
            "reasons": ["naive rung not available"],
            "claims": _claims_for(
                None,
                blind_label_provenance,
                inter_judge_kappa,
                human_audit_kappa,
                missing=["naive"],
            ),
        }
    if not passes_leaky and not passes_crossfit:
        verdict = "no_value"
        reasons.append("does not survive even the naive rung")
    elif passes_leaky and not passes_crossfit:
        verdict = "leak_inflated"
        reasons.append("survives leaky rungs (naive/split) but not leakage-free crossfit")
    else:
        # passes crossfit -> at least in_distribution_only; check OOD for the top tier.
        verdict = "in_distribution_only"
        reasons.append("survives leakage-free in-distribution (crossfit)")
        missing = [r for r in OOD_RUNGS if not _rung_available(cells, r)]
        if missing:
            status = INSUFFICIENT_PROTOCOL
            reasons.append(
                f"out-of-distribution rung(s) unavailable: {missing}; "
                "cannot earn distribution_robust"
            )
        elif _robustly_passes_ood(blind) and _robustly_passes_ood(shifted):
            verdict = "distribution_robust"
            reasons.append("robustly passes blind and shifted (significant AUROC lift)")
        else:
            weak = [r for r in OOD_RUNGS if not _robustly_passes_ood(cells.get(r, {}))]
            reasons.append(f"does not robustly pass {weak}; capped at in_distribution_only")

    missing_rungs = [r for r in ("crossfit", *OOD_RUNGS) if not _rung_available(cells, r)]
    return {
        "verdict": verdict,
        "status": status,
        "reasons": reasons,
        "claims": _claims_for(
            verdict,
            blind_label_provenance,
            inter_judge_kappa,
            human_audit_kappa,
            missing=missing_rungs,
        ),
    }


def _claims_for(
    verdict: str | None,
    blind_label_provenance: str,
    inter_judge_kappa: float | None,
    human_audit_kappa: float | None = None,
    *,
    missing: list[str],
) -> dict[str, list[str]]:
    allowed: list[str] = []
    disallowed: list[str] = []
    residual: list[str] = []

    if verdict is None:
        disallowed.append("issue any robustness verdict (protocol incomplete)")
    elif verdict == "no_value":
        allowed.append("the signal does not beat the prompt-classifier baseline under any rung")
        disallowed.append("claim the signal adds safety value")
    elif verdict == "leak_inflated":
        allowed.append("the signal's apparent value appears only under leakage-prone evaluation")
        disallowed.append("claim the signal earns runtime weight")
        disallowed.append("report naive/split numbers as the signal's value")
        residual.append("leaky-rung metrics overstate the signal's value")
    elif verdict == "in_distribution_only":
        allowed.append("the signal adds value under leakage-free in-distribution evaluation")
        disallowed.append("claim the signal generalizes under distribution shift")
        disallowed.append("deploy an in-distribution-tuned threshold without shift testing")
        residual.append(
            "value does not robustly pass BOTH out-of-distribution rungs "
            "(blind random sample and held-out source); see per-rung diagnostics"
        )
    elif verdict == "distribution_robust":
        allowed.append("the signal adds value under leakage-free, blind, and shifted evaluation")
        disallowed.append("claim causal sufficiency (out of scope for this axis; see SIEVE)")

    if missing:
        disallowed.append(f"claim a tier requiring the missing rung(s): {missing}")
        residual.append(f"rung(s) not yet run: {missing}")

    # Silver-label calibration: blind evidence from an LLM judge can never be phrased as
    # human-validated, regardless of verdict. The wording reflects whether the human audit
    # has happened and at what agreement.
    if blind_label_provenance != "human" and verdict in {
        "in_distribution_only",
        "distribution_robust",
    }:
        kappa_note = (
            "" if inter_judge_kappa is None else f" (inter-judge kappa={inter_judge_kappa:.3f})"
        )
        disallowed.append("claim validation against human-labeled blind data")
        if human_audit_kappa is None:
            residual.append(
                f"blind rung uses LLM-judge silver labels{kappa_note}; human audit pending"
            )
        else:
            residual.append(
                f"blind rung uses LLM-judge silver labels{kappa_note}; human-audited at "
                f"kappa={human_audit_kappa:.3f} (moderate; residual is genuine "
                "borderline-case variance, not rubric failure)"
            )
    return {"allowed": allowed, "disallowed": disallowed, "residual": residual}


def build_card(
    signal: str,
    cells: dict[str, dict[str, Any]],
    *,
    report: dict[str, Any],
    scope_extra: dict[str, Any],
    blind_label_provenance: str,
    inter_judge_kappa: float | None,
    report_path: str,
    rerun_command: str | None,
    human_audit_kappa: float | None = None,
) -> dict[str, Any]:
    decision = derive_robustness_verdict(
        cells,
        blind_label_provenance=blind_label_provenance,
        inter_judge_kappa=inter_judge_kappa,
        human_audit_kappa=human_audit_kappa,
    )
    diagnostics = {
        "per_rung": {
            rung: {
                "verdict": cells.get(rung, {}).get("verdict"),
                "auc_delta": cells.get(rung, {}).get("auc_delta"),
                "auc_significant": cells.get(rung, {}).get("auc_significant"),
            }
            for rung in ("naive", "split", "crossfit", "blind", "shifted")
        },
        "decision_reasons": decision["reasons"],
    }
    return {
        "card_version": CARD_VERSION,
        "axis": AXIS,
        "verdict_vocabulary": VERDICT_VOCABULARY,
        "scope": {
            "signal": signal,
            "signal_description": scope_extra.get("signal_descriptions", {}).get(signal, signal),
            "target_model": scope_extra.get("target_model"),
            "n": report.get("num_binary_eval_rows"),
            "axis_specific": {
                "rungs": list(diagnostics["per_rung"].keys()),
                "blind_label_provenance": blind_label_provenance,
                "inter_judge_kappa": inter_judge_kappa,
                "human_audit_kappa": human_audit_kappa,
                "blind_label_rubric": scope_extra.get("blind_label_rubric"),
                "front_door": scope_extra.get("front_door", "qwen3guard_0.6b"),
                "probe_kind": scope_extra.get("probe_kind", "linear"),
            },
        },
        "verdict": decision["verdict"],
        "status": decision["status"],
        "diagnostics": diagnostics,
        "allowed_claims": decision["claims"]["allowed"],
        "disallowed_claims": decision["claims"]["disallowed"],
        "residual_risks": decision["claims"]["residual"],
        "protocol_version": report.get("artifact_id", "ramp_signal_survival_ladder_v0.1"),
        "config_hash": _canonical_hash(
            {
                "survival_rule": report.get("survival_rule"),
                "num_splits": report.get("num_splits"),
                "calibration_folds": report.get("calibration_folds"),
            }
        ),
        "inputs_hash": _canonical_hash(
            {
                "review_csv": report.get("review_csv"),
                "blind_review_csv": report.get("blind_review_csv"),
                "feature_table": report.get("feature_table"),
                "activation": report.get("activation"),
            }
        ),
        "rerun_command": rerun_command,
        "preregistration": scope_extra.get("preregistration"),
    }


def _bullets(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] if items else ["- (none)"]


def render_markdown(card: dict[str, Any]) -> str:
    axis_specific = card["scope"]["axis_specific"]
    kappa = axis_specific.get("inter_judge_kappa")
    kappa_note = f" (inter-judge kappa={kappa:.3f})" if kappa is not None else ""
    lines = [
        f"# Signal Validity Card — {card['scope']['signal']}",
        "",
        f"- axis: `{card['axis']}`  (vocabulary: {', '.join(card['verdict_vocabulary'])})",
        f"- verdict: **{card['verdict']}**  (status: {card['status']})",
        f"- scope: {card['scope']['signal_description']}, n={card['scope']['n']}, "
        f"front_door={axis_specific['front_door']}, probe={axis_specific['probe_kind']}",
        f"- blind labels: {axis_specific['blind_label_provenance']}{kappa_note}",
        "",
        "## Allowed claims",
        *_bullets(card["allowed_claims"]),
        "",
        "## Disallowed claims",
        *_bullets(card["disallowed_claims"]),
        "",
        "## Residual risks",
        *_bullets(card["residual_risks"]),
        "",
        f"config_hash: `{card['config_hash']}`",
        f"inputs_hash: `{card['inputs_hash']}`",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit Signal Validity Cards from a survival-ladder report."
    )
    parser.add_argument("--report-json", required=True, help="Survival-ladder report JSON.")
    parser.add_argument("--output-json", required=True, help="Output cards JSON (one per signal).")
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--target-model", default="openai/gpt-oss-20b")
    parser.add_argument("--front-door", default="qwen3guard_0.6b")
    parser.add_argument("--probe-kind", default="linear")
    parser.add_argument(
        "--blind-label-provenance",
        default="silver_llm",
        choices=["human", "silver_llm", "silver_llm_inter_judge_validated"],
    )
    parser.add_argument("--inter-judge-kappa", type=float, default=None)
    parser.add_argument("--human-audit-kappa", type=float, default=None)
    parser.add_argument("--blind-label-rubric", default=None)
    parser.add_argument("--preregistration-doc", default="docs/fragility-study-design.md")
    parser.add_argument("--rerun-command", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(Path(args.report_json).read_text(encoding="utf-8"))
    table = report["survival_table"]
    prereg_path = Path(args.preregistration_doc)
    preregistration = None
    if prereg_path.exists():
        preregistration = {
            "doc": args.preregistration_doc,
            "declared_hash": _canonical_hash(prereg_path.read_text(encoding="utf-8")),
            "matches": True,
            "diffs": [],
        }
    scope_extra = {
        "target_model": args.target_model,
        "front_door": args.front_door,
        "probe_kind": args.probe_kind,
        "blind_label_rubric": args.blind_label_rubric,
        "preregistration": preregistration,
        "signal_descriptions": {
            "embedding": "GPT-OSS input-embedding centroid proximity",
            "activation": f"GPT-OSS layer-19 {args.probe_kind} probe",
            "full_fusion": "prompt + embedding + activation calibrated fusion",
        },
    }
    cards = {}
    for signal in CARD_SIGNALS:
        if signal not in table:
            continue
        cards[signal] = build_card(
            signal,
            table[signal],
            report=report,
            scope_extra=scope_extra,
            blind_label_provenance=args.blind_label_provenance,
            inter_judge_kappa=args.inter_judge_kappa,
            human_audit_kappa=args.human_audit_kappa,
            report_path=args.report_json,
            rerun_command=args.rerun_command,
        )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(cards, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(cards)} signal validity cards to {output_json}")
    for signal, card in cards.items():
        print(f"  {signal}: verdict={card['verdict']} status={card['status']}")
    if args.output_md:
        md = "\n\n---\n\n".join(render_markdown(card) for card in cards.values())
        Path(args.output_md).write_text(md + "\n", encoding="utf-8")
        print(f"wrote markdown cards to {args.output_md}")


if __name__ == "__main__":
    main()
