"""Signal Validity Card — the scoped, caveat-bound, reproducible output.

Conforms to ``docs/signal-validity-card-spec-v0_1.md``. This is RAMP-native:
ZERO dependency on the SIEVE codebase. The two projects interoperate by the
vendored spec, not by a shared import. SIEVE emits ``causal_sufficiency`` cards;
this emits ``evaluation_robustness`` cards; a downstream system card can cite
one of each for the same signal.

Ported from ``scripts/emit_signal_validity_card.py``. The one deliberate
difference: the RAMP-specific defaults (``signal_description``, ``front_door``,
``probe_kind``) are read from the ``LadderBundle`` scope rather than hardcoded
in the emitter, so the package is signal-agnostic. The card *shape*, hashes, and
claim text are unchanged.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .bundle import LadderBundle
from .engine import _kappa_descriptor  # noqa: F401 (kept importable for parity with source)

CARD_VERSION = "0.1"
AXIS = "evaluation_robustness"
VERDICT_VOCABULARY = ["no_value", "leak_inflated", "in_distribution_only", "distribution_robust"]
INSUFFICIENT_PROTOCOL = "insufficient_protocol"


def _canonical_hash(obj: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
    )


def build_card(bundle: LadderBundle, verdict: dict[str, Any]) -> dict[str, Any]:
    """Assemble the SVC envelope from a bundle and the engine's verdict dict.

    ``verdict`` is the output of ``engine.audit(bundle)``: it carries the
    cross-rung verdict/status/claims and the per-rung survival table.

    Note: ``config_hash``/``inputs_hash`` are canonical hashes of ``bundle.config``
    / ``bundle.inputs`` as-is; for cross-path hash interop with the study emitter
    those dicts must carry the canonical keys (see ``bundle`` module docstring).
    """
    cells = verdict["survival_table"]
    diagnostics = {
        "per_rung": {
            rung: {
                "verdict": cells.get(rung, {}).get("verdict"),
                "auc_delta": cells.get(rung, {}).get("auc_delta"),
                "auc_significant": cells.get(rung, {}).get("auc_significant"),
            }
            for rung in ("naive", "split", "crossfit", "blind", "shifted")
        },
        "decision_reasons": verdict["reasons"],
    }
    return {
        "card_version": CARD_VERSION,
        "axis": AXIS,
        "verdict_vocabulary": VERDICT_VOCABULARY,
        "scope": {
            "signal": bundle.signal,
            "signal_description": bundle.signal_description,
            "target_model": bundle.target_model,
            "n": bundle.n,
            "axis_specific": {
                "rungs": list(diagnostics["per_rung"].keys()),
                "blind_label_provenance": bundle.blind_label_provenance,
                "inter_judge_kappa": bundle.inter_judge_kappa,
                "human_audit_kappa": bundle.human_audit_kappa,
                "blind_label_rubric": bundle.blind_label_rubric,
                "front_door": bundle.front_door,
                "probe_kind": bundle.probe_kind,
            },
        },
        "verdict": verdict["verdict"],
        "status": verdict["status"],
        "diagnostics": diagnostics,
        "allowed_claims": verdict["claims"]["allowed"],
        "disallowed_claims": verdict["claims"]["disallowed"],
        "residual_risks": verdict["claims"]["residual"],
        "protocol_version": bundle.protocol_version,
        "config_hash": _canonical_hash(bundle.config),
        "inputs_hash": _canonical_hash(bundle.inputs),
        "rerun_command": bundle.rerun_command,
        "preregistration": bundle.preregistration,
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
