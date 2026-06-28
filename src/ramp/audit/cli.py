"""RAMP robustness-audit command-line interface.

Sibling to ``sieve`` (``~/sieve-audit``):

- ``ramp audit --bundle bundle.json [--out card.json] [--md card.md]``: audit a
  ladder bundle and emit a Signal Validity Card (``evaluation_robustness`` axis).
- ``ramp audit-selftest``: run the rigged ground-truth scenarios and verify the
  engine returns exactly the rigged verdicts (the auditor auditing itself).

Wired in ``pyproject.toml`` as ``ramp-audit = "ramp.audit.cli:main"``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bundle import LadderBundle
from .card import build_card, render_markdown
from .engine import INSUFFICIENT_PROTOCOL, audit
from .selftest import SCENARIOS, run_selftest


def _cmd_audit(args: argparse.Namespace) -> int:
    # Load + audit can fail on a missing file, malformed JSON, or a structurally
    # invalid bundle (validate()). Surface these as a clean one-line error on stderr
    # and a non-zero exit, never a traceback -- the CLI is a user-facing surface.
    try:
        bundle = LadderBundle.load(args.bundle)
        verdict = audit(bundle)
    except FileNotFoundError as exc:
        print(f"[ramp-audit] error: bundle not found: {exc.filename or args.bundle}",
              file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"[ramp-audit] error: bundle is not valid JSON: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"[ramp-audit] error: {exc}", file=sys.stderr)
        return 1

    card = build_card(bundle, verdict)

    label = card["verdict"] if card["verdict"] is not None else card["status"]
    print(f"[ramp-audit] signal: {card['scope']['signal']}")
    print(f"[ramp-audit] verdict: {label} (status: {card['status']})")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")
        print(f"[ramp-audit] card JSON: {out_path}")
    if args.md:
        md_path = Path(args.md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(card) + "\n")
        print(f"[ramp-audit] card markdown: {md_path}")
    if not args.out and not args.md:
        print(json.dumps(card, indent=2, sort_keys=True))

    if card["status"] == INSUFFICIENT_PROTOCOL:
        print("[ramp-audit] protocol incomplete - claim is capped:")
        for reason in card["diagnostics"]["decision_reasons"]:
            print(f"  - {reason}")
    return 0


def _cmd_selftest(args: argparse.Namespace) -> int:
    try:
        run_selftest(verbose=True)
    except AssertionError as exc:
        print(str(exc))
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ramp-audit",
        description=(
            "RAMP robustness auditor - turn a ladder bundle into a Signal Validity Card "
            "(evaluation_robustness axis; see docs/signal-validity-card-spec-v0_1.md)."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    p_audit = sub.add_parser("audit", help="audit a ladder bundle and emit a validity card")
    p_audit.add_argument("--bundle", required=True, help="path to a ladder bundle JSON")
    p_audit.add_argument("--out", default=None, help="write the card JSON here")
    p_audit.add_argument("--md", default=None, help="write the card markdown here")
    p_audit.set_defaults(func=_cmd_audit)

    p_self = sub.add_parser(
        "audit-selftest",
        help=f"audit {len(SCENARIOS)} rigged ground-truth scenarios; verdicts must match",
    )
    p_self.set_defaults(func=_cmd_selftest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
