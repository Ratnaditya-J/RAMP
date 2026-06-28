"""ramp.audit - robustness validity checks for RAMP safety signals.

A pure, installable auditor that turns a *ladder bundle* (the recorded evidence
for one signal under the naive/split/crossfit/blind/shifted evaluation ladder)
into a Signal Validity Card on the ``evaluation_robustness`` axis. The verdict
logic is ported verbatim from the study scripts (``evaluate_signal_survival_ladder.py``,
``emit_signal_validity_card.py``) so a card matches what the study produced.

A bundle can be built from already-recorded metrics
(``LadderBundle.from_survival_report`` / ``from_dict``) or re-derived from raw
per-row labels and scores (``LadderBundle.from_raw_scores``, which recomputes
AUROC/F1 and a seeded paired bootstrap via :mod:`ramp.audit.stats`).

Sibling to SIEVE (``~/sieve-audit``): same envelope (the vendored Signal Validity
Card spec), different axis. No model, no GPU, no network — numpy (only on the
raw-scores path) + stdlib.

See ``docs/signal-validity-card-spec-v0_1.md``.
"""
from ramp.audit import stats
from ramp.audit.bundle import LadderBundle
from ramp.audit.card import build_card, render_markdown
from ramp.audit.engine import (
    INSUFFICIENT_PROTOCOL,
    VERDICT_VOCABULARY,
    audit,
    cell_verdict,
    derive_robustness_verdict,
    survival_table,
)
from ramp.audit.selftest import SCENARIOS, run_selftest

__version__ = "0.1.0"

__all__ = [
    "INSUFFICIENT_PROTOCOL",
    "SCENARIOS",
    "VERDICT_VOCABULARY",
    "LadderBundle",
    "audit",
    "build_card",
    "cell_verdict",
    "derive_robustness_verdict",
    "render_markdown",
    "run_selftest",
    "stats",
    "survival_table",
    "__version__",
]
