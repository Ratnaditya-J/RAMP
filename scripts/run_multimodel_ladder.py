#!/usr/bin/env python
"""Multi-model signal-survival ladder runner (local, NO GPU).

Given a target model's candidate-layer activation JSONLs, this runner:

  1. runs the signal-survival ladder once per candidate layer with the CHEAP
     rung subset (``naive,crossfit``) to read each layer's leakage-free
     in-distribution activation AUROC (the crossfit rung);
  2. selects the best layer by that crossfit activation AUROC (the
     leakage-free in-distribution signal);
  3. re-runs the FULL 5-rung ladder for the selected layer; and
  4. derives the activation-signal's Signal Validity Card verdict via the
     packaged auditor (``ramp.audit``) and emits the SVC + a per-layer
     activation-AUROC table.

It is ADDITIVE: it shells out to ``scripts/evaluate_signal_survival_ladder.py``
(unchanged) for the heavy lifting, then reads the recorded report. No model, no
GPU, no network — numpy is only pulled in transitively by the ladder script.

The point of the harness: swapping ``--activation`` to a new model's layer file
while keeping the prompt baseline / feature table / review CSVs fixed yields
*that model's* activation verdict on the SAME adaptive+blind evaluation. This is
how we test whether GPT-OSS-20b's ``leak_inflated`` activation verdict
generalizes to Llama-3.1-8B, Mistral-7B, and Phi-4.

Usage (sweep a model's layer dir, full pipeline)::

    PYTHONPATH=src .venv/bin/python scripts/run_multimodel_ladder.py \
        --model-name llama31_8b \
        --activations-dir .artifacts/multimodel/activations/llama31_8b \
        --output-dir .artifacts/multimodel/ladder/llama31_8b

Correctness gate (reproduce the canonical GPT-OSS layer_19 verdict)::

    PYTHONPATH=src .venv/bin/python scripts/run_multimodel_ladder.py --verify-gpt-oss

Layer files are discovered by the ``*.layer_<N>.jsonl`` convention emitted by
``scripts/extract_multimodel_activations.py``; pass explicit files with
``--activation-file`` (repeatable) to override discovery.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# This script imports the packaged auditor (ramp.audit); run with PYTHONPATH=src.
# It shells out to the ladder script with PYTHONPATH=src:scripts so the flat
# script-namespace imports in evaluate_signal_survival_ladder.py resolve.
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"
LADDER_SCRIPT = SCRIPTS_DIR / "evaluate_signal_survival_ladder.py"

# Canonical shared inputs that produced the GPT-OSS reference verdict
# (read from .artifacts/prompt_label_audit/ramp_signal_survival_ladder_v0_1.json).
# These hold the prompt baseline, the embedding signal, and the labels constant;
# only --activation changes between models.
DEFAULT_REVIEW_CSV = "data/reviewed/ramp_prompt_label_review_combined_v0_1_v0_2_v0_4_cleaned.csv"
DEFAULT_FEATURE_TABLE = (
    ".artifacts/cumulative_signal_eval/"
    "ramp_qwen3guard_prompt_internal_signal_feature_table_expanded_reviewed_activation_v0_1.jsonl"
)
DEFAULT_BLIND_REVIEW_CSV = "data/reviewed/ramp_blind_review_batch_v0_1.judge_labeled.csv"

# GPT-OSS-20b reference activation (layer 19) + the canonical report to match.
GPT_OSS_ACTIVATION = (
    ".artifacts/corrections/ramp_benchmark_comprehensive_v0.layer_19.taxonomy_corrected_v0_1.jsonl"
)
CANONICAL_REPORT = ".artifacts/prompt_label_audit/ramp_signal_survival_ladder_v0_1.json"

# The canonical per-rung activation verdict + SVC verdict (the correctness gate).
CANONICAL_ACTIVATION_PER_RUNG = {
    "naive": "survives",
    "split": "survives",
    "crossfit": "mixed",
    "blind": "mixed",
    "shifted": "fails",
}
CANONICAL_SVC_VERDICT = "leak_inflated"

RUNG_ORDER = ("naive", "split", "crossfit", "blind", "shifted")
SWEEP_RUNGS = "naive,crossfit"
FULL_RUNGS = "naive,split,crossfit,blind,shifted"

# Selection metric: leakage-free in-distribution activation AUROC = the crossfit
# rung's calibrated prompt+activation mean AUROC.
ACTIVATION_CONDITION = "prompt_activation_calibrated"
BASELINE_CONDITION = "prompt_only_calibrated"

LAYER_FILE_RE = re.compile(r"\.layer_(\d+)\.jsonl$")


# --------------------------------------------------------------------------- #
# layer-file discovery
# --------------------------------------------------------------------------- #
def discover_layer_files(activations_dir: Path) -> dict[int, Path]:
    """Map layer index -> JSONL path for every ``*.layer_<N>.jsonl`` in a dir."""
    found: dict[int, Path] = {}
    if not activations_dir.is_dir():
        raise FileNotFoundError(f"activations dir not found: {activations_dir}")
    for path in sorted(activations_dir.glob("*.layer_*.jsonl")):
        match = LAYER_FILE_RE.search(path.name)
        if match:
            found[int(match.group(1))] = path
    if not found:
        raise FileNotFoundError(
            f"no *.layer_<N>.jsonl files found in {activations_dir}; "
            "run scripts/extract_multimodel_activations.py first"
        )
    return found


def layer_of(path: Path) -> int:
    match = LAYER_FILE_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot parse layer index from filename: {path.name}")
    return int(match.group(1))


# --------------------------------------------------------------------------- #
# ladder subprocess
# --------------------------------------------------------------------------- #
def run_ladder(
    *,
    activation: Path,
    layer: int,
    rungs: str,
    output_json: Path,
    review_csv: str,
    feature_table: str,
    blind_review_csv: str | None,
    num_splits: int,
    output_md: Path | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Invoke evaluate_signal_survival_ladder.py and return the parsed report."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(LADDER_SCRIPT),
        "--review-csv",
        review_csv,
        "--feature-table",
        feature_table,
        "--activation",
        str(activation),
        "--layer",
        str(layer),
        "--rungs",
        rungs,
        "--num-splits",
        str(num_splits),
        "--output-json",
        str(output_json),
    ]
    if blind_review_csv and ("blind" in rungs.split(",")):
        cmd += ["--blind-review-csv", blind_review_csv]
    if output_md is not None:
        cmd += ["--output-md", str(output_md)]
    if extra_args:
        cmd += extra_args

    env_pythonpath = f"{SRC_DIR}:{SCRIPTS_DIR}"
    print(f"  -> ladder [{rungs}] layer={layer} activation={activation.name}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env={**_base_env(), "PYTHONPATH": env_pythonpath},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"ladder failed (layer {layer}, rungs {rungs}); see output above")
    return json.loads(output_json.read_text(encoding="utf-8"))


def _base_env() -> dict[str, str]:
    import os

    return dict(os.environ)


# --------------------------------------------------------------------------- #
# metric extraction
# --------------------------------------------------------------------------- #
def crossfit_activation_auroc(report: dict[str, Any]) -> dict[str, float | None]:
    """Pull the leakage-free in-distribution activation AUROC from a report.

    Returns the crossfit-rung mean AUROC for the calibrated prompt+activation
    condition and the prompt-only baseline, plus their delta. None if the rung
    did not complete.
    """
    crossfit = report.get("rungs", {}).get("crossfit", {})
    if crossfit.get("status") != "completed":
        return {"activation_auroc": None, "baseline_auroc": None, "delta": None}
    agg = crossfit.get("aggregate_holdout_metrics", {})
    act = agg.get(ACTIVATION_CONDITION, {}).get("auc", {}).get("mean")
    base = agg.get(BASELINE_CONDITION, {}).get("auc", {}).get("mean")
    delta = (act - base) if (act is not None and base is not None) else None
    return {"activation_auroc": act, "baseline_auroc": base, "delta": delta}


def per_rung_activation_verdicts(report: dict[str, Any]) -> dict[str, str]:
    """The activation signal's per-rung survival verdict from a full report."""
    table = report.get("survival_table", {}).get("activation", {})
    return {rung: table.get(rung, {}).get("verdict", "not_run") for rung in RUNG_ORDER}


# --------------------------------------------------------------------------- #
# SVC derivation (packaged auditor)
# --------------------------------------------------------------------------- #
def derive_svc(
    report: dict[str, Any],
    *,
    target_model: str,
    blind_label_provenance: str,
    inter_judge_kappa: float | None,
    human_audit_kappa: float | None,
    front_door: str,
    probe_kind: str,
    rerun_command: str | None,
) -> dict[str, Any]:
    """Build the activation-signal SVC verdict + card via ramp.audit."""
    from ramp.audit import build_card, engine, render_markdown
    from ramp.audit.bundle import LadderBundle

    bundle = LadderBundle.from_survival_report(
        report,
        "activation",
        signal_description=f"{target_model} {probe_kind} activation probe",
        target_model=target_model,
        blind_label_provenance=blind_label_provenance,
        inter_judge_kappa=inter_judge_kappa,
        human_audit_kappa=human_audit_kappa,
        front_door=front_door,
        probe_kind=probe_kind,
        rerun_command=rerun_command,
    )
    verdict = engine.audit(bundle)
    card = build_card(bundle, verdict)
    return {"verdict": verdict, "card": card, "card_markdown": render_markdown(card)}


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run_model(
    *,
    model_name: str,
    layer_files: dict[int, Path],
    output_dir: Path,
    review_csv: str,
    feature_table: str,
    blind_review_csv: str | None,
    num_splits: int,
    svc_kwargs: dict[str, Any],
    full_layer_override: int | None = None,
    sweep_extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Sweep a model's layers, select the best, run the full ladder + SVC."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir = output_dir / "sweep"

    print(f"[{model_name}] sweeping {len(layer_files)} layers "
          f"({SWEEP_RUNGS}) to pick best by crossfit activation AUROC", flush=True)
    per_layer: list[dict[str, Any]] = []
    for layer in sorted(layer_files):
        activation = layer_files[layer]
        report = run_ladder(
            activation=activation,
            layer=layer,
            rungs=SWEEP_RUNGS,
            output_json=sweep_dir / f"layer_{layer}.naive_crossfit.json",
            review_csv=review_csv,
            feature_table=feature_table,
            blind_review_csv=blind_review_csv,
            num_splits=num_splits,
            extra_args=sweep_extra_args,
        )
        auroc = crossfit_activation_auroc(report)
        per_layer.append(
            {
                "layer": layer,
                "activation_file": str(activation),
                "n_binary_eval_rows": report.get("num_binary_eval_rows"),
                **auroc,
            }
        )
        a = auroc["activation_auroc"]
        print(f"     layer {layer:>2}: crossfit activation AUROC = "
              f"{a:.6f}" if a is not None else f"     layer {layer:>2}: crossfit AUROC = n/a",
              flush=True)

    scored = [row for row in per_layer if row["activation_auroc"] is not None]
    if not scored:
        raise RuntimeError(f"[{model_name}] no layer produced a crossfit activation AUROC")
    if full_layer_override is not None:
        best = next((r for r in per_layer if r["layer"] == full_layer_override), None)
        if best is None:
            raise ValueError(f"--full-layer {full_layer_override} not among swept layers")
    else:
        best = max(scored, key=lambda row: row["activation_auroc"])
    best_layer = best["layer"]
    print(f"[{model_name}] selected layer {best_layer} "
          f"(crossfit activation AUROC = {best['activation_auroc']:.6f})", flush=True)

    print(f"[{model_name}] running FULL ladder ({FULL_RUNGS}) for layer {best_layer}", flush=True)
    full_report = run_ladder(
        activation=layer_files[best_layer],
        layer=best_layer,
        rungs=FULL_RUNGS,
        output_json=output_dir / f"{model_name}.survival_ladder.layer_{best_layer}.json",
        output_md=output_dir / f"{model_name}.survival_ladder.layer_{best_layer}.md",
        review_csv=review_csv,
        feature_table=feature_table,
        blind_review_csv=blind_review_csv,
        num_splits=num_splits,
    )

    rerun_command = svc_kwargs.pop("rerun_command", None) or (
        f"PYTHONPATH=src .venv/bin/python scripts/run_multimodel_ladder.py "
        f"--model-name {model_name} --full-layer {best_layer}"
    )
    svc = derive_svc(full_report, rerun_command=rerun_command, **svc_kwargs)

    per_rung = per_rung_activation_verdicts(full_report)
    summary = {
        "model_name": model_name,
        "selected_layer": best_layer,
        "selection_metric": "crossfit (leakage-free in-distribution) activation AUROC",
        "per_layer_activation_auroc": per_layer,
        "selected_layer_per_rung_activation_verdict": per_rung,
        "activation_svc_verdict": svc["verdict"]["verdict"],
        "activation_svc_status": svc["verdict"]["status"],
        "activation_svc_reasons": svc["verdict"]["reasons"],
        "full_report_path": str(
            output_dir / f"{model_name}.survival_ladder.layer_{best_layer}.json"
        ),
        "shared_inputs": {
            "review_csv": review_csv,
            "feature_table": feature_table,
            "blind_review_csv": blind_review_csv,
            "num_splits": num_splits,
        },
    }

    # Persist the model summary + the SVC card.
    (output_dir / f"{model_name}.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / f"{model_name}.activation_svc.json").write_text(
        json.dumps(svc["card"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / f"{model_name}.activation_svc.md").write_text(
        svc["card_markdown"] + "\n", encoding="utf-8"
    )

    _print_model_report(summary, svc)
    return {"summary": summary, "svc": svc, "full_report": full_report}


def _print_model_report(summary: dict[str, Any], svc: dict[str, Any]) -> None:
    name = summary["model_name"]
    print()
    print(f"================ {name}: ACTIVATION SIGNAL VERDICT ================")
    print(f"selected layer: {summary['selected_layer']} "
          f"(by {summary['selection_metric']})")
    print()
    print("per-layer crossfit activation AUROC:")
    print(f"  {'layer':>5}  {'act_auroc':>10}  {'prompt_auroc':>12}  {'delta':>9}")
    for row in summary["per_layer_activation_auroc"]:
        a = row["activation_auroc"]
        b = row["baseline_auroc"]
        d = row["delta"]
        mark = "  <-- selected" if row["layer"] == summary["selected_layer"] else ""
        if a is None:
            print(f"  {row['layer']:>5}  {'n/a':>10}{mark}")
        else:
            print(f"  {row['layer']:>5}  {a:>10.6f}  {b:>12.6f}  {d:>+9.6f}{mark}")
    print()
    per_rung = summary["selected_layer_per_rung_activation_verdict"]
    print("selected-layer per-rung activation verdict:")
    print("  " + ", ".join(f"{rung}={per_rung[rung]}" for rung in RUNG_ORDER))
    print()
    print(f"activation SVC verdict: {summary['activation_svc_verdict']} "
          f"(status: {summary['activation_svc_status']})")
    print("================================================================")
    print()


# --------------------------------------------------------------------------- #
# GPT-OSS correctness gate
# --------------------------------------------------------------------------- #
def verify_gpt_oss(args: argparse.Namespace) -> int:
    """Reproduce the canonical GPT-OSS layer_19 activation verdict.

    Runs the full ladder on the GPT-OSS layer_19 activation with the resolved
    shared inputs and confirms it reproduces the canonical activation verdict in
    the canonical report (per-rung + SVC). Returns 0 on MATCH, 1 on NO-MATCH.
    """
    output_dir = Path(args.output_dir or ".artifacts/multimodel/ladder/gpt_oss_20b_verify")
    activation = Path(GPT_OSS_ACTIVATION)
    if not activation.is_file():
        raise FileNotFoundError(f"GPT-OSS reference activation not found: {activation}")

    print("=== GPT-OSS correctness gate: reproducing canonical activation verdict ===")
    print(f"activation: {activation}")
    print(f"shared inputs: review={args.review_csv}")
    print(f"               feature_table={args.feature_table}")
    print(f"               blind={args.blind_review_csv}")
    print(f"               num_splits={args.num_splits}")
    print()

    full_report = run_ladder(
        activation=activation,
        layer=19,
        rungs=FULL_RUNGS,
        output_json=output_dir / "gpt_oss_20b.survival_ladder.layer_19.json",
        output_md=output_dir / "gpt_oss_20b.survival_ladder.layer_19.md",
        review_csv=args.review_csv,
        feature_table=args.feature_table,
        blind_review_csv=args.blind_review_csv,
        num_splits=args.num_splits,
    )

    repro_per_rung = per_rung_activation_verdicts(full_report)
    svc = derive_svc(
        full_report,
        target_model="openai/gpt-oss-20b",
        blind_label_provenance=args.blind_label_provenance,
        inter_judge_kappa=args.inter_judge_kappa,
        human_audit_kappa=args.human_audit_kappa,
        front_door=args.front_door,
        probe_kind=args.probe_kind,
        rerun_command="PYTHONPATH=src .venv/bin/python scripts/run_multimodel_ladder.py "
        "--verify-gpt-oss",
    )
    repro_svc_verdict = svc["verdict"]["verdict"]

    # Also read the canonical report's activation table directly for a side-by-side.
    canonical = json.loads(Path(CANONICAL_REPORT).read_text(encoding="utf-8"))
    canon_per_rung = {
        rung: canonical["survival_table"]["activation"].get(rung, {}).get("verdict", "?")
        for rung in RUNG_ORDER
    }

    print()
    print("=================== GPT-OSS REPRODUCTION ===================")
    print(f"  {'rung':>9}  {'reproduced':>12}  {'canonical-file':>14}  {'spec':>10}  match")
    rung_ok = True
    for rung in RUNG_ORDER:
        r = repro_per_rung[rung]
        c = canon_per_rung[rung]
        s = CANONICAL_ACTIVATION_PER_RUNG[rung]
        ok = (r == c == s)
        rung_ok = rung_ok and ok
        print(f"  {rung:>9}  {r:>12}  {c:>14}  {s:>10}  {'OK' if ok else 'MISMATCH'}")
    print()
    svc_ok = repro_svc_verdict == CANONICAL_SVC_VERDICT
    print(f"  SVC verdict: reproduced={repro_svc_verdict}  "
          f"canonical={CANONICAL_SVC_VERDICT}  {'OK' if svc_ok else 'MISMATCH'}")
    print("===========================================================")

    matched = rung_ok and svc_ok
    print()
    if matched:
        print("RESULT: MATCH - harness reproduces the canonical GPT-OSS activation "
              "verdict (leak_inflated; naive/split survive, crossfit/blind mixed, "
              "shifted fails).")
    else:
        print("RESULT: NO-MATCH - reproduced verdict differs from canonical. "
              "Do NOT proceed to the new models; the paths/harness are wrong.")

    (output_dir / "gpt_oss_20b.verify_summary.json").write_text(
        json.dumps(
            {
                "match": matched,
                "reproduced_per_rung": repro_per_rung,
                "canonical_file_per_rung": canon_per_rung,
                "spec_per_rung": CANONICAL_ACTIVATION_PER_RUNG,
                "reproduced_svc_verdict": repro_svc_verdict,
                "canonical_svc_verdict": CANONICAL_SVC_VERDICT,
                "report_path": str(output_dir / "gpt_oss_20b.survival_ladder.layer_19.json"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0 if matched else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--verify-gpt-oss",
        action="store_true",
        help="Correctness gate: reproduce the canonical GPT-OSS layer_19 activation verdict "
        "and exit (no new-model run).",
    )
    parser.add_argument("--model-name", default=None, help="Output label, e.g. llama31_8b.")
    parser.add_argument(
        "--activations-dir",
        default=None,
        help="Directory of <slug>.layer_<N>.jsonl files for the model.",
    )
    parser.add_argument(
        "--activation-file",
        action="append",
        default=None,
        help="Explicit layer file (repeatable); overrides --activations-dir discovery.",
    )
    parser.add_argument(
        "--full-layer",
        type=int,
        default=None,
        help="Force the full ladder on this layer instead of the AUROC-selected best.",
    )
    parser.add_argument("--output-dir", default=None, help="Where to write reports/cards.")

    # Shared (held-constant) inputs; defaults are the canonical GPT-OSS inputs.
    parser.add_argument("--review-csv", default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--feature-table", default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--blind-review-csv", default=DEFAULT_BLIND_REVIEW_CSV)
    parser.add_argument("--num-splits", type=int, default=30)

    # SVC scope metadata (defaults match the canonical card emitter).
    parser.add_argument("--target-model", default=None, help="HF id for the SVC scope.")
    parser.add_argument("--front-door", default="qwen3guard_0.6b")
    parser.add_argument("--probe-kind", default="linear")
    parser.add_argument(
        "--blind-label-provenance",
        default="silver_llm",
        choices=["human", "silver_llm", "silver_llm_inter_judge_validated"],
    )
    parser.add_argument("--inter-judge-kappa", type=float, default=0.887)
    parser.add_argument("--human-audit-kappa", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.verify_gpt_oss:
        return verify_gpt_oss(args)

    if not args.model_name:
        raise SystemExit("--model-name is required (or pass --verify-gpt-oss)")
    if not args.activations_dir and not args.activation_file:
        raise SystemExit("provide --activations-dir or one or more --activation-file")

    if args.activation_file:
        layer_files = {layer_of(Path(p)): Path(p) for p in args.activation_file}
    else:
        layer_files = discover_layer_files(Path(args.activations_dir))

    output_dir = Path(args.output_dir or f".artifacts/multimodel/ladder/{args.model_name}")
    svc_kwargs = {
        "target_model": args.target_model or args.model_name,
        "front_door": args.front_door,
        "probe_kind": args.probe_kind,
        "blind_label_provenance": args.blind_label_provenance,
        "inter_judge_kappa": args.inter_judge_kappa,
        "human_audit_kappa": args.human_audit_kappa,
    }
    run_model(
        model_name=args.model_name,
        layer_files=layer_files,
        output_dir=output_dir,
        review_csv=args.review_csv,
        feature_table=args.feature_table,
        blind_review_csv=args.blind_review_csv,
        num_splits=args.num_splits,
        svc_kwargs=svc_kwargs,
        full_layer_override=args.full_layer,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
