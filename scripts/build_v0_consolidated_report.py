#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_POLICY = "data/fusion_policy/ramp_multistage_policy_v0_1.json"
DEFAULT_INPUT_POLICY = "data/fusion_policy/ramp_fusion_policy_v0_2.json"
DEFAULT_OUTPUT_CALIBRATION = (
    ".artifacts/output_eval/"
    "ramp_prompt_embedding_activation_output_calibration_refined_v0_1.json"
)
DEFAULT_SESSION_RJUDGE = (
    ".artifacts/session_eval/ramp_session_signal_fusion_eval_rjudge_qwen_v0_1.json"
)
DEFAULT_SESSION_MHJ = ".artifacts/session_eval/ramp_session_signal_fusion_eval_mhj_qwen_v0_1.json"
DEFAULT_ACTIVATION = ".artifacts/activation_probes/ramp_activation_probe_layer_comparison_v0_1.json"
DEFAULT_EMBEDDING = ".artifacts/centroids/ramp_input_embedding_feature_calibration_v0_1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the consolidated RAMP v0 research report.")
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--input-policy", default=DEFAULT_INPUT_POLICY)
    parser.add_argument("--output-calibration", default=DEFAULT_OUTPUT_CALIBRATION)
    parser.add_argument("--session-rjudge", default=DEFAULT_SESSION_RJUDGE)
    parser.add_argument("--session-mhj", default=DEFAULT_SESSION_MHJ)
    parser.add_argument("--activation-comparison", default=DEFAULT_ACTIVATION)
    parser.add_argument("--embedding-calibration", default=DEFAULT_EMBEDDING)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any] | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def metric_mean(metrics: dict[str, Any], metric: str) -> float | None:
    value = metrics.get(metric)
    if isinstance(value, dict):
        return value.get("mean")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def session_row(report: dict[str, Any], name: str) -> dict[str, Any]:
    metric = report["metrics"][name]
    auc = report["threshold_sweeps"][name]["auc"]
    return {
        "auc": auc,
        "recall": metric["recall"],
        "false_positive_rate": metric["false_positive_rate"],
        "f1": metric["f1"],
        "single_turn_false_negatives_caught": len(
            report["single_turn_false_negatives_caught"].get(name, [])
        ),
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_json(args.policy) or {}
    input_policy = load_json(args.input_policy) or {}
    output = load_json(args.output_calibration) or {}
    session_rjudge = load_json(args.session_rjudge) or {}
    session_mhj = load_json(args.session_mhj) or {}
    activation = load_json(args.activation_comparison) or {}
    embedding = load_json(args.embedding_calibration) or {}

    reviewed = input_policy.get("reviewed_split_stability", {})
    output_metrics = output.get("aggregate_holdout_metrics", {})
    summary = {
        "artifact_id": "ramp_v0_consolidated_research_report",
        "policy": policy,
        "input_side_policy": {
            "decision": input_policy.get("decision"),
            "selected_runtime_score": input_policy.get("selected_runtime_score"),
            "protocol": reviewed.get("protocol"),
            "limitations": input_policy.get("limitations", []),
            "prompt_only": reviewed.get("prompt_only_calibrated"),
            "prompt_embedding": reviewed.get("prompt_embedding_calibrated"),
            "prompt_activation": reviewed.get("prompt_activation_calibrated"),
            "prompt_embedding_activation": reviewed.get(
                "prompt_embedding_activation_calibrated"
            ),
        },
        "activation_probe": {
            "selected_layer_id": activation.get("selected_layer_id"),
            "selection_rule": activation.get("selection_rule"),
            "selected_artifact_path": activation.get("selected_artifact_path"),
        },
        "embedding": {
            "target_fpr": embedding.get("target_fpr"),
            "runs": [
                {
                    "name": run.get("name"),
                    "recommended_role": run.get("recommendation", {}).get(
                        "recommended_role"
                    ),
                    "conservative_recall": run.get("recommendation", {}).get(
                        "conservative_recall"
                    ),
                    "conservative_fpr": run.get("recommendation", {}).get(
                        "conservative_false_positive_rate"
                    ),
                }
                for run in embedding.get("runs", [])
            ],
        },
        "output_classifier": {
            name: {
                "auc_mean": metric_mean(metrics, "auc"),
                "recall_mean": metric_mean(metrics, "recall"),
                "false_positive_rate_mean": metric_mean(
                    metrics, "false_positive_rate"
                ),
            }
            for name, metrics in output_metrics.items()
        },
        "session_classifier": {
            "rjudge": {
                name: session_row(session_rjudge, name)
                for name in session_rjudge.get("metrics", {})
            },
            "mhj": {
                name: session_row(session_mhj, name)
                for name in session_mhj.get("metrics", {})
            },
        },
    }
    return summary


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    input_policy = summary["input_side_policy"]
    selected = input_policy.get("selected_runtime_score") or {}
    weights = selected.get("weights") or {}
    po = input_policy.get("prompt_only") or {}
    pe = input_policy.get("prompt_embedding") or {}
    pa = input_policy.get("prompt_activation") or {}
    pea = input_policy.get("prompt_embedding_activation") or {}

    output = summary["output_classifier"]
    session_rjudge = summary["session_classifier"]["rjudge"]
    session_mhj = summary["session_classifier"]["mhj"]
    policy = summary["policy"]

    lines = [
        "# RAMP v0 Consolidated Research Report",
        "",
        "RAMP v0 is feature-complete as a research multi-stage classifier. The current evidence",
        "now supersedes the earlier prompt-plus-activation headline: under cross-fitted,",
        "leakage-free reviewed-label calibration, the selected runtime policy is prompt plus",
        "input-embedding proximity. Activation, output, and session signals remain implemented",
        "audit, research, post-generation, or escalation evidence until larger blind holdouts",
        "justify positive runtime weight.",
        "",
        "## Frozen v0.2 Input-Side Policy",
        "",
        f"- Policy artifact: `ramp_fusion_policy_v0.2`",
        f"- Supersedes: `ramp_fusion_policy_v0.1`",
        f"- Decision: `{input_policy.get('decision', 'N/A')}`",
        f"- Runtime threshold: `{selected.get('threshold', 'N/A')}`",
        f"- Prompt weight: `{weights.get('prompt_risk_score', 'N/A')}`",
        f"- Activation weight: `{weights.get('activation_probability', 'N/A')}`",
        f"- Embedding runtime weight: `{weights.get('embedding_prior_score', 'N/A')}`",
        "- Output classifier: post-generation audit, no positive v0 runtime weight",
        "- Session classifier: calibrated escalation/audit signal, no naive OR/max blocking",
        "- Tool/action gate: reference deterministic gate, benchmark evaluation still pending",
        "",
        "## Input-Side Reviewed Split Stability",
        "",
        "| Condition | AUC mean | Recall mean | FPR mean | FP mean | FN mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            "| `prompt_only_calibrated` | {} | {} | {} | {} | {} |"
        ).format(
            fmt(po.get("auc_mean")),
            fmt(po.get("recall_mean")),
            fmt(po.get("false_positive_rate_mean")),
            fmt(po.get("false_positive_count_mean")),
            fmt(po.get("false_negative_count_mean")),
        ),
        (
            "| `prompt_embedding_calibrated` | {} | {} | {} | {} | {} |"
        ).format(
            fmt(pe.get("auc_mean")),
            fmt(pe.get("recall_mean")),
            fmt(pe.get("false_positive_rate_mean")),
            fmt(pe.get("false_positive_count_mean")),
            fmt(pe.get("false_negative_count_mean")),
        ),
        (
            "| `prompt_activation_calibrated` | {} | {} | {} | {} | {} |"
        ).format(
            fmt(pa.get("auc_mean")),
            fmt(pa.get("recall_mean")),
            fmt(pa.get("false_positive_rate_mean")),
            fmt(pa.get("false_positive_count_mean")),
            fmt(pa.get("false_negative_count_mean")),
        ),
        (
            "| `prompt_embedding_activation_calibrated` | {} | {} | {} | {} | {} |"
        ).format(
            fmt(pea.get("auc_mean")),
            fmt(pea.get("recall_mean")),
            fmt(pea.get("false_positive_rate_mean")),
            fmt(pea.get("false_positive_count_mean")),
            fmt(pea.get("false_negative_count_mean")),
        ),
        "",
        "Decision: prompt+embedding is the selected v0.2 runtime policy because it improves",
        "AUROC, F1, accuracy, and FPR over prompt-only under the cross-fitted leakage-free",
        "protocol, with a small recall tradeoff. Full prompt+embedding+activation has marginally",
        "higher AUROC but lower recall/F1, so activation remains an audit/research signal pending",
        "blind-holdout validation.",
        "",
        "## Output Classifier",
        "",
        "| Condition | AUC mean | Recall mean | FPR mean |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in (
        "prompt_embedding_activation_calibrated",
        "prompt_activation_output_calibrated",
        "prompt_embedding_activation_output_calibrated",
    ):
        metrics = output.get(name, {})
        lines.append(
            "| `{}` | {} | {} | {} |".format(
                name,
                fmt(metrics.get("auc_mean")),
                fmt(metrics.get("recall_mean")),
                fmt(metrics.get("false_positive_rate_mean")),
            )
        )
    lines.extend(
        [
            "",
            "Decision: output scoring is implemented and useful for post-generation audit, but",
            "the v0 prompt/response set does not justify positive fusion weight.",
            "",
            "## Session Classifier",
            "",
            "R-Judge labeled session comparison at threshold `0.55`:",
            "",
            "| Condition | AUC | Recall | FPR | Single-turn FNs caught |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in (
        "single_turn_max",
        "compact_session_classifier",
        "full_transcript_session_classifier",
        "max_session_signal",
    ):
        metrics = session_rjudge.get(name, {})
        lines.append(
            "| `{}` | {} | {} | {} | {} |".format(
                name,
                fmt(metrics.get("auc")),
                fmt(metrics.get("recall")),
                fmt(metrics.get("false_positive_rate")),
                metrics.get("single_turn_false_negatives_caught", "N/A"),
            )
        )
    lines.extend(
        [
            "",
            "MHJ unsafe-only stress comparison at threshold `0.55`:",
            "",
            "| Condition | Recall | Single-turn FNs caught |",
            "| --- | ---: | ---: |",
        ]
    )
    for name in (
        "single_turn_max",
        "compact_session_classifier",
        "full_transcript_session_classifier",
        "max_session_signal",
    ):
        metrics = session_mhj.get(name, {})
        lines.append(
            "| `{}` | {} | {} |".format(
                name,
                fmt(metrics.get("recall")),
                metrics.get("single_turn_false_negatives_caught", "N/A"),
            )
        )
    lines.extend(
        [
            "",
            "Decision: full-transcript session scoring shows real session signal, but compact",
            "state is too lossy and naive max/OR fusion raises false positives. Use session",
            "classification as escalation/audit in v0.",
            "",
            "## Negative And Limited Results",
            "",
            (
                "- Input embeddings are not a standalone decision feature, but prompt+embedding "
                "is the selected v0.2 input-side runtime policy pending blind holdout."
            ),
            "- Output classification does not improve the best input-side v0 fusion yet.",
            "- Compact session evidence does not recover enough full-transcript signal yet.",
            "- Naive OR/max fusion improves unsafe recall but increases false positives.",
            (
                "- Tool/action gating is implemented as a design pattern but lacks benchmark "
                "validation."
            ),
            "",
            "## Paper Status",
            "",
            "This is a feature-complete research v0, not a paper-final claim. The next paper-grade",
            "step is broader reviewed labeling, especially for output responses, session FPR/AUC,",
            "and tool/action examples.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = build_summary(args)
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    write_markdown(Path(args.output_md), summary)
    print(f"wrote consolidated v0 markdown to {args.output_md}")
    if args.output_json:
        print(f"wrote consolidated v0 json to {args.output_json}")


if __name__ == "__main__":
    main()
