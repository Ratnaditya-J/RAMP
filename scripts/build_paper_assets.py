#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"


def load_json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    supplemental = "/System/Library/Fonts/Supplemental"
    candidates = [
        f"{supplemental}/Arial Bold.ttf" if bold else f"{supplemental}/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    fnt: ImageFont.ImageFont,
    width: int,
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if text_size(draw, candidate, fnt)[0] <= width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def draw_centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: str,
) -> None:
    width, height = text_size(draw, text, fnt)
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=fnt, fill=fill)


def save(img: Image.Image, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    img.save(FIG_DIR / name, optimize=True)


def build_pipeline() -> None:
    img = Image.new("RGB", (2200, 1050), "#ffffff")
    draw = ImageDraw.Draw(img)
    title = font(54, bold=True)
    label = font(30, bold=True)
    small = font(24)
    tiny = font(20)
    draw.text((90, 65), "RAMP multi-stage risk accumulation", font=title, fill="#14213d")
    draw.text(
        (90, 130),
        (
            "Cheap prompt evidence acts first; expensive or delayed signals become "
            "audit/escalation evidence unless calibration earns runtime weight."
        ),
        font=small,
        fill="#485465",
    )

    stages = [
        ("Prompt classifier", "Qwen3Guard\nruntime weight 0.25", "#26547c"),
        ("Input embedding", "GPT-OSS centroids\naudit/taxonomy", "#2a9d8f"),
        ("Activation probe", "GPT-OSS layer 19\nruntime weight 0.75", "#7b2cbf"),
        ("Output classifier", "post-generation\naudit signal", "#f77f00"),
        ("Session classifier", "full transcript works\ncompact not ready", "#d62828"),
        ("Tool/action gate", "agent action\nreference gate", "#4d908e"),
    ]
    x0, y, w, h, gap = 90, 310, 300, 210, 42
    for idx, (name, desc, color) in enumerate(stages):
        x = x0 + idx * (w + gap)
        draw.rounded_rectangle(
            (x, y, x + w, y + h),
            radius=28,
            fill="#f7f9fc",
            outline=color,
            width=5,
        )
        draw.rectangle((x, y, x + w, y + 18), fill=color)
        draw_centered(draw, (x + w // 2, y + 68), name, label, "#1f2937")
        for line_idx, line in enumerate(desc.split("\n")):
            draw_centered(draw, (x + w // 2, y + 128 + 34 * line_idx), line, small, "#374151")
        if idx < len(stages) - 1:
            ax = x + w + 8
            ay = y + h // 2
            draw.line((ax, ay, ax + gap - 20, ay), fill="#6b7280", width=5)
            draw.polygon(
                [(ax + gap - 20, ay - 14), (ax + gap + 2, ay), (ax + gap - 20, ay + 14)],
                fill="#6b7280",
            )

    draw.rounded_rectangle(
        (515, 720, 1685, 910),
        radius=30,
        fill="#fff8e8",
        outline="#d97706",
        width=4,
    )
    draw_centered(draw, (1100, 775), "Frozen v0 decision policy", label, "#92400e")
    policy_lines = [
        "Primary runtime score = 0.25 prompt + 0.75 activation; threshold 0.53",
        (
            "Embedding, output, and session are retained as audit/escalation signals "
            "until future reviewed data earns positive weight."
        ),
    ]
    for i, line in enumerate(policy_lines):
        draw_centered(draw, (1100, 830 + 36 * i), line, tiny, "#78350f")
    save(img, "ramp_pipeline.png")


def bar_chart(
    name: str,
    title: str,
    subtitle: str,
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    max_value: float = 1.0,
) -> None:
    img = Image.new("RGB", (1700, 1050), "#ffffff")
    draw = ImageDraw.Draw(img)
    title_f = font(48, bold=True)
    small = font(22)
    axis = font(20)
    draw.text((80, 55), title, font=title_f, fill="#14213d")
    draw.text((80, 115), subtitle, font=small, fill="#4b5563")

    chart_x, chart_y, chart_w, chart_h = 150, 230, 1400, 620
    draw.line(
        (chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h),
        fill="#111827",
        width=3,
    )
    draw.line((chart_x, chart_y, chart_x, chart_y + chart_h), fill="#111827", width=3)
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = chart_y + chart_h - int((tick / max_value) * chart_h)
        draw.line((chart_x - 8, y, chart_x + chart_w, y), fill="#e5e7eb", width=2)
        draw.text((chart_x - 70, y - 12), f"{tick:.2f}", font=axis, fill="#374151")

    groups = len(labels)
    series_count = len(series)
    group_w = chart_w / groups
    bar_w = min(82, (group_w - 90) / max(1, series_count))
    for i, label_text in enumerate(labels):
        base_x = chart_x + i * group_w + group_w / 2
        for j, (_, values, color) in enumerate(series):
            value = values[i]
            x1 = int(base_x - (series_count * bar_w) / 2 + j * bar_w)
            x2 = int(x1 + bar_w - 8)
            y2 = chart_y + chart_h
            y1 = y2 - int((value / max_value) * chart_h)
            draw.rectangle((x1, y1, x2, y2), fill=color)
            draw.text((x1 - 5, y1 - 30), f"{value:.3f}", font=axis, fill="#111827")
        for line_idx, line in enumerate(wrap_text(draw, label_text, small, int(group_w - 20))):
            draw_centered(
                draw,
                (int(base_x), chart_y + chart_h + 45 + 28 * line_idx),
                line,
                small,
                "#1f2937",
            )

    legend_x = 220
    legend_y = 915
    for idx, (legend, _, color) in enumerate(series):
        x = legend_x + idx * 360
        draw.rectangle((x, legend_y, x + 32, legend_y + 32), fill=color)
        draw.text((x + 44, legend_y + 2), legend, font=small, fill="#111827")
    save(img, name)


def build_charts() -> None:
    policy = load_json("data/fusion_policy/ramp_fusion_policy_v0_1.json")
    split = policy["reviewed_split_stability"]
    pa = split["prompt_activation_calibrated"]
    pea = split["prompt_embedding_activation_calibrated"]
    bar_chart(
        "input_side_split_stability.png",
        "Input-side reviewed split stability",
        (
            "Embedding ties AUC but adds false-positive cost; prompt+activation is "
            "frozen as v0 runtime core."
        ),
        ["AUC", "Recall", "FPR"],
        [
            (
                "prompt + activation",
                [pa["auc_mean"], pa["recall_mean"], pa["false_positive_rate_mean"]],
                "#7b2cbf",
            ),
            (
                "prompt + embedding + activation",
                [pea["auc_mean"], pea["recall_mean"], pea["false_positive_rate_mean"]],
                "#2a9d8f",
            ),
        ],
    )

    output = load_json(
        ".artifacts/output_eval/ramp_prompt_embedding_activation_output_calibration_refined_v0_1.json"
    )["aggregate_holdout_metrics"]

    def mean(name: str, metric: str) -> float:
        return output[name][metric]["mean"]

    bar_chart(
        "output_ablation.png",
        "Output classifier ablation",
        "Output scoring is useful for audit, but did not improve the best v0 input-side fusion.",
        ["AUC", "Recall", "FPR"],
        [
            (
                "prompt + embedding + activation",
                [
                    mean("prompt_embedding_activation_calibrated", "auc"),
                    mean("prompt_embedding_activation_calibrated", "recall"),
                    mean("prompt_embedding_activation_calibrated", "false_positive_rate"),
                ],
                "#26547c",
            ),
            (
                "prompt + activation + output",
                [
                    mean("prompt_activation_output_calibrated", "auc"),
                    mean("prompt_activation_output_calibrated", "recall"),
                    mean("prompt_activation_output_calibrated", "false_positive_rate"),
                ],
                "#f77f00",
            ),
            (
                "prompt + embedding + activation + output",
                [
                    mean("prompt_embedding_activation_output_calibrated", "auc"),
                    mean("prompt_embedding_activation_output_calibrated", "recall"),
                    mean(
                        "prompt_embedding_activation_output_calibrated",
                        "false_positive_rate",
                    ),
                ],
                "#d62828",
            ),
        ],
    )

    rjudge = load_json(
        ".artifacts/session_eval/ramp_session_signal_fusion_eval_rjudge_qwen_v0_1.json"
    )
    labels = [
        "single-turn max",
        "compact session",
        "full transcript",
        "max session signal",
    ]
    keys = [
        "single_turn_max",
        "compact_session_classifier",
        "full_transcript_session_classifier",
        "max_session_signal",
    ]
    bar_chart(
        "session_ablation_rjudge.png",
        "Session classifier ablation on R-Judge",
        (
            "Full transcript catches single-turn misses, but max/OR fusion has a high "
            "false-positive rate."
        ),
        labels,
        [
            ("Recall", [rjudge["metrics"][key]["recall"] for key in keys], "#7b2cbf"),
            ("FPR", [rjudge["metrics"][key]["false_positive_rate"] for key in keys], "#d62828"),
        ],
    )


def main() -> None:
    build_pipeline()
    build_charts()
    print(f"wrote paper figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
