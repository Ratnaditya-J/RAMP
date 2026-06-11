#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"

INK = "#111111"
MUTED = "#555555"
GRID = "#d8d8d8"
LIGHT = "#f7f7f7"
SERIES_A = "#111111"
SERIES_B = "#6b6b6b"
SERIES_C = "#b0b0b0"


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


def text_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    fnt: ImageFont.ImageFont,
    fill: str = INK,
) -> None:
    draw.text(xy, text, font=fnt, fill=fill)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def centered(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: str = INK,
) -> None:
    width, height = text_size(draw, text, fnt)
    draw.text((center[0] - width / 2, center[1] - height / 2), text, font=fnt, fill=fill)


def wrap(
    draw: ImageDraw.ImageDraw,
    text: str,
    fnt: ImageFont.ImageFont,
    width: int,
) -> list[str]:
    lines: list[str] = []
    current: list[str] = []
    for word in text.split():
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


def save(img: Image.Image, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    img.save(FIG_DIR / name, optimize=True)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    fill: str = INK,
) -> None:
    draw.line((*start, *end), fill=fill, width=3)
    x, y = end
    draw.polygon([(x, y), (x - 14, y - 8), (x - 14, y + 8)], fill=fill)


def build_pipeline() -> None:
    img = Image.new("RGB", (1800, 720), "#ffffff")
    draw = ImageDraw.Draw(img)
    title = font(28, bold=True)
    body = font(22)
    small = font(18)

    stages = [
        ("Prompt", "runtime\nw=0.80"),
        ("Embedding", "runtime\nw=0.20"),
        ("Activation", "audit\nw=0"),
        ("Output", "post-gen\naudit"),
        ("Session", "escalate\naudit"),
        ("Tool/action", "gate\nscaffold"),
    ]

    x0, y0, box_w, box_h, gap = 80, 170, 220, 150, 58
    centers: list[tuple[int, int]] = []
    for idx, (name, role) in enumerate(stages):
        x = x0 + idx * (box_w + gap)
        centers.append((x + box_w // 2, y0 + box_h // 2))
        draw.rectangle((x, y0, x + box_w, y0 + box_h), fill=LIGHT, outline=INK, width=3)
        centered(draw, (x + box_w // 2, y0 + 45), name, title)
        for line_idx, line in enumerate(role.split("\n")):
            centered(draw, (x + box_w // 2, y0 + 92 + 28 * line_idx), line, body, MUTED)
        if idx:
            prev_x = x - gap + 12
            draw_arrow(draw, (prev_x, y0 + box_h // 2), (x - 12, y0 + box_h // 2))

    policy_x1, policy_y1, policy_x2, policy_y2 = 360, 470, 1440, 605
    draw.rectangle(
        (policy_x1, policy_y1, policy_x2, policy_y2),
        fill="#ffffff",
        outline=INK,
        width=3,
    )
    centered(draw, ((policy_x1 + policy_x2) // 2, policy_y1 + 36), "current policy", title)
    policy = (
        "Primary score: 0.80 prompt + 0.20 embedding, threshold 0.50. "
        "Activation, output, and session signals remain audit/escalation evidence "
        "until leakage-free blind holdouts justify positive runtime weight."
    )
    for line_idx, line in enumerate(wrap(draw, policy, small, policy_x2 - policy_x1 - 80)):
        centered(
            draw,
            ((policy_x1 + policy_x2) // 2, policy_y1 + 78 + 25 * line_idx),
            line,
            small,
            MUTED,
        )

    bus_y = 410
    prompt_x = centers[0][0]
    embedding_x = centers[1][0]
    policy_mid_x = (policy_x1 + policy_x2) // 2
    for source_x in (prompt_x, embedding_x):
        draw.line((source_x, y0 + box_h, source_x, bus_y), fill=INK, width=2)
    draw.line((prompt_x, bus_y, policy_mid_x, bus_y), fill=INK, width=2)
    draw.line((policy_mid_x, bus_y, policy_mid_x, policy_y1), fill=INK, width=2)
    save(img, "ramp_pipeline.png")


def draw_grouped_bar_chart(
    name: str,
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    y_label: str,
    max_value: float = 1.0,
) -> None:
    img = Image.new("RGB", (1300, 820), "#ffffff")
    draw = ImageDraw.Draw(img)
    axis_f = font(20)
    label_f = font(22)
    value_f = font(17)

    left, top, width, height = 135, 95, 1045, 545
    bottom = top + height
    draw.line((left, bottom, left + width, bottom), fill=INK, width=3)
    draw.line((left, top, left, bottom), fill=INK, width=3)

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = bottom - int(height * tick / max_value)
        draw.line((left, y, left + width, y), fill=GRID, width=1)
        draw.line((left - 7, y, left, y), fill=INK, width=2)
        draw.text((left - 62, y - 12), f"{tick:.2f}", font=axis_f, fill=INK)

    draw.text((left, 46), y_label, font=axis_f, fill=INK)

    n_groups = len(labels)
    n_series = len(series)
    group_width = width / n_groups
    bar_width = min(58, (group_width - 55) / n_series)
    patterns = [None, "outline", "stripe"]
    for group_idx, label in enumerate(labels):
        base_x = left + group_idx * group_width + group_width / 2
        for series_idx, (_, values, color) in enumerate(series):
            value = values[group_idx]
            x1 = int(base_x - (n_series * bar_width) / 2 + series_idx * bar_width)
            x2 = int(x1 + bar_width - 6)
            y1 = bottom - int(height * value / max_value)
            draw.rectangle((x1, y1, x2, bottom), fill=color, outline=INK, width=1)
            if patterns[series_idx % len(patterns)] == "stripe":
                for sx in range(x1 - 20, x2 + 20, 10):
                    draw.line((sx, bottom, sx + 70, y1), fill="#777777", width=1)
            if patterns[series_idx % len(patterns)] == "outline":
                draw.rectangle((x1 + 5, y1 + 5, x2 - 5, bottom - 5), outline="#ffffff", width=2)
            centered(draw, ((x1 + x2) // 2, y1 - 18), f"{value:.3f}", value_f)
        for line_idx, line in enumerate(wrap(draw, label, label_f, int(group_width - 18))):
            centered(draw, (int(base_x), bottom + 42 + 28 * line_idx), line, label_f)

    legend_y = 735
    legend_x = 150
    for idx, (legend, _, color) in enumerate(series):
        x = legend_x + idx * 360
        draw.rectangle((x, legend_y, x + 34, legend_y + 22), fill=color, outline=INK)
        draw.text((x + 46, legend_y - 1), legend, font=axis_f, fill=INK)
    save(img, name)


def build_charts() -> None:
    policy = load_json("data/fusion_policy/ramp_fusion_policy_v0_2.json")
    split = policy["reviewed_split_stability"]
    po = split["prompt_only_calibrated"]
    pe = split["prompt_embedding_calibrated"]
    pea = split["prompt_embedding_activation_calibrated"]
    draw_grouped_bar_chart(
        "input_side_split_stability.png",
        ["AUROC", "Recall", "FPR"],
        [
            (
                "prompt only",
                [po["auc_mean"], po["recall_mean"], po["false_positive_rate_mean"]],
                SERIES_A,
            ),
            (
                "prompt + embedding",
                [pe["auc_mean"], pe["recall_mean"], pe["false_positive_rate_mean"]],
                SERIES_B,
            ),
            (
                "prompt + embedding + activation",
                [pea["auc_mean"], pea["recall_mean"], pea["false_positive_rate_mean"]],
                SERIES_C,
            ),
        ],
        "Mean metric value",
    )

    output = load_json(
        ".artifacts/output_eval/"
        "ramp_prompt_embedding_activation_output_calibration_refined_v0_1.json"
    )["aggregate_holdout_metrics"]

    def mean(name: str, metric: str) -> float:
        return output[name][metric]["mean"]

    draw_grouped_bar_chart(
        "output_ablation.png",
        ["AUROC", "Recall", "FPR"],
        [
            (
                "P+E+A",
                [
                    mean("prompt_embedding_activation_calibrated", "auc"),
                    mean("prompt_embedding_activation_calibrated", "recall"),
                    mean("prompt_embedding_activation_calibrated", "false_positive_rate"),
                ],
                SERIES_A,
            ),
            (
                "P+A+O",
                [
                    mean("prompt_activation_output_calibrated", "auc"),
                    mean("prompt_activation_output_calibrated", "recall"),
                    mean("prompt_activation_output_calibrated", "false_positive_rate"),
                ],
                SERIES_B,
            ),
            (
                "P+E+A+O",
                [
                    mean("prompt_embedding_activation_output_calibrated", "auc"),
                    mean("prompt_embedding_activation_output_calibrated", "recall"),
                    mean(
                        "prompt_embedding_activation_output_calibrated",
                        "false_positive_rate",
                    ),
                ],
                SERIES_C,
            ),
        ],
        "Mean metric value",
    )

    rjudge = load_json(
        ".artifacts/session_eval/ramp_session_signal_fusion_eval_rjudge_qwen_v0_1.json"
    )
    keys = [
        "single_turn_max",
        "compact_session_classifier",
        "full_transcript_session_classifier",
        "max_session_signal",
    ]
    draw_grouped_bar_chart(
        "session_ablation_rjudge.png",
        ["single-turn max", "compact session", "full transcript", "max signal"],
        [
            ("Recall", [rjudge["metrics"][key]["recall"] for key in keys], SERIES_A),
            (
                "FPR",
                [rjudge["metrics"][key]["false_positive_rate"] for key in keys],
                SERIES_B,
            ),
        ],
        "Metric value",
    )


def main() -> None:
    build_pipeline()
    build_charts()
    print(f"wrote paper figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
