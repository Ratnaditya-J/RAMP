#!/usr/bin/env python
# ruff: noqa: E501
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

ROLE_STYLE = {
    "harmful": {"label": "Harmful", "color": "#c2410c", "bg": "#fff7ed"},
    "benign_near_neighbors": {
        "label": "Benign near-neighbor",
        "color": "#047857",
        "bg": "#ecfdf5",
    },
}

DOMAIN_GROUPS = {
    "Direct harm": {
        "cyber_abuse",
        "cbrn_and_hazardous_materials",
        "weapons_and_physical_violence",
        "critical_infrastructure_and_sabotage",
        "nonviolent_illegal_activity",
        "child_safety",
        "self_harm_and_wellbeing",
    },
    "Information and identity": {
        "privacy_identity_and_secrets",
        "ip_and_content_rights",
        "misinformation_manipulation_and_civic",
        "regulated_advice",
    },
    "Social and agentic integrity": {
        "hate_harassment_and_abuse",
        "sexual_safety_and_content",
        "agent_tool_and_system_integrity",
        "model_autonomy_and_control",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a polished RAMP taxonomy visual.")
    parser.add_argument("--taxonomy", required=True, help="Taxonomy JSON path.")
    parser.add_argument(
        "--corrections",
        default=None,
        help="Optional taxonomy corrections JSON path.",
    )
    parser.add_argument("--output-html", required=True, help="Output HTML path.")
    parser.add_argument("--output-svg", default=None, help="Optional standalone SVG path.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pretty(value: str) -> str:
    return value.replace("_", " ")


def domain_group(domain_id: str) -> str:
    for group, domains in DOMAIN_GROUPS.items():
        if domain_id in domains:
            return group
    return "Other"


def wrap_words(text: str, limit: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join([*current, word])
        if len(trial) > limit and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def text_lines(
    lines: list[str],
    *,
    x: int,
    y: int,
    klass: str,
    line_height: int = 17,
) -> str:
    output = []
    for idx, line in enumerate(lines):
        output.append(
            f'<text class="{klass}" x="{x}" y="{y + idx * line_height}">'
            f"{html.escape(line)}</text>"
        )
    return "\n".join(output)


def pill(x: int, y: int, text: str, *, color: str, bg: str, width: int) -> str:
    label = html.escape(text)
    return f"""
    <g>
      <rect x="{x}" y="{y}" width="{width}" height="26" rx="13" fill="{bg}" stroke="{color}" stroke-opacity="0.35"/>
      <text class="pill-text" x="{x + 13}" y="{y + 17}" fill="{color}">{label}</text>
    </g>
    """


def domain_card(domain: dict[str, Any], x: int, y: int, width: int) -> str:
    domain_id = str(domain["domain_id"])
    group = domain_group(domain_id)
    harmful = [pretty(item) for item in domain["harmful"]]
    benign = [pretty(item) for item in domain["benign_near_neighbors"]]
    highlighted = domain_id == "self_harm_and_wellbeing"
    height = 176 if highlighted else 154
    stroke = "#b45309" if highlighted else "#d6dee8"
    card_class = "domain-card highlighted" if highlighted else "domain-card"
    title_lines = wrap_words(pretty(domain_id), 28)
    harmful_text = ", ".join(harmful[:4]) + (", ..." if len(harmful) > 4 else "")
    benign_text = ", ".join(benign[:4]) + (", ..." if len(benign) > 4 else "")
    badge = (
        pill(x + width - 128, y + 16, "patched", color="#b45309", bg="#fffbeb", width=92)
        if highlighted
        else ""
    )
    return f"""
    <g class="{card_class}">
      <rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" fill="#ffffff" stroke="{stroke}"/>
      <text class="group-label" x="{x + 22}" y="{y + 28}">{html.escape(group)}</text>
      {badge}
      {text_lines(title_lines, x=x + 22, y=y + 56, klass="domain-title", line_height=20)}
      {pill(x + 22, y + 90, f"{len(harmful)} harmful", color=ROLE_STYLE["harmful"]["color"], bg=ROLE_STYLE["harmful"]["bg"], width=116)}
      {pill(x + 148, y + 90, f"{len(benign)} benign", color=ROLE_STYLE["benign_near_neighbors"]["color"], bg=ROLE_STYLE["benign_near_neighbors"]["bg"], width=112)}
      <text class="mini-label" x="{x + 22}" y="{y + 135}">Harmful: {html.escape(harmful_text)}</text>
      <text class="mini-label" x="{x + 22}" y="{y + 157}">Benign: {html.escape(benign_text)}</text>
    </g>
    """


def correction_panel(corrections: dict[str, Any] | None, x: int, y: int, width: int) -> str:
    if not corrections:
        body = "No correction overlay loaded."
    else:
        rows = corrections.get("corrections", [])
        first = rows[0] if rows else {}
        corrected = first.get("corrected", {})
        body = (
            f"{len(rows)} correction overlay: {first.get('source_id', 'n/a')} remapped to "
            f"{pretty(str(corrected.get('domain', 'unknown')))} / "
            f"{pretty(str(corrected.get('subcluster_id', 'unknown')))}."
        )
    return f"""
    <g>
      <rect x="{x}" y="{y}" width="{width}" height="104" rx="18" fill="#111827"/>
      <text class="dark-title" x="{x + 24}" y="{y + 35}">Taxonomy patch queue</text>
      <text class="dark-copy" x="{x + 24}" y="{y + 67}">{html.escape(body)}</text>
    </g>
    """


def build_svg(taxonomy: dict[str, Any], corrections: dict[str, Any] | None) -> str:
    width = 1440
    card_width = 420
    x_positions = [54, 510, 966]
    y_start = 238
    y_gap = 184
    domains = sorted(taxonomy["domains"], key=lambda item: (domain_group(item["domain_id"]), item["domain_id"]))
    cards = []
    col_heights = [0, 0, 0]
    for idx, domain in enumerate(domains):
        col = idx % 3
        x = x_positions[col]
        y = y_start + col_heights[col]
        cards.append(domain_card(domain, x, y, card_width))
        col_heights[col] += y_gap + (22 if domain["domain_id"] == "self_harm_and_wellbeing" else 0)
    height = y_start + max(col_heights) + 130

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="RAMP safety taxonomy map">
  <style>
    .bg {{ fill: #f8fafc; }}
    .eyebrow {{ font: 700 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing: .08em; text-transform: uppercase; fill: #64748b; }}
    .title {{ font: 800 44px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #0f172a; }}
    .subtitle {{ font: 500 18px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #475569; }}
    .domain-card rect {{ filter: drop-shadow(0 18px 32px rgba(15, 23, 42, .07)); }}
    .highlighted rect {{ filter: drop-shadow(0 18px 38px rgba(180, 83, 9, .16)); }}
    .group-label {{ font: 700 11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing: .06em; text-transform: uppercase; fill: #64748b; }}
    .domain-title {{ font: 800 18px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #0f172a; }}
    .pill-text {{ font: 700 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .mini-label {{ font: 500 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #475569; }}
    .dark-title {{ font: 800 20px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #ffffff; }}
    .dark-copy {{ font: 500 15px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #cbd5e1; }}
    .legend-title {{ font: 800 16px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #0f172a; }}
  </style>
  <rect class="bg" width="{width}" height="{height}"/>
  <text class="eyebrow" x="56" y="70">RAMP taxonomy v0.1</text>
  <text class="title" x="54" y="124">Safety clusters and near-neighbor contrasts</text>
  <text class="subtitle" x="56" y="162">Harmful clusters define risky intent; benign near-neighbors define the safe boundary cases RAMP must avoid overblocking.</text>
  {pill(56, 190, "harmful clusters", color=ROLE_STYLE["harmful"]["color"], bg=ROLE_STYLE["harmful"]["bg"], width=146)}
  {pill(218, 190, "benign near-neighbors", color=ROLE_STYLE["benign_near_neighbors"]["color"], bg=ROLE_STYLE["benign_near_neighbors"]["bg"], width=190)}
  {correction_panel(corrections, 966, 62, 420)}
  {''.join(cards)}
</svg>"""


def html_page(svg: str, taxonomy: dict[str, Any]) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAMP Safety Taxonomy Map</title>
  <style>
    body {{
      margin: 0;
      background: #e5edf6;
      color: #0f172a;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 28px;
    }}
    .frame {{
      overflow: auto;
      border: 1px solid #d6dee8;
      border-radius: 24px;
      background: white;
      box-shadow: 0 26px 70px rgba(15, 23, 42, .14);
    }}
    svg {{
      display: block;
      width: 100%;
      min-width: 1180px;
      height: auto;
    }}
    .meta {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin: 18px 4px 0;
      color: #475569;
      font-size: 14px;
    }}
    .meta span {{
      border: 1px solid #cbd5e1;
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(255, 255, 255, .78);
    }}
  </style>
</head>
<body>
  <main>
    <div class="frame">{svg}</div>
    <div class="meta">
      <span>{html.escape(taxonomy["taxonomy_id"])}</span>
      <span>{len(taxonomy["domains"])} domains</span>
      <span>harmful vs benign near-neighbor contrast map</span>
    </div>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    taxonomy = load_json(Path(args.taxonomy))
    corrections = load_json(Path(args.corrections)) if args.corrections else None
    svg = build_svg(taxonomy, corrections)

    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_page(svg, taxonomy), encoding="utf-8")
    print(f"wrote taxonomy HTML visual to {output_html}")

    if args.output_svg:
        output_svg = Path(args.output_svg)
        output_svg.parent.mkdir(parents=True, exist_ok=True)
        output_svg.write_text(svg, encoding="utf-8")
        print(f"wrote taxonomy SVG visual to {output_svg}")


if __name__ == "__main__":
    main()
