#!/usr/bin/env python
# ruff: noqa: E501
from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

Vector = list[float]


PALETTE = [
    "#2563eb",
    "#dc2626",
    "#059669",
    "#7c3aed",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4d7c0f",
    "#9333ea",
    "#0f766e",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a self-contained HTML/SVG visualization for RAMP centroid artifacts."
    )
    parser.add_argument("--centroids", required=True, help="Centroid artifact JSON.")
    parser.add_argument("--output", required=True, help="Output HTML path.")
    parser.add_argument(
        "--title",
        default="RAMP GPT-OSS Input Embedding Centroids",
        help="Report title.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=0,
        help="Draw this many nearest-neighbor links per centroid.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="SVG width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=820,
        help="SVG height in pixels.",
    )
    return parser.parse_args()


def dot(left: Vector, right: Vector) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def norm(vector: Vector) -> float:
    return math.sqrt(dot(vector, vector))


def normalize(vector: Vector) -> Vector:
    length = norm(vector)
    if length == 0.0:
        return list(vector)
    return [value / length for value in vector]


def subtract(left: Vector, right: Vector) -> Vector:
    return [a - b for a, b in zip(left, right, strict=True)]


def mean_vector(vectors: list[Vector]) -> Vector:
    dimension = len(vectors[0])
    return [sum(vector[idx] for vector in vectors) / len(vectors) for idx in range(dimension)]


def matvec_from_centered(centered: list[Vector], vector: Vector) -> Vector:
    """Multiply vector by X^T X without materializing the covariance matrix."""

    result = [0.0 for _ in vector]
    for row in centered:
        row_dot = dot(row, vector)
        for idx, value in enumerate(row):
            result[idx] += value * row_dot
    return result


def first_component(centered: list[Vector], seed_index: int = 0, iterations: int = 80) -> Vector:
    dimension = len(centered[0])
    seed = centered[seed_index % len(centered)]
    vector = normalize(seed if norm(seed) > 0.0 else [1.0] + [0.0] * (dimension - 1))
    for _ in range(iterations):
        vector = normalize(matvec_from_centered(centered, vector))
    return vector


def pca2(vectors: list[Vector]) -> list[tuple[float, float]]:
    if len(vectors) < 2:
        return [(0.0, 0.0) for _ in vectors]

    mean = mean_vector(vectors)
    centered = [subtract(vector, mean) for vector in vectors]
    pc1 = first_component(centered, seed_index=0)

    def remove_component(row: Vector, component: Vector) -> Vector:
        projection = dot(row, component)
        return [value - projection * component[idx] for idx, value in enumerate(row)]

    deflated = [remove_component(row, pc1) for row in centered]
    pc2 = first_component(deflated, seed_index=1)
    return [(dot(row, pc1), dot(row, pc2)) for row in centered]


def scale_points(
    points: list[tuple[float, float]],
    *,
    width: int,
    height: int,
    margin: int = 96,
) -> list[tuple[float, float]]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x_span = max(max_x - min_x, 1e-9)
    y_span = max(max_y - min_y, 1e-9)
    usable_w = width - margin * 2
    usable_h = height - margin * 2
    return [
        (
            margin + ((x - min_x) / x_span) * usable_w,
            height - margin - ((y - min_y) / y_span) * usable_h,
        )
        for x, y in points
    ]


def css_class(value: str) -> str:
    safe = "".join(ch if ch.isalnum() else "-" for ch in value.lower()).strip("-")
    return safe or "unknown"


def role_label(role: str) -> str:
    return {
        "harmful": "Harmful",
        "benign_near_neighbor": "Benign near-neighbor",
        "evasion": "Evasion",
        "optimization": "Optimization",
    }.get(role, role.replace("_", " ").title())


def radius_for_count(count: int, max_count: int) -> float:
    if max_count <= 1:
        return 10.0
    return 7.0 + (math.log1p(count) / math.log1p(max_count)) * 12.0


def nearest_neighbor_edges(centroids: list[dict[str, Any]], neighbors: int) -> list[tuple[int, int, float]]:
    if neighbors <= 0:
        return []
    edges: set[tuple[int, int]] = set()
    scored_edges: list[tuple[int, int, float]] = []
    for idx, centroid in enumerate(centroids):
        scores = []
        for other_idx, other in enumerate(centroids):
            if idx == other_idx:
                continue
            similarity = dot(centroid["centroid"], other["centroid"])
            scores.append((similarity, other_idx))
        for similarity, other_idx in sorted(scores, reverse=True)[:neighbors]:
            edge = tuple(sorted((idx, other_idx)))
            if edge in edges:
                continue
            edges.add(edge)
            scored_edges.append((edge[0], edge[1], similarity))
    return scored_edges


def short_name(centroid: dict[str, Any]) -> str:
    return str(centroid["subcluster_id"]).replace("_", " ")


def indexed_centroids(centroids: list[dict[str, Any]]) -> list[tuple[int, int, dict[str, Any]]]:
    ranked = sorted(enumerate(centroids), key=lambda item: int(item[1]["count"]), reverse=True)
    return [(rank, original_idx, centroid) for rank, (original_idx, centroid) in enumerate(ranked, 1)]


def build_svg(
    artifact: dict[str, Any],
    centroids: list[dict[str, Any]],
    *,
    title: str,
    width: int,
    height: int,
    neighbors: int,
) -> str:
    points = scale_points(
        pca2([centroid["centroid"] for centroid in centroids]),
        width=width,
        height=height,
        margin=72,
    )
    domains = sorted({str(centroid["domain"]) for centroid in centroids})
    colors = {domain: PALETTE[idx % len(PALETTE)] for idx, domain in enumerate(domains)}
    max_count = max(int(centroid["count"]) for centroid in centroids)
    edges = nearest_neighbor_edges(centroids, neighbors)
    ranks_by_index = {original_idx: rank for rank, original_idx, _ in indexed_centroids(centroids)}

    edge_markup = []
    for left_idx, right_idx, similarity in edges:
        x1, y1 = points[left_idx]
        x2, y2 = points[right_idx]
        opacity = min(0.42, max(0.10, (similarity + 1.0) / 5.0))
        edge_markup.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'class="edge" opacity="{opacity:.3f}"><title>cosine {similarity:.3f}</title></line>'
        )

    node_markup = []
    for idx, centroid in enumerate(centroids):
        x, y = points[idx]
        role = str(centroid["subcluster_role"])
        domain = str(centroid["domain"])
        color = colors[domain]
        radius = radius_for_count(int(centroid["count"]), max_count)
        label = str(ranks_by_index[idx])
        tooltip = (
            f"{domain} / {role_label(role)} / {centroid['subcluster_id']} "
            f"({centroid['count']} spans)"
        )
        node_markup.append(
            f'<g class="node role-{css_class(role)}">'
            f"<title>{html.escape(tooltip)}</title>"
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" '
            f'fill="{color}" class="dot" />'
            f'<text x="{x:.2f}" y="{y + 4:.2f}" class="node-number">'
            f"{html.escape(label)}</text>"
            f"</g>"
        )

    return f"""
<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
  <rect width="{width}" height="{height}" class="canvas" />
  <g class="plot-frame">
    <line x1="72" y1="{height - 72}" x2="{width - 72}" y2="{height - 72}" class="axis" />
    <line x1="72" y1="72" x2="72" y2="{height - 72}" class="axis" />
    <text x="{width - 180}" y="{height - 32}" class="axis-label">PCA component 1</text>
    <text x="24" y="62" class="axis-label">PCA component 2</text>
  </g>
  <g class="edges">
    {"".join(edge_markup)}
  </g>
  <g class="nodes">
    {"".join(node_markup)}
  </g>
</svg>
""".strip()


def domain_colors(centroids: list[dict[str, Any]]) -> dict[str, str]:
    domains = sorted({str(centroid["domain"]) for centroid in centroids})
    return {domain: PALETTE[idx % len(PALETTE)] for idx, domain in enumerate(domains)}


def role_color(role: str) -> str:
    return {
        "harmful": "#dc2626",
        "benign_near_neighbor": "#059669",
        "evasion": "#7c3aed",
        "optimization": "#ea580c",
    }.get(role, "#64748b")


def build_centroid_index(centroids: list[dict[str, Any]]) -> str:
    rows = []
    for rank, _, centroid in indexed_centroids(centroids):
        domain = str(centroid["domain"])
        role = str(centroid["subcluster_role"])
        rows.append(
            "<li class='index-row'>"
            f"<span class='rank' style='background:{role_color(role)}'>{rank}</span>"
            "<span class='index-main'>"
            f"<strong>{html.escape(short_name(centroid))}</strong>"
            f"<em>{html.escape(domain.replace('_', ' '))}</em>"
            "</span>"
            f"<span class='role role-{css_class(role)}'>{html.escape(role_label(role))}</span>"
            f"<span class='count'>{html.escape(str(centroid['count']))}</span>"
            "</li>"
        )
    return "<ol class='centroid-index'>" + "\n".join(rows) + "</ol>"


def build_role_legend() -> str:
    items = []
    for role in ("harmful", "benign_near_neighbor", "evasion"):
        items.append(
            "<span class='domain-chip'>"
            f"<i style='background:{role_color(role)}'></i>"
            f"{html.escape(role_label(role))}"
            "</span>"
        )
    return "<div class='domain-legend'>" + "\n".join(items) + "</div>"


def harm_weight(role: str) -> float:
    return {
        "harmful": 1.0,
        "evasion": 0.85,
        "optimization": 0.75,
        "benign_near_neighbor": 0.28,
    }.get(role, 0.55)


def normalized_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x_span = max(max_x - min_x, 1e-9)
    y_span = max(max_y - min_y, 1e-9)
    return [((x - min_x) / x_span, 1.0 - ((y - min_y) / y_span)) for x, y in points]


def build_graph_data(centroids: list[dict[str, Any]]) -> str:
    ranks_by_index = {original_idx: rank for rank, original_idx, _ in indexed_centroids(centroids)}
    points = normalized_points(pca2([centroid["centroid"] for centroid in centroids]))
    max_count = max(int(centroid["count"]) for centroid in centroids)
    nodes = []
    for idx, centroid in enumerate(centroids):
        role = str(centroid["subcluster_role"])
        domain = str(centroid["domain"])
        count = int(centroid["count"])
        count_score = math.log1p(count) / math.log1p(max_count)
        weight = harm_weight(role)
        source_counts = dict(centroid.get("source_counts", {}))
        label_counts = dict(centroid.get("label_counts", {}))
        nodes.append(
            {
                "id": idx,
                "rank": ranks_by_index[idx],
                "x": points[idx][0],
                "y": points[idx][1],
                "domain": domain,
                "domainLabel": domain.replace("_", " "),
                "role": role,
                "roleLabel": role_label(role),
                "subcluster": str(centroid["subcluster_id"]),
                "subclusterLabel": short_name(centroid),
                "count": count,
                "countScore": count_score,
                "harmWeight": weight,
                "color": role_color(role),
                "labels": label_counts,
                "sources": source_counts,
            }
        )

    edges = []
    for left_idx, right_idx, similarity in nearest_neighbor_edges(centroids, 3):
        edges.append({"source": left_idx, "target": right_idx, "similarity": similarity})
    return json.dumps({"nodes": nodes, "edges": edges}, separators=(",", ":"))


def build_table(centroids: list[dict[str, Any]]) -> str:
    rows = []
    for centroid in sorted(centroids, key=lambda item: int(item["count"]), reverse=True):
        label_counts = ", ".join(f"{key}: {value}" for key, value in centroid["label_counts"].items())
        source_counts = ", ".join(
            f"{key}: {value}" for key, value in centroid["source_counts"].items()
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(centroid['count']))}</td>"
            f"<td>{html.escape(str(centroid['domain']).replace('_', ' '))}</td>"
            f"<td>{html.escape(role_label(str(centroid['subcluster_role'])))}</td>"
            f"<td>{html.escape(str(centroid['subcluster_id']).replace('_', ' '))}</td>"
            f"<td>{html.escape(label_counts)}</td>"
            f"<td>{html.escape(source_counts)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def build_summary(artifact: dict[str, Any], centroids: list[dict[str, Any]]) -> str:
    role_counts: dict[str, int] = defaultdict(int)
    domain_counts: dict[str, int] = defaultdict(int)
    for centroid in centroids:
        role_counts[str(centroid["subcluster_role"])] += int(centroid["count"])
        domain_counts[str(centroid["domain"])] += int(centroid["count"])

    warning_count = len(artifact.get("warnings", []))
    artifact_id = artifact.get("centroid_artifact_id", artifact.get("artifact_id", "unknown"))
    cards = [
        ("Artifact", str(artifact_id)),
        ("Centroids", str(len(centroids))),
        ("Rows", str(artifact.get("total_embedding_rows", "unknown"))),
        ("Dimensions", str(artifact.get("dimension", "unknown"))),
        ("Warnings", str(warning_count)),
    ]
    card_html = "".join(
        f"<div class='metric'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>"
        for label, value in cards
    )
    role_html = "".join(
        f"<li><strong>{html.escape(role_label(role))}</strong><span>{count}</span></li>"
        for role, count in sorted(role_counts.items())
    )
    domain_html = "".join(
        f"<li><strong>{html.escape(domain.replace('_', ' '))}</strong><span>{count}</span></li>"
        for domain, count in sorted(domain_counts.items(), key=lambda item: item[1], reverse=True)
    )
    return f"""
<section class="metrics">{card_html}</section>
<section class="split">
  <div>
    <h2>Role Coverage</h2>
    <ul class="coverage">{role_html}</ul>
  </div>
  <div>
    <h2>Domain Coverage</h2>
    <ul class="coverage">{domain_html}</ul>
  </div>
</section>
""".strip()


def build_html(artifact: dict[str, Any], *, title: str, width: int, height: int, neighbors: int) -> str:
    centroids = artifact["centroids"]
    artifact_id = artifact.get("centroid_artifact_id", artifact.get("artifact_id", "unknown"))
    build_svg(
        artifact,
        centroids,
        title=title,
        width=width,
        height=height,
        neighbors=neighbors,
    )
    summary = build_summary(artifact, centroids)
    centroid_index = build_centroid_index(centroids)
    legend = build_role_legend()
    graph_data = build_graph_data(centroids)
    table_rows = build_table(centroids)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{html.escape(title)}</title>
<style>
:root {{
  color-scheme: light;
  --ink: #172033;
  --muted: #64748b;
  --line: #d8dee9;
  --soft: #f6f8fb;
  --paper: #ffffff;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: #eef2f7;
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
main {{
  width: min(1440px, calc(100vw - 40px));
  margin: 28px auto 48px;
}}
header {{
  margin-bottom: 18px;
}}
h1 {{
  margin: 0 0 8px;
  font-size: 30px;
  line-height: 1.12;
  letter-spacing: 0;
}}
p {{
  margin: 0;
  color: var(--muted);
  max-width: 900px;
}}
.eyebrow {{
  color: var(--muted);
  font-size: 13px;
  font-weight: 750;
  margin-bottom: 8px;
  text-transform: uppercase;
}}
.panel {{
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
  overflow: hidden;
}}
.viz-layout {{
  border-bottom: 1px solid var(--line);
  display: grid;
  grid-template-columns: minmax(0, 1fr) 390px;
}}
.viz {{
  background: linear-gradient(180deg, #fbfcff 0%, #f8fafc 100%);
  padding: 18px;
  position: relative;
}}
.graph-toolbar {{
  align-items: center;
  display: flex;
  gap: 10px;
  justify-content: space-between;
  margin-bottom: 12px;
}}
.segmented {{
  background: #e8eef6;
  border: 1px solid #dbe4ee;
  border-radius: 999px;
  display: inline-flex;
  gap: 2px;
  padding: 3px;
}}
.segmented button {{
  background: transparent;
  border: 0;
  border-radius: 999px;
  color: #526173;
  cursor: pointer;
  font: inherit;
  font-size: 12px;
  font-weight: 800;
  padding: 7px 11px;
}}
.segmented button.active {{
  background: #ffffff;
  box-shadow: 0 1px 4px rgba(15, 23, 42, 0.14);
  color: #172033;
}}
.toolbar-note {{
  color: var(--muted);
  font-size: 12px;
}}
.side {{
  background: #ffffff;
  border-left: 1px solid var(--line);
  padding: 18px;
}}
.side h2 {{
  margin-bottom: 6px;
}}
.side p {{
  font-size: 13px;
  line-height: 1.45;
}}
svg {{
  display: block;
  width: 100%;
  height: auto;
}}
.canvas {{
  fill: transparent;
}}
.svg-title {{
  font-size: 22px;
  font-weight: 760;
  fill: var(--ink);
}}
.svg-subtitle, .axis-label, .legend-text {{
  font-size: 13px;
  fill: var(--muted);
}}
.axis {{
  stroke: #cbd5e1;
  stroke-width: 1;
}}
.edge {{
  stroke: #94a3b8;
  stroke-width: 1.5;
}}
.dot {{
  stroke: #ffffff;
  stroke-width: 2.25;
  opacity: 0.94;
  transition: opacity 140ms ease, stroke-width 140ms ease;
}}
.role-benign-near-neighbor .dot {{
  stroke-dasharray: 4 3;
}}
.role-evasion .dot {{
  stroke: #111827;
}}
.node-number {{
  fill: #ffffff;
  font-size: 11px;
  font-weight: 850;
  pointer-events: none;
  text-anchor: middle;
}}
.graph-node {{
  cursor: pointer;
}}
.graph-node.dimmed .dot,
.graph-edge.dimmed {{
  opacity: 0.14;
}}
.graph-node.active .dot {{
  stroke: #111827;
  stroke-width: 3.2;
}}
.graph-edge {{
  stroke: #94a3b8;
  stroke-linecap: round;
  stroke-width: 1.5;
  opacity: 0;
  pointer-events: none;
  transition: opacity 140ms ease;
}}
.graph-edge.active {{
  opacity: 0.62;
}}
.tooltip {{
  background: #111827;
  border-radius: 8px;
  box-shadow: 0 14px 34px rgba(15, 23, 42, 0.22);
  color: #ffffff;
  display: none;
  font-size: 12px;
  left: 0;
  max-width: 310px;
  padding: 10px 12px;
  pointer-events: none;
  position: absolute;
  top: 0;
  transform: translate(14px, 14px);
  z-index: 4;
}}
.tooltip strong {{
  display: block;
  font-size: 13px;
  margin-bottom: 4px;
}}
.tooltip span {{
  color: #cbd5e1;
  display: block;
  line-height: 1.35;
}}
.detail-card {{
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  margin-top: 14px;
  padding: 13px;
}}
.detail-card strong {{
  display: block;
  font-size: 15px;
  margin-bottom: 4px;
}}
.detail-card p {{
  font-size: 12px;
  line-height: 1.45;
  margin-top: 8px;
}}
.detail-grid {{
  display: grid;
  gap: 7px;
  grid-template-columns: 1fr 1fr;
  margin-top: 10px;
}}
.detail-grid span {{
  color: var(--muted);
  display: block;
  font-size: 10px;
  font-weight: 800;
  text-transform: uppercase;
}}
.detail-grid b {{
  display: block;
  font-size: 13px;
  margin-top: 2px;
}}
.legend-bg {{
  fill: rgba(255,255,255,0.86);
  stroke: #e2e8f0;
}}
.centroid-index {{
  list-style: none;
  margin: 16px 0 0;
  max-height: 650px;
  overflow: auto;
  padding: 0;
}}
.index-row {{
  align-items: center;
  border-bottom: 1px solid #edf2f7;
  display: grid;
  gap: 10px;
  grid-template-columns: 30px minmax(0, 1fr) auto 52px;
  padding: 9px 0;
}}
.index-row.active {{
  background: #f8fafc;
  margin-left: -8px;
  margin-right: -8px;
  padding-left: 8px;
  padding-right: 8px;
}}
.rank {{
  align-items: center;
  border-radius: 999px;
  color: #ffffff;
  display: inline-flex;
  font-size: 12px;
  font-weight: 850;
  height: 26px;
  justify-content: center;
  width: 26px;
}}
.index-main {{
  min-width: 0;
}}
.index-main strong {{
  display: block;
  font-size: 13px;
  line-height: 1.2;
}}
.index-main em {{
  color: var(--muted);
  display: block;
  font-size: 11px;
  font-style: normal;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
.role {{
  border: 1px solid #dbe4ee;
  border-radius: 999px;
  color: #334155;
  font-size: 10px;
  font-weight: 800;
  padding: 4px 7px;
  text-transform: uppercase;
  white-space: nowrap;
}}
.role-harmful {{
  background: #fff1f2;
  border-color: #fecdd3;
  color: #be123c;
}}
.role-benign-near-neighbor {{
  background: #ecfdf5;
  border-color: #bbf7d0;
  color: #047857;
}}
.role-evasion {{
  background: #f5f3ff;
  border-color: #ddd6fe;
  color: #6d28d9;
}}
.count {{
  color: #172033;
  font-size: 12px;
  font-variant-numeric: tabular-nums;
  font-weight: 850;
  text-align: right;
}}
.domain-legend {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 14px 18px;
}}
.domain-chip {{
  align-items: center;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 999px;
  color: #475569;
  display: inline-flex;
  font-size: 12px;
  gap: 7px;
  padding: 6px 9px;
}}
.domain-chip i {{
  border-radius: 999px;
  display: inline-block;
  height: 9px;
  width: 9px;
}}
.metrics {{
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 1px;
  border-top: 1px solid var(--line);
  background: var(--line);
}}
.metric {{
  background: #fff;
  padding: 18px;
}}
.metric span {{
  display: block;
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 6px;
  text-transform: uppercase;
}}
.metric strong {{
  display: block;
  overflow-wrap: anywhere;
  font-size: 18px;
}}
.split {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1px;
  border-top: 1px solid var(--line);
  background: var(--line);
}}
.split > div {{
  background: #fff;
  padding: 18px;
}}
h2 {{
  margin: 0 0 12px;
  font-size: 18px;
}}
.coverage {{
  list-style: none;
  margin: 0;
  padding: 0;
}}
.coverage li {{
  display: flex;
  justify-content: space-between;
  gap: 18px;
  padding: 7px 0;
  border-bottom: 1px solid #eef2f7;
}}
.table-wrap {{
  margin-top: 22px;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
}}
th, td {{
  padding: 11px 12px;
  border-bottom: 1px solid #eef2f7;
  text-align: left;
  vertical-align: top;
  font-size: 13px;
}}
th {{
  background: #f8fafc;
  color: #334155;
  font-size: 12px;
  text-transform: uppercase;
}}
td:first-child {{
  font-variant-numeric: tabular-nums;
  font-weight: 700;
}}
@media (max-width: 900px) {{
  main {{ width: min(100vw - 24px, 1440px); }}
  .viz-layout {{ grid-template-columns: 1fr; }}
  .side {{ border-left: 0; border-top: 1px solid var(--line); }}
  .metrics {{ grid-template-columns: 1fr 1fr; }}
  .split {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">{html.escape(str(artifact_id))} · {len(centroids)} centroids · {artifact.get("dimension", "unknown")} dimensions</div>
    <h1>{html.escape(title)}</h1>
    <p>Centroids are projected with deterministic two-component PCA from the full embedding vectors. Color encodes safety role, bubble size reflects the selected weight, and numbers map each point to the centroid index.</p>
  </header>
  <section class="panel">
    <div class="viz-layout">
      <div class="viz">
        <div class="graph-toolbar">
          <div class="segmented" aria-label="Bubble size">
            <button type="button" class="active" data-size-mode="harm">Harm-weighted</button>
            <button type="button" data-size-mode="count">Count</button>
            <button type="button" data-size-mode="equal">Equal</button>
          </div>
          <span class="toolbar-note">Hover to inspect. Click to pin.</span>
        </div>
        <svg id="centroid-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
          <rect width="{width}" height="{height}" class="canvas" />
          <g class="plot-frame">
            <line x1="72" y1="{height - 72}" x2="{width - 72}" y2="{height - 72}" class="axis" />
            <line x1="72" y1="72" x2="72" y2="{height - 72}" class="axis" />
            <text x="{width - 180}" y="{height - 32}" class="axis-label">PCA component 1</text>
            <text x="24" y="62" class="axis-label">PCA component 2</text>
          </g>
          <g id="edge-layer"></g>
          <g id="node-layer"></g>
        </svg>
        <div id="centroid-tooltip" class="tooltip"></div>
      </div>
      <aside class="side">
        <h2>Centroid Index</h2>
        <p>Numbers are ranked by corpus count. Red means harmful, green means benign near-neighbor, and purple means evasion. Bubble area can encode harm-weighted mass, raw count, or equal centroid presence.</p>
        {centroid_index}
        <div id="centroid-detail" class="detail-card"></div>
      </aside>
    </div>
    {legend}
    {summary}
  </section>
  <section class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Count</th>
          <th>Domain</th>
          <th>Role</th>
          <th>Subcluster</th>
          <th>Labels</th>
          <th>Sources</th>
        </tr>
      </thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>
  </section>
</main>
<script>
const GRAPH = {graph_data};
const WIDTH = {width};
const HEIGHT = {height};
const MARGIN = 72;
const svg = document.getElementById("centroid-svg");
const edgeLayer = document.getElementById("edge-layer");
const nodeLayer = document.getElementById("node-layer");
const tooltip = document.getElementById("centroid-tooltip");
const detail = document.getElementById("centroid-detail");
let sizeMode = "harm";
let pinned = null;

function radius(node) {{
  if (sizeMode === "equal") return 13;
  if (sizeMode === "count") return 8 + node.countScore * 26;
  return 7 + Math.sqrt(node.countScore * node.harmWeight) * 31;
}}

function initialLayout() {{
  const usableW = WIDTH - MARGIN * 2;
  const usableH = HEIGHT - MARGIN * 2;
  return GRAPH.nodes.map(node => ({{
    ...node,
    px: MARGIN + node.x * usableW,
    py: MARGIN + node.y * usableH,
  }}));
}}

function relax(nodes) {{
  for (let step = 0; step < 260; step++) {{
    for (let i = 0; i < nodes.length; i++) {{
      for (let j = i + 1; j < nodes.length; j++) {{
        const a = nodes[i];
        const b = nodes[j];
        const dx = b.px - a.px;
        const dy = b.py - a.py;
        const d = Math.max(Math.sqrt(dx * dx + dy * dy), 0.01);
        const minD = radius(a) + radius(b) + 10;
        if (d < minD) {{
          const push = (minD - d) * 0.52;
          const ux = dx / d;
          const uy = dy / d;
          a.px -= ux * push;
          a.py -= uy * push;
          b.px += ux * push;
          b.py += uy * push;
        }}
      }}
    }}
    for (const node of nodes) {{
      const targetX = MARGIN + node.x * (WIDTH - MARGIN * 2);
      const targetY = MARGIN + node.y * (HEIGHT - MARGIN * 2);
      node.px += (targetX - node.px) * 0.025;
      node.py += (targetY - node.py) * 0.025;
      const r = radius(node);
      node.px = Math.max(MARGIN + r, Math.min(WIDTH - MARGIN - r, node.px));
      node.py = Math.max(MARGIN + r, Math.min(HEIGHT - MARGIN - r, node.py));
    }}
  }}
  return nodes;
}}

function formatMap(obj) {{
  return Object.entries(obj).map(([key, value]) => `${{key}}: ${{value}}`).join(", ");
}}

function renderDetail(node) {{
  detail.innerHTML = `
    <strong>${{node.rank}}. ${{node.subclusterLabel}}</strong>
    <span class="role role-${{node.role.replaceAll("_", "-")}}">${{node.roleLabel}}</span>
    <div class="detail-grid">
      <div><span>Domain</span><b>${{node.domainLabel}}</b></div>
      <div><span>Spans</span><b>${{node.count.toLocaleString()}}</b></div>
      <div><span>Harm weight</span><b>${{node.harmWeight.toFixed(2)}}</b></div>
      <div><span>Count score</span><b>${{node.countScore.toFixed(2)}}</b></div>
    </div>
    <p><b>Labels:</b> ${{formatMap(node.labels)}}</p>
    <p><b>Sources:</b> ${{formatMap(node.sources)}}</p>
  `;
}}

function neighborsOf(id) {{
  const ids = new Set([id]);
  for (const edge of GRAPH.edges) {{
    if (edge.source === id) ids.add(edge.target);
    if (edge.target === id) ids.add(edge.source);
  }}
  return ids;
}}

function setActive(node, event) {{
  const ids = node ? neighborsOf(node.id) : new Set();
  document.querySelectorAll(".graph-node").forEach(el => {{
    const active = node && Number(el.dataset.id) === node.id;
    const near = node && ids.has(Number(el.dataset.id));
    el.classList.toggle("active", active);
    el.classList.toggle("dimmed", Boolean(node) && !near);
  }});
  document.querySelectorAll(".graph-edge").forEach(el => {{
    const active = node && (Number(el.dataset.source) === node.id || Number(el.dataset.target) === node.id);
    el.classList.toggle("active", active);
    el.classList.toggle("dimmed", Boolean(node) && !active);
  }});
  document.querySelectorAll(".index-row").forEach((el, idx) => {{
    el.classList.toggle("active", node && idx + 1 === node.rank);
  }});
  if (node) renderDetail(node);
  if (node && event) {{
    tooltip.style.display = "block";
    tooltip.style.left = `${{event.offsetX}}px`;
    tooltip.style.top = `${{event.offsetY}}px`;
    tooltip.innerHTML = `<strong>${{node.rank}}. ${{node.subclusterLabel}}</strong><span>${{node.domainLabel}} · ${{node.roleLabel}}</span><span>${{node.count.toLocaleString()}} spans · harm weight ${{node.harmWeight.toFixed(2)}}</span>`;
  }} else {{
    tooltip.style.display = "none";
  }}
}}

function draw() {{
  const nodes = relax(initialLayout());
  const byId = new Map(nodes.map(node => [node.id, node]));
  edgeLayer.innerHTML = "";
  nodeLayer.innerHTML = "";
  for (const edge of GRAPH.edges) {{
    const a = byId.get(edge.source);
    const b = byId.get(edge.target);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", a.px);
    line.setAttribute("y1", a.py);
    line.setAttribute("x2", b.px);
    line.setAttribute("y2", b.py);
    line.setAttribute("class", "graph-edge");
    line.dataset.source = edge.source;
    line.dataset.target = edge.target;
    edgeLayer.appendChild(line);
  }}
  for (const node of nodes) {{
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", `graph-node role-${{node.role.replaceAll("_", "-")}}`);
    g.dataset.id = node.id;
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("class", "dot");
    c.setAttribute("cx", node.px);
    c.setAttribute("cy", node.py);
    c.setAttribute("r", radius(node));
    c.setAttribute("fill", node.color);
    const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("class", "node-number");
    t.setAttribute("x", node.px);
    t.setAttribute("y", node.py + 4);
    t.textContent = node.rank;
    g.appendChild(c);
    g.appendChild(t);
    g.addEventListener("mouseenter", event => setActive(node, event));
    g.addEventListener("mousemove", event => setActive(node, event));
    g.addEventListener("mouseleave", () => {{ if (!pinned) setActive(null); }});
    g.addEventListener("click", event => {{
      pinned = pinned && pinned.id === node.id ? null : node;
      setActive(pinned, event);
    }});
    nodeLayer.appendChild(g);
  }}
  renderDetail(nodes.slice().sort((a, b) => a.rank - b.rank)[0]);
}}

document.querySelectorAll("[data-size-mode]").forEach(button => {{
  button.addEventListener("click", () => {{
    sizeMode = button.dataset.sizeMode;
    pinned = null;
    document.querySelectorAll("[data-size-mode]").forEach(item => item.classList.toggle("active", item === button));
    setActive(null);
    draw();
  }});
}});

draw();
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    input_path = Path(args.centroids)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = json.loads(input_path.read_text(encoding="utf-8"))
    output_path.write_text(
        build_html(
            artifact,
            title=args.title,
            width=args.width,
            height=args.height,
            neighbors=args.neighbors,
        ),
        encoding="utf-8",
    )
    print(f"wrote centroid visualization to {output_path}")


if __name__ == "__main__":
    main()
