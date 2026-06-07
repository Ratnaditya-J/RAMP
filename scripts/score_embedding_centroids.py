#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from ramp.features.embedding_risk import (
    EmbeddingCluster,
    center_and_normalize,
    clusters_from_centroid_artifact,
    load_centroid_artifact_json,
    nearest_cluster,
    nearest_cluster_or_none,
    normalize,
)

Vector = tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-score precomputed embeddings against RAMP centroids."
    )
    parser.add_argument("--embeddings", required=True, help="Embedding JSONL to score.")
    parser.add_argument("--centroids", required=True, help="Centroid artifact JSON.")
    parser.add_argument("--output", required=True, help="Output scored JSONL.")
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional summary JSON output. Defaults to OUTPUT.summary.json.",
    )
    parser.add_argument(
        "--similarity-mode",
        choices=["cosine", "centered_cosine"],
        default="cosine",
    )
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def split_clusters(clusters: list[EmbeddingCluster]) -> tuple[
    list[EmbeddingCluster],
    list[EmbeddingCluster],
    list[EmbeddingCluster],
    list[EmbeddingCluster],
]:
    harmful = [cluster for cluster in clusters if cluster.subcluster_role == "harmful"]
    benign = [cluster for cluster in clusters if cluster.subcluster_role == "benign_near_neighbor"]
    evasion = [cluster for cluster in clusters if cluster.subcluster_role == "evasion"]
    optimization = [cluster for cluster in clusters if cluster.subcluster_role == "optimization"]
    return harmful, benign, evasion, optimization


def prepare_clusters(
    clusters: list[EmbeddingCluster],
    *,
    similarity_mode: str,
    corpus_mean_vector: Vector,
) -> list[EmbeddingCluster]:
    if similarity_mode == "cosine":
        return clusters
    return [
        EmbeddingCluster(
            cluster_id=cluster.cluster_id,
            category=cluster.category,
            centroid=center_and_normalize(cluster.centroid, corpus_mean_vector),
            kind=cluster.kind,
            version=cluster.version,
            description=cluster.description,
            harm_domain=cluster.harm_domain,
            subcluster_role=cluster.subcluster_role,
        )
        for cluster in clusters
    ]


def score_record(
    record: dict[str, Any],
    *,
    harmful: list[EmbeddingCluster],
    benign: list[EmbeddingCluster],
    evasion: list[EmbeddingCluster],
    optimization: list[EmbeddingCluster],
    similarity_mode: str,
    corpus_mean_vector: Vector,
) -> dict[str, Any]:
    vector = normalize(record["embedding"])
    if similarity_mode == "centered_cosine":
        vector = center_and_normalize(vector, corpus_mean_vector)

    harmful_cluster, harmful_similarity = nearest_cluster(vector, harmful)
    benign_cluster, benign_similarity = nearest_cluster(vector, benign)
    evasion_cluster, evasion_similarity = nearest_cluster_or_none(vector, evasion)
    optimization_cluster, optimization_similarity = nearest_cluster_or_none(vector, optimization)

    harmful_margin = harmful_similarity - benign_similarity
    evasion_margin = evasion_similarity - benign_similarity
    optimization_margin = optimization_similarity - benign_similarity
    risk_margin = max(harmful_margin, evasion_margin, optimization_margin)
    top_cluster = max(
        [
            (harmful_cluster, harmful_margin),
            (evasion_cluster, evasion_margin),
            (optimization_cluster, optimization_margin),
        ],
        key=lambda item: item[1],
    )[0]
    if top_cluster is None:
        top_cluster = harmful_cluster

    return {
        "id": record.get("id"),
        "label": record.get("label"),
        "source": record.get("source"),
        "domain": record.get("domain"),
        "subcluster_role": record.get("subcluster_role"),
        "subcluster_id": record.get("subcluster_id"),
        "span_text": record.get("span_text"),
        "similarity_mode": similarity_mode,
        "harm_similarity": harmful_similarity,
        "benign_similarity": benign_similarity,
        "evasion_similarity": evasion_similarity,
        "optimization_similarity": optimization_similarity,
        "harm_minus_benign_margin": harmful_margin,
        "risk_margin": risk_margin,
        "top_harm_cluster": harmful_cluster.cluster_id,
        "top_benign_cluster": benign_cluster.cluster_id,
        "top_evasion_cluster": evasion_cluster.cluster_id if evasion_cluster else None,
        "top_optimization_cluster": (
            optimization_cluster.cluster_id if optimization_cluster else None
        ),
        "top_risk_cluster": top_cluster.cluster_id,
        "top_risk_domain": top_cluster.harm_domain,
        "top_risk_role": top_cluster.subcluster_role,
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * pct)))
    return sorted_values[idx]


def build_summary(scored: list[dict[str, Any]], *, similarity_mode: str) -> dict[str, Any]:
    labels = Counter(str(row.get("label", "unknown")) for row in scored)
    roles = Counter(str(row.get("subcluster_role", "unknown")) for row in scored)
    by_label: dict[str, list[float]] = defaultdict(list)
    by_role: dict[str, list[float]] = defaultdict(list)
    top_risk = Counter(str(row["top_risk_cluster"]) for row in scored)
    for row in scored:
        margin = float(row["risk_margin"])
        by_label[str(row.get("label", "unknown"))].append(margin)
        by_role[str(row.get("subcluster_role", "unknown"))].append(margin)

    def distribution(values: list[float]) -> dict[str, float | int]:
        return {
            "count": len(values),
            "mean": mean(values) if values else 0.0,
            "p10": percentile(values, 0.10),
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
        }

    return {
        "similarity_mode": similarity_mode,
        "num_records": len(scored),
        "labels": dict(sorted(labels.items())),
        "subcluster_roles": dict(sorted(roles.items())),
        "risk_margin_by_label": {
            label: distribution(values) for label, values in sorted(by_label.items())
        },
        "risk_margin_by_subcluster_role": {
            role: distribution(values) for role, values in sorted(by_role.items())
        },
        "top_risk_clusters": dict(top_risk.most_common(20)),
    }


def main() -> None:
    args = parse_args()
    embeddings_path = Path(args.embeddings)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output or f"{args.output}.summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = load_centroid_artifact_json(args.centroids)
    corpus_mean_vector = tuple(float(value) for value in artifact.get("corpus_mean_vector", ()))
    if args.similarity_mode == "centered_cosine" and not corpus_mean_vector:
        raise ValueError("centered_cosine requires artifact['corpus_mean_vector']")

    clusters = prepare_clusters(
        clusters_from_centroid_artifact(artifact),
        similarity_mode=args.similarity_mode,
        corpus_mean_vector=corpus_mean_vector,
    )
    harmful, benign, evasion, optimization = split_clusters(clusters)

    scored: list[dict[str, Any]] = []
    with embeddings_path.open(encoding="utf-8") as input_file, output_path.open(
        "w", encoding="utf-8"
    ) as output_file:
        for idx, line in enumerate(input_file):
            if args.max_records is not None and idx >= args.max_records:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            scored_record = score_record(
                record,
                harmful=harmful,
                benign=benign,
                evasion=evasion,
                optimization=optimization,
                similarity_mode=args.similarity_mode,
                corpus_mean_vector=corpus_mean_vector,
            )
            scored.append(scored_record)
            output_file.write(json.dumps(scored_record, separators=(",", ":")) + "\n")

    summary = build_summary(scored, similarity_mode=args.similarity_mode)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(scored)} scored rows to {output_path}")
    print(f"wrote summary to {summary_path}")
    print(json.dumps(summary["risk_margin_by_label"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
