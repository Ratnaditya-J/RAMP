#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

Vector = list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RAMP embedding centroids.")
    parser.add_argument(
        "--embeddings",
        required=True,
        help="Embedding JSONL produced by gpt-oss extractor.",
    )
    parser.add_argument("--embedding-source", required=True, help="Embedding source config JSON.")
    parser.add_argument("--taxonomy", required=True, help="RAMP taxonomy JSON.")
    parser.add_argument("--output", required=True, help="Output centroid JSON.")
    parser.add_argument("--artifact-id", default="ramp_embedding_centroids_v0.1")
    parser.add_argument("--min-count-warning", type=int, default=25)
    parser.add_argument(
        "--max-centroid-preview",
        type=int,
        default=0,
        help="If >0, also store the first N vector values for easier manual inspection.",
    )
    return parser.parse_args()


def l2_normalize(vector: Iterable[float]) -> Vector:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]


def add_in_place(total: Vector | None, vector: Vector) -> Vector:
    if total is None:
        return list(vector)
    if len(total) != len(vector):
        raise ValueError(f"dimension mismatch: {len(total)} != {len(vector)}")
    for idx, value in enumerate(vector):
        total[idx] += value
    return total


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def centroid_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record["domain"]),
        str(record["subcluster_role"]),
        str(record["subcluster_id"]),
    )


def main() -> None:
    args = parse_args()
    embeddings_path = Path(args.embeddings)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embedding_source = load_json(Path(args.embedding_source))
    taxonomy = load_json(Path(args.taxonomy))

    sums: dict[tuple[str, str, str], Vector] = {}
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    labels: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sources: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    first_provenance: dict[str, Any] | None = None
    dimension: int | None = None
    total_rows = 0

    with embeddings_path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            record = json.loads(line)
            total_rows += 1
            vector = l2_normalize(record["embedding"])
            dimension = dimension or len(vector)
            key = centroid_key(record)
            sums[key] = add_in_place(sums.get(key), vector)
            counts[key] += 1
            labels[key][str(record.get("label", "unknown"))] += 1
            sources[key][str(record.get("source", "unknown"))] += 1
            if first_provenance is None:
                first_provenance = record.get("provenance", {})

    centroids: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for key in sorted(sums):
        domain, subcluster_role, subcluster_id = key
        count = counts[key]
        centroid = l2_normalize(value / count for value in sums[key])
        centroid_record: dict[str, Any] = {
            "domain": domain,
            "subcluster_role": subcluster_role,
            "subcluster_id": subcluster_id,
            "count": count,
            "dimension": len(centroid),
            "centroid": centroid,
            "label_counts": dict(sorted(labels[key].items())),
            "source_counts": dict(sorted(sources[key].items())),
        }
        if args.max_centroid_preview > 0:
            centroid_record["centroid_preview"] = centroid[: args.max_centroid_preview]
        centroids.append(centroid_record)
        if count < args.min_count_warning:
            warnings.append(
                {
                    "domain": domain,
                    "subcluster_role": subcluster_role,
                    "subcluster_id": subcluster_id,
                    "count": count,
                    "warning": "below_min_count_warning",
                }
            )

    artifact = {
        "centroid_artifact_id": args.artifact_id,
        "created_at": datetime.now(UTC).isoformat(),
        "embedding_source_id": embedding_source["embedding_source_id"],
        "taxonomy_id": taxonomy["taxonomy_id"],
        "input_embeddings": str(embeddings_path),
        "total_embedding_rows": total_rows,
        "num_centroids": len(centroids),
        "dimension": dimension,
        "centroid_method": "mean_of_l2_normalized_vectors_then_l2_normalize",
        "min_count_warning": args.min_count_warning,
        "warnings": warnings,
        "source_provenance_sample": first_provenance or {},
        "centroids": centroids,
    }
    output_path.write_text(json.dumps(artifact, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {len(centroids)} centroids from {total_rows} rows to {output_path}")
    if warnings:
        print(f"warnings: {len(warnings)} centroids below min-count {args.min_count_warning}")


if __name__ == "__main__":
    main()
