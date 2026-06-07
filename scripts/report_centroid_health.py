#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

Vector = list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report health metrics for RAMP centroid artifacts."
    )
    parser.add_argument("--centroids", required=True, help="Centroid artifact JSON.")
    parser.add_argument("--output", default=None, help="Optional JSON report output path.")
    parser.add_argument("--min-count", type=int, default=25)
    parser.add_argument("--source-concentration-threshold", type=float, default=0.90)
    parser.add_argument("--collision-threshold", type=float, default=0.82)
    return parser.parse_args()


def cosine(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimensionality")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def nearest(
    centroid: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, float]:
    if not candidates:
        return None, 0.0
    return max(
        (
            (candidate, cosine(centroid["centroid"], candidate["centroid"]))
            for candidate in candidates
        ),
        key=lambda item: item[1],
    )


def centroid_id(centroid: dict[str, Any]) -> str:
    return (
        f"{centroid['domain']}/"
        f"{centroid['subcluster_role']}/"
        f"{centroid['subcluster_id']}"
    )


def compact_centroid(centroid: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": centroid_id(centroid),
        "domain": centroid["domain"],
        "subcluster_role": centroid["subcluster_role"],
        "subcluster_id": centroid["subcluster_id"],
        "count": centroid["count"],
    }


def source_concentration(centroid: dict[str, Any]) -> dict[str, Any]:
    source_counts = centroid.get("source_counts", {})
    if not source_counts:
        return {"source": None, "count": 0, "share": 0.0}
    source, count = max(source_counts.items(), key=lambda item: item[1])
    total = sum(source_counts.values())
    return {"source": source, "count": count, "share": count / total if total else 0.0}


def build_report(
    artifact: dict[str, Any],
    *,
    min_count: int,
    source_concentration_threshold: float,
    collision_threshold: float,
) -> dict[str, Any]:
    centroids = artifact["centroids"]
    risk = [
        centroid
        for centroid in centroids
        if centroid["subcluster_role"] in {"harmful", "evasion", "optimization"}
    ]
    benign = [
        centroid for centroid in centroids if centroid["subcluster_role"] == "benign_near_neighbor"
    ]

    role_counts = Counter(centroid["subcluster_role"] for centroid in centroids)
    role_rows: Counter[str] = Counter()
    domain_role_rows: dict[str, Counter[str]] = defaultdict(Counter)
    source_rows: Counter[str] = Counter()
    for centroid in centroids:
        role = centroid["subcluster_role"]
        role_rows[role] += centroid["count"]
        domain_role_rows[centroid["domain"]][role] += centroid["count"]
        source_rows.update(centroid.get("source_counts", {}))

    low_count = [
        {
            **compact_centroid(centroid),
            "warning": "below_min_count",
        }
        for centroid in centroids
        if centroid["count"] < min_count
    ]

    concentrated = []
    for centroid in centroids:
        concentration = source_concentration(centroid)
        if concentration["share"] >= source_concentration_threshold:
            concentrated.append(
                {
                    **compact_centroid(centroid),
                    "dominant_source": concentration["source"],
                    "dominant_source_share": round(concentration["share"], 4),
                    "warning": "source_concentration",
                }
            )

    nearest_benign = []
    collisions = []
    for centroid in risk:
        same_domain_benign = [
            candidate for candidate in benign if candidate["domain"] == centroid["domain"]
        ]
        nearest_same_domain, same_domain_similarity = nearest(centroid, same_domain_benign)
        nearest_any, any_similarity = nearest(centroid, benign)
        record = {
            **compact_centroid(centroid),
            "nearest_same_domain_benign": (
                compact_centroid(nearest_same_domain) if nearest_same_domain else None
            ),
            "nearest_same_domain_similarity": round(same_domain_similarity, 6),
            "nearest_any_benign": compact_centroid(nearest_any) if nearest_any else None,
            "nearest_any_benign_similarity": round(any_similarity, 6),
            "missing_same_domain_benign": nearest_same_domain is None,
        }
        nearest_benign.append(record)
        if nearest_any is not None and any_similarity >= collision_threshold:
            collisions.append(
                {
                    **record,
                    "warning": "harm_benign_collision",
                    "collision_threshold": collision_threshold,
                }
            )

    missing_same_domain = [
        record for record in nearest_benign if record["missing_same_domain_benign"]
    ]

    return {
        "centroid_artifact_id": artifact.get("centroid_artifact_id"),
        "embedding_source_id": artifact.get("embedding_source_id"),
        "taxonomy_id": artifact.get("taxonomy_id"),
        "total_embedding_rows": artifact.get("total_embedding_rows"),
        "num_centroids": artifact.get("num_centroids", len(centroids)),
        "dimension": artifact.get("dimension"),
        "thresholds": {
            "min_count": min_count,
            "source_concentration_threshold": source_concentration_threshold,
            "collision_threshold": collision_threshold,
        },
        "role_counts": dict(sorted(role_counts.items())),
        "role_rows": dict(sorted(role_rows.items())),
        "source_rows": dict(sorted(source_rows.items())),
        "domain_role_rows": {
            domain: dict(sorted(counts.items()))
            for domain, counts in sorted(domain_role_rows.items())
        },
        "low_count_centroids": low_count,
        "source_concentration_warnings": concentrated,
        "nearest_benign_by_risk_centroid": nearest_benign,
        "harm_benign_collisions": collisions,
        "missing_same_domain_benign": missing_same_domain,
        "summary": {
            "num_low_count_centroids": len(low_count),
            "num_source_concentration_warnings": len(concentrated),
            "num_harm_benign_collisions": len(collisions),
            "num_missing_same_domain_benign": len(missing_same_domain),
        },
    }


def print_text_summary(report: dict[str, Any]) -> None:
    print(f"artifact: {report['centroid_artifact_id']}")
    print(f"rows: {report['total_embedding_rows']}")
    print(f"centroids: {report['num_centroids']}")
    print(f"dimension: {report['dimension']}")
    print(f"role_counts: {report['role_counts']}")
    print(f"role_rows: {report['role_rows']}")
    print(f"summary: {report['summary']}")
    if report["low_count_centroids"]:
        print("\nlow-count centroids:")
        for item in report["low_count_centroids"]:
            print(f"  {item['count']:5} {item['id']}")
    if report["missing_same_domain_benign"]:
        print("\nmissing same-domain benign anchors:")
        for item in report["missing_same_domain_benign"]:
            print(f"  {item['id']} -> nearest any benign {item['nearest_any_benign_similarity']}")
    if report["harm_benign_collisions"]:
        print("\nharm/benign collisions:")
        for item in report["harm_benign_collisions"]:
            nearest_any = item["nearest_any_benign"]
            print(
                f"  {item['id']} <-> {nearest_any['id']} "
                f"cos={item['nearest_any_benign_similarity']}"
            )


def main() -> None:
    args = parse_args()
    artifact = json.loads(Path(args.centroids).read_text(encoding="utf-8"))
    report = build_report(
        artifact,
        min_count=args.min_count,
        source_concentration_threshold=args.source_concentration_threshold,
        collision_threshold=args.collision_threshold,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote centroid health report to {output_path}")
    print_text_summary(report)


if __name__ == "__main__":
    main()
