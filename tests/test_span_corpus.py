from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TAXONOMY_PATH = Path("data/taxonomy/ramp_taxonomy_v0_1.json")
CORPUS_PATH = Path("data/span_corpus/ramp_span_corpus_v0.jsonl")
SOURCE_MANIFEST_PATH = Path("data/span_corpus/source_manifest_v0.json")

REQUIRED_FIELDS = {
    "id",
    "span_text",
    "domain",
    "subcluster_role",
    "subcluster_id",
    "label",
    "source",
    "source_record_id",
    "source_record_hash",
    "span_derivation",
    "raw_prompt_stored",
    "license",
    "safety_redaction",
    "policy_mapping",
    "harm_severity",
    "actionability",
    "intent_confidence",
    "reviewer_notes",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def test_span_corpus_records_match_taxonomy() -> None:
    taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    source_manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    records = load_jsonl(CORPUS_PATH)

    domain_map = {domain["domain_id"]: domain for domain in taxonomy["domains"]}
    source_ids = {source["source_id"] for source in source_manifest["sources"]}
    overlays = taxonomy["overlays"]
    overlay_values = {
        value
        for key in ("ambiguous", "evasion", "optimization")
        for value in overlays[key]
    }
    valid_labels = set(overlays["severity"])
    valid_harm_severity = set(overlays["harm_severity"])
    valid_actionability = set(overlays["actionability"])
    valid_intent_confidence = set(overlays["intent_confidence"])

    assert len(records) >= 30
    assert len({record["id"] for record in records}) == len(records)

    roles_seen: set[str] = set()
    labels_seen: set[str] = set()
    domains_seen: set[str] = set()

    for record in records:
        assert REQUIRED_FIELDS <= record.keys()
        assert record["span_text"].strip()
        assert record["domain"] in domain_map
        assert record["label"] in valid_labels
        assert record["harm_severity"] in valid_harm_severity
        assert record["actionability"] in valid_actionability
        assert record["intent_confidence"] in valid_intent_confidence
        assert isinstance(record["policy_mapping"], list)
        assert record["policy_mapping"]
        assert record["source"] in source_ids
        assert record["source_record_id"]
        assert record["source_record_hash"].startswith("sha256:")
        assert record["span_derivation"] in {
            "synthetic_non_instructional",
            "verbatim_safe_span",
            "redacted_span",
            "abstracted_by_reviewer",
        }
        assert isinstance(record["raw_prompt_stored"], bool)
        assert record["license"]
        assert record["safety_redaction"] in {"none", "redacted", "abstracted", "restricted"}

        domain = domain_map[record["domain"]]
        role = record["subcluster_role"]
        subcluster_id = record["subcluster_id"]

        if role == "harmful":
            assert subcluster_id in domain["harmful"]
        elif role == "benign_near_neighbor":
            assert subcluster_id in domain["benign_near_neighbors"]
        elif role in {"ambiguous", "evasion", "optimization"}:
            assert subcluster_id in overlays[role]
        else:
            raise AssertionError(f"unknown subcluster_role: {role}")

        for overlay in record.get("overlays", []):
            assert overlay in {"ambiguous", "evasion", "optimization"}
            if subcluster_id in overlay_values:
                assert subcluster_id in overlays[overlay]

        roles_seen.add(role)
        labels_seen.add(record["label"])
        domains_seen.add(record["domain"])

    assert {"harmful", "benign_near_neighbor", "ambiguous", "evasion", "optimization"} <= roles_seen
    assert {"safe", "controversial", "unsafe"} <= labels_seen
    assert set(domain_map) <= domains_seen


def test_source_manifest_has_expected_benchmark_coverage() -> None:
    manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    source_ids = {source["source_id"] for source in manifest["sources"]}

    assert {
        "xstest",
        "wildguardmix",
        "harmbench",
        "jailbreakbench",
        "strongreject",
        "cyberseceval",
        "wmdp",
        "agentharm",
    } <= source_ids

    for source in manifest["sources"]:
        assert source["name"]
        assert source["url"]
        assert source["priority"] in {"seed", "high", "medium", "restricted"}
        assert source["recommended_use"]
        assert source["domains"]
        assert source["roles"]
