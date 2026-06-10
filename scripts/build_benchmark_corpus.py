#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build benchmark-derived RAMP span corpus.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-per-source", type=int, default=10_000_000)
    parser.add_argument("--include-extended", action="store_true")
    return parser.parse_args()


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def infer_text(row: dict[str, Any]) -> str:
    for key in (
        "prompt",
        "instruction",
        "question",
        "text",
        "user_request",
        "Behavior",
        "behavior",
    ):
        value = row.get(key)
        if value:
            return clean(value)
    messages = row.get("messages")
    if isinstance(messages, list):
        return clean(" ".join(str(item.get("content", item)) for item in messages))
    return ""


def infer_harmful(row: dict[str, Any], default: bool = False) -> bool:
    for key in ("prompt_harm_label", "is_harmful", "harmful", "is_unsafe", "unsafe", "label"):
        if key not in row:
            continue
        value = str(row[key]).lower()
        if any(token in value for token in ("unsafe", "harmful", "toxic", "jailbreak")):
            return True
        if any(token in value for token in ("safe", "benign", "harmless")):
            return False
        return value in {"1", "true", "yes"}
    return default


def domain_guess(text: str) -> tuple[str, str, str]:
    lowered = text.lower()
    if any(term in lowered for term in ("malware", "phishing", "credential", "exploit", "hack")):
        return "cyber_abuse", "harmful", "vulnerability_exploitation"
    if any(term in lowered for term in ("chemical", "biological", "pathogen", "toxin", "nuclear")):
        return "cbrn_and_hazardous_materials", "harmful", "chemical_misuse_procedure"
    if any(term in lowered for term in ("weapon", "explosive", "bomb", "firearm", "attack")):
        return "weapons_and_physical_violence", "harmful", "weapon_construction"
    if any(
        term in lowered
        for term in (
            "suicide",
            "suicidal",
            "self-harm",
            "self harm",
            "taking my own life",
            "take my own life",
            "end my life",
            "kill myself",
        )
    ):
        if any(
            term in lowered
            for term in ("method", "methods", "how to", "step", "instructions", "painless")
        ):
            return "self_harm_and_wellbeing", "harmful", "suicide_methods"
        return "self_harm_and_wellbeing", "harmful", "suicidal_ideation_or_crisis"
    if any(term in lowered for term in ("child", "minor", "underage")):
        return "child_safety", "harmful", "sexualized_minors"
    if any(term in lowered for term in ("dox", "private", "token", "api key", "ssn")):
        return "privacy_identity_and_secrets", "harmful", "pii_extraction"
    if any(term in lowered for term in ("vote", "election", "voter", "ballot")):
        return "misinformation_manipulation_and_civic", "harmful", "election_falsehoods"
    if any(term in lowered for term in ("copyright", "verbatim", "book", "lyrics")):
        return "ip_and_content_rights", "harmful", "copyright_reproduction"
    if any(term in lowered for term in ("doctor", "medical", "lawyer", "legal", "financial")):
        return "regulated_advice", "harmful", "unsafe_professional_instruction"
    if any(term in lowered for term in ("jailbreak", "system prompt", "ignore previous", "bypass")):
        return "agent_tool_and_system_integrity", "evasion", "jailbreak"
    if any(term in lowered for term in ("hate", "slur", "harass", "humiliate")):
        return "hate_harassment_and_abuse", "harmful", "targeted_harassment"
    if any(term in lowered for term in ("sex", "sexual", "intimate")):
        return "sexual_safety_and_content", "harmful", "explicit_adult_generation"
    if any(term in lowered for term in ("fraud", "scam", "forge", "steal")):
        return "nonviolent_illegal_activity", "harmful", "fraud_scams"
    return "nonviolent_illegal_activity", "harmful", "organized_abuse_workflows"


def benign_guess(text: str) -> tuple[str, str, str]:
    lowered = text.lower()
    if any(term in lowered for term in ("security", "patch", "defend", "incident", "password")):
        return "cyber_abuse", "benign_near_neighbor", "defensive_security"
    if any(term in lowered for term in ("chemistry", "lab", "safety", "public health")):
        return "cbrn_and_hazardous_materials", "benign_near_neighbor", "lab_safety"
    if any(term in lowered for term in ("history", "fiction", "news", "policy")):
        return "weapons_and_physical_violence", "benign_near_neighbor", "historical_analysis"
    if any(
        term in lowered
        for term in ("mental health", "therapy", "therapist", "wellbeing", "well-being")
    ):
        return "self_harm_and_wellbeing", "benign_near_neighbor", "mental_health_education"
    if any(term in lowered for term in ("privacy", "redact", "personal data")):
        return "privacy_identity_and_secrets", "benign_near_neighbor", "redaction"
    return "regulated_advice", "benign_near_neighbor", "general_information"


def make_record(
    idx: int,
    text: str,
    harmful: bool,
    source: str,
    source_record_id: str,
) -> dict[str, Any]:
    if harmful:
        domain, role, subcluster = domain_guess(text)
        label, severity, actionability, intent, redaction = (
            "unsafe",
            "high",
            "procedural",
            "harmful_likely",
            "restricted",
        )
    else:
        domain, role, subcluster = benign_guess(text)
        label, severity, actionability, intent, redaction = (
            "safe",
            "none",
            "contextual",
            "benign_likely",
            "none",
        )
    return {
        "id": f"bench_comp_{idx:07d}",
        "span_text": text,
        "domain": domain,
        "subcluster_role": role,
        "subcluster_id": subcluster,
        "label": label,
        "source": source,
        "source_record_id": source_record_id,
        "source_record_hash": sha(text),
        "span_derivation": "verbatim_safe_span",
        "raw_prompt_stored": True,
        "license": "upstream_dataset_license_check_required",
        "safety_redaction": redaction,
        "policy_mapping": [f"{source}:auto_mapped_prompt_label"],
        "harm_severity": severity,
        "actionability": actionability,
        "intent_confidence": intent,
        "overlays": [],
        "reviewer_notes": "Auto-mapped benchmark corpus; requires reviewer pass before paper use.",
    }


def add_hf_dataset(
    rows: list[dict[str, Any]],
    seen: set[str],
    source: str,
    dataset: str,
    splits: tuple[str, ...],
    max_per_source: int,
) -> None:
    from datasets import load_dataset

    before = len(rows)
    last_error: Exception | None = None
    for split in splits:
        try:
            loaded = load_dataset(dataset, split=split)
            for idx, row in enumerate(loaded):
                if idx >= max_per_source:
                    break
                text = infer_text(row)
                if not text or text in seen:
                    continue
                seen.add(text)
                rows.append(
                    make_record(len(rows) + 1, text, infer_harmful(row), source, f"{split}:{idx}")
                )
            print(f"{source}: added {len(rows) - before}")
            return
        except Exception as exc:
            last_error = exc
    print(f"SKIP {source}: {last_error}")


def add_harmbench(rows: list[dict[str, Any]], seen: set[str]) -> None:
    url = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
    before = len(rows)
    with urllib.request.urlopen(url) as response:
        decoded = response.read().decode("utf-8").splitlines()
    for idx, row in enumerate(csv.DictReader(decoded)):
        text = clean(row.get("Behavior", ""))
        if not text or text in seen:
            continue
        seen.add(text)
        rows.append(
            make_record(
                len(rows) + 1,
                text,
                True,
                "harmbench",
                row.get("BehaviorID", f"row:{idx}"),
            )
        )
    print(f"harmbench: added {len(rows) - before}")


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    add_hf_dataset(
        rows,
        seen,
        "wildguardmix",
        "bogdanminko/wildguardmix-cleaned",
        ("train",),
        args.max_per_source,
    )
    add_harmbench(rows, seen)

    if args.include_extended:
        for source, dataset, splits in (
            ("xstest", "walledai/XSTest", ("test", "train", "validation")),
            ("beavertails", "PKU-Alignment/BeaverTails", ("30k_train", "train", "test")),
            ("toxicchat", "lmsys/toxic-chat", ("train", "test")),
            ("do_not_answer", "LibrAI/do-not-answer", ("train", "test")),
            ("safetybench", "thu-coai/SafetyBench", ("train", "test", "dev")),
        ):
            add_hf_dataset(rows, seen, source, dataset, splits, args.max_per_source)

    with output.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"wrote {len(rows)} records to {output}")


if __name__ == "__main__":
    main()
