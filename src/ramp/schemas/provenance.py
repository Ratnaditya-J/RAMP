from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class RuntimeProvenance:
    """Runtime identity attached to a safety decision."""

    request_id: str
    session_id: str
    base_model: str = "unknown"
    model_version: str = "unknown"
    system_prompt_hash: str = "sha256:unknown"
    tool_registry_hash: str = "sha256:unknown"
    memory_snapshot_hash: str = "sha256:unknown"
    policy_version: str = "policy_v0"
    prompt_classifier_version: str = "unknown"
    embedding_feature_version: str = "unknown"
    harm_cluster_version: str = "unknown"
    activation_probe_version: str = "unknown"
    output_classifier_version: str = "unknown"
    session_classifier_version: str = "unknown"
    risk_fusion_version: str = "risk_fusion_v0"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

