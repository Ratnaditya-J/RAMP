from __future__ import annotations

import json
from pathlib import Path

from ramp.features import ActivationProbeFeature, FeatureInput
from ramp.risk_state import RiskState
from ramp.schemas.feature_result import FeatureStage
from ramp.schemas.provenance import RuntimeProvenance


def make_state() -> RiskState:
    provenance = RuntimeProvenance(
        request_id="req_activation_probe_test",
        session_id="sess_activation_probe_test",
    )
    return RiskState(
        request_id=provenance.request_id,
        session_id=provenance.session_id,
        provenance=provenance,
    )


def test_activation_probe_scores_context_activation_vector(tmp_path: Path) -> None:
    artifact = {
        "artifact_id": "activation_probe_test",
        "layer_id": "19",
        "weights": [2.0, -1.0],
        "bias": 0.0,
        "mean": [0.0, 0.0],
        "scale": [1.0, 1.0],
        "selected_threshold": 0.5,
        "model_version": "linear_activation_probe_test",
        "training_summary": {"test_rows": 2},
    }
    artifact_path = tmp_path / "probe.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    feature = ActivationProbeFeature.from_artifact(artifact_path)

    result = feature.extract(
        FeatureInput(prompt="probe", context={"activation_vector": [1.0, 0.0]}),
        make_state(),
    )

    assert result.stage == FeatureStage.ACTIVATION_PROBE
    assert result.risk_score > 0.5
    assert result.label == "activation_probe_unsafe"
    assert result.metadata["selected_layer"] == "19"
    assert result.metadata["activated"] is True


def test_activation_probe_reports_missing_activation(tmp_path: Path) -> None:
    artifact = {
        "artifact_id": "activation_probe_test",
        "layer_id": "12",
        "weights": [1.0],
        "bias": 0.0,
        "mean": [0.0],
        "scale": [1.0],
        "selected_threshold": 0.5,
    }
    artifact_path = tmp_path / "probe.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    feature = ActivationProbeFeature.from_artifact(artifact_path)

    result = feature.extract(FeatureInput(prompt="probe"), make_state())

    assert result.label == "missing_activation"
    assert result.risk_score == 0.0
