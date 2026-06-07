from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _common import make_state, print_decision

from ramp.features import FeatureInput
from ramp.pipeline import default_pipeline
from ramp.schemas.feature_result import FeatureStage


def main() -> None:
    pipeline = default_pipeline()
    state = make_state("req_async_audit_001", "sess_demo")
    feature_input = FeatureInput(prompt="Can you discuss defensive account security?")
    hot_path_decision = pipeline.evaluate(feature_input, state)
    print("before output exists:")
    print_decision(hot_path_decision)

    audit_input = FeatureInput(
        prompt=feature_input.prompt,
        output="A safe alternative is to use MFA, logging, alerts, and recovery procedures.",
    )
    audit_decision = pipeline.add_feature(audit_input, state, FeatureStage.OUTPUT_RISK)
    print("after output exists:")
    print_decision(audit_decision)


if __name__ == "__main__":
    main()
