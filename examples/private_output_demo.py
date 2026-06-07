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
    state = make_state("req_private_output_001", "sess_demo")
    feature_input = FeatureInput(
        prompt="Explain defensive incident response at a high level.",
        output=(
            "A safe alternative is to focus on preparation, detection, "
            "containment, and recovery."
        ),
    )
    decision = pipeline.evaluate(feature_input, state)
    if not state.has_feature(FeatureStage.OUTPUT_RISK):
        decision = pipeline.add_feature(feature_input, state, FeatureStage.OUTPUT_RISK)
    print_decision(decision)


if __name__ == "__main__":
    main()
