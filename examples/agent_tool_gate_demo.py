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
    state = make_state("req_tool_001", "sess_agent", action_stakes="high")
    feature_input = FeatureInput(
        prompt="Publish the current branch.",
        output="I will push the branch to the remote.",
        tool_name="git_push",
        tool_arguments={"remote": "origin", "branch": "main"},
    )
    pipeline.evaluate(feature_input, state)
    decision = pipeline.add_feature(feature_input, state, FeatureStage.TOOL_ACTION_RISK)
    print_decision(decision)


if __name__ == "__main__":
    main()
