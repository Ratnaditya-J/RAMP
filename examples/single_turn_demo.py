from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _common import make_state, print_decision

from ramp.features import FeatureInput
from ramp.pipeline import default_pipeline


def main() -> None:
    pipeline = default_pipeline()
    state = make_state("req_low_001", "sess_demo")
    decision = pipeline.evaluate(
        FeatureInput(prompt="Can you explain safe password manager best practices?"),
        state,
    )
    print_decision(decision)


if __name__ == "__main__":
    main()
