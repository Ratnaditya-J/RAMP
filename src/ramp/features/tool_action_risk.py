from __future__ import annotations

from ramp.features.base import FeatureExtractor, FeatureInput
from ramp.risk_state import RiskState
from ramp.schemas.feature_result import FeatureResult, FeatureStage


class SideEffectToolActionRiskFeature(FeatureExtractor):
    stage = FeatureStage.TOOL_ACTION_RISK
    side_effecting_tools = {
        "write_file",
        "delete_file",
        "send_message",
        "make_payment",
        "deploy",
        "git_push",
        "external_api_call",
        "update_permissions",
        "mutate_memory",
    }

    def extract(self, feature_input: FeatureInput, state: RiskState) -> FeatureResult:
        tool_name = feature_input.tool_name or ""
        side_effecting = tool_name in self.side_effecting_tools
        risk = 0.82 if side_effecting else 0.18
        decision = "escalate" if side_effecting else "allow"
        return FeatureResult(
            stage=self.stage,
            risk_score=risk,
            confidence=0.90,
            label=decision,
            cost_tier=0,
            version="side_effect_tool_gate_v0",
            metadata={
                "tool": tool_name,
                "side_effecting": side_effecting,
                "required_approval": side_effecting,
            },
        )

