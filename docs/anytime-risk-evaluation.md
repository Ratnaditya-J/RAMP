# Anytime Risk Evaluation

RAMP maintains a live `RiskState`. The state can be fused after every feature result and does not require all stages to be present.

Early exits are supported in both directions:

- low risk plus high confidence can produce `allow_fast_path`
- high risk plus high confidence can produce `block` or `escalate`
- disagreement, medium risk, or low confidence produces `continue_evaluation`

Missing features remain explicit through `RiskDecision.features_missing`. A missing activation probe or output classifier is not treated as zero risk.

