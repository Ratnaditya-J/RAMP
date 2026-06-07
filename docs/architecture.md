# Architecture

RAMP is organized around partial evidence. Each feature extractor emits a `FeatureResult`; the fusion engine combines whatever is currently available; the scheduler decides whether to exit early or request the next feature.

## Runtime Loop

1. Create a `RiskState` with request/session/provenance metadata.
2. Run cheap features first.
3. Fuse the available feature results into a `RiskDecision`.
4. Stop early if the decision is decisive.
5. Otherwise schedule a deeper feature.
6. Persist the decision and update session state.

## Extension Points

- `ramp.features.FeatureExtractor`: implement this for model-backed classifiers, embedding search, activation probes, output classifiers, or tool gates.
- `ramp.fusion.WeightedRiskFusion`: replace this with calibrated rules, logistic regression, a tree model, Bayesian updates, or an ensemble.
- `ramp.scheduler.AnytimeScheduler`: replace this when scheduling needs modelled information gain, SLA-specific latency, or deployment-specific policy.
- `ramp.schemas.RuntimeProvenance`: extend this when new runtime state must be bound to decisions.

## Package Boundaries

The scaffold intentionally keeps feature extraction, fusion, scheduling, provenance, and session state separate. This lets a deployment swap in one production-grade component without rewriting the rest.

## Session Boundaries

Session features should distinguish explicit session IDs from inferred context windows. If a caller provides a conversation, task, or agent-run ID, RAMP can compute session-risk features over that known boundary. If no session ID exists, RAMP should use a bounded sliding window only when the caller can provide trustworthy local context, and should lower confidence accordingly.

Missing or uncertain session context should be represented explicitly rather than treated as low risk.
