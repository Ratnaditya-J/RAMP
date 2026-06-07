# Latency

The scheduler treats feature cost as part of the decision. The current implementation is rule-based and intentionally simple; it establishes the interface for later information-gain scheduling.

Suggested cost tiers:

- Tier 0: policy heuristics, session prior, tool side-effect checks, cached decisions
- Tier 1: prompt classifier, lightweight embedding risk, simple session drift
- Tier 2: activation probes, richer span analysis, stronger local classifiers
- Tier 3: output classifiers, LLM judges, full session analysis, human review

Production schedulers should consider current risk, confidence, latency budget, action stakes, feature cost, expected information gain, and deployment mode.

