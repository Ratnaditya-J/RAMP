# Deployment Topologies

Streaming and non-streaming are deployment concepts, not RAMP implementation modes.

The core evaluator does not carry a streaming flag and does not branch on a deployment mode. It accepts whatever evidence is currently available, fuses that evidence, and either recommends an action or asks for the next useful available feature.

## Private Output Review

Some callers can generate model output privately before release. In that topology, the caller can provide both prompt evidence and generated output to RAMP before deciding whether to release, rewrite, block, or escalate.

RAMP does not need to know this is "non-streaming." It only sees that an output is available, so `output_risk_feature` can be scheduled or added.

## Asynchronous Output Audit

Some callers begin user-visible generation before full output classification is possible. In that topology, the caller first evaluates prompt/session/embedding evidence, then later submits generated output as additional evidence.

RAMP does not need to know this is "streaming." It only sees two moments in time: one evaluation before output exists, and another update after output exists.

## Agent Tool Gate

Tool calls are also availability-driven. When a proposed tool/action exists, the caller submits it to RAMP before execution. The tool/action feature can then contribute a side-effect risk signal independent of the text-generation topology.

## Core Boundary

The core implementation owns:

- feature result schemas
- risk state
- feature availability
- scheduling among available features
- fusion of partial evidence
- decision/provenance records

Deployment adapters own:

- whether output is private or user-visible
- buffering policy
- stream interruption policy
- when asynchronous audit results are applied
- product-specific handling of `allow`, `block`, `rewrite`, `escalate`, and `caution`
