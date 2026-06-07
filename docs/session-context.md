# Session Context

Session features are only as reliable as the boundary that defines the session.

RAMP should treat session context as an input with provenance and confidence, not as a guaranteed fact. The classifier should distinguish between explicit session state, inferred session windows, and missing session context.

## Explicit Session ID

The cleanest case is an explicit `session_id` from the host application.

Examples:

- chat conversation ID
- agent run ID
- authenticated support thread ID
- task/workflow ID
- evaluation trajectory ID

In this case, RAMP can compute session-level features over the known turn sequence:

- accumulating severity across turns
- repeated attempts around the same harm category
- drift from educational framing to actionable framing
- reformulation after refusal
- cross-turn assembly of risky instructions
- tool/action escalation over the trajectory

The session feature should record:

```text
session_boundary = explicit
session_confidence = high
window_size = all available turns or configured recent-N turns
```

## No Session ID

When no explicit session ID exists, RAMP should not invent a durable user-level session by default.

Instead, use a bounded context window when the caller can provide one:

- last `N` turns in the current visible conversation
- last `N` requests in the current agent run
- events within a short time window
- semantically adjacent prompts in the same local interaction
- offline reconstructed trajectories for evaluation datasets

The feature should mark this as inferred context:

```text
session_boundary = inferred_sliding_window
session_confidence = medium or low
window_size = N
window_policy = time_decay | last_n_turns | semantic_continuity
```

If no trustworthy context is available, the session feature should be missing rather than silently treated as low risk.

```text
session_boundary = unavailable
session_confidence = none
feature_status = missing
```

## Sliding Window Policies

Useful fallback policies:

| Policy | Use When | Risk |
|---|---|---|
| `last_n_turns` | Caller has a current chat transcript but no durable session ID. | May miss long-horizon buildup. |
| `time_window` | Events are timestamped and scoped to one local interaction. | Can merge unrelated requests if scope is too broad. |
| `semantic_continuity` | Offline evaluation needs to group related prompts. | Can accidentally join different intents. |
| `hybrid` | Production-like evaluation with timestamps and semantic similarity. | More complex and needs calibration. |

## Conservative Boundary Rule

When uncertain, RAMP should prefer under-linking over over-linking.

Merging unrelated users or unrelated tasks into one session can create false positives and invalid research results. A session feature with uncertain boundaries should lower confidence and expose the boundary source in metadata.

## Recommended Feature Metadata

Session-risk outputs should include:

```json
{
  "session_boundary": "explicit",
  "session_confidence": 0.95,
  "window_policy": "all_turns",
  "window_size": 8,
  "oldest_turn_age_seconds": 420,
  "turn_ids": ["turn_001", "turn_002"],
  "top_accumulating_category": "cyber_misuse",
  "trend": "increasing"
}
```

For inferred windows:

```json
{
  "session_boundary": "inferred_sliding_window",
  "session_confidence": 0.52,
  "window_policy": "last_n_turns",
  "window_size": 4,
  "boundary_reason": "no explicit session_id provided"
}
```

This lets RAMP compare explicit session classifiers against fallback sliding-window classifiers without hiding boundary uncertainty.

