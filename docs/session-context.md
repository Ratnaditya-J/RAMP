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

## Session Severity

Session risk should track harm severity separately from single-turn risk.

A single turn may be ambiguous or moderate, while the session sequence can become high severity through accumulation. This matters when the user gradually assembles a harmful workflow across turns.

Session severity should consider:

- highest severity seen in any turn
- whether severity is increasing across turns
- whether benign-looking turns combine into a harmful workflow
- whether the session adds evasion, optimization, targeting, procurement, or deployment details
- whether the session crosses into high-impact domains such as child safety, self-harm methods, CBRN, infrastructure sabotage, or autonomous tool abuse

Recommended metadata:

```json
{
  "session_max_severity": "high",
  "session_severity_trend": "increasing",
  "severity_accumulation_score": 0.72,
  "cross_turn_composition": [
    "credential_theft",
    "phishing_social_engineering",
    "avoid_detection",
    "automate_at_scale"
  ],
  "highest_severity_turn": "turn-4"
}
```

This keeps prompt-level classification focused on local evidence while allowing session-level classification to measure accumulated severity and harm drift.

## Current Evaluation Status

Session scoring should not be treated as a positive v0 runtime signal yet.

RAMP now has a benchmark-backed session evaluation path using R-Judge:

| Artifact | Path |
|---|---|
| Session corpus | `.artifacts/session_eval/ramp_session_eval_corpus_rjudge_v0_1.jsonl` |
| Flattened turns | `.artifacts/session_eval/ramp_session_eval_corpus_rjudge_v0_1.turns.jsonl` |
| Qwen turn scores | `.artifacts/session_eval/ramp_session_eval_corpus_rjudge_v0_1.qwen_turn_scores.jsonl` |
| Joined session corpus | `.artifacts/session_eval/ramp_session_eval_corpus_rjudge_v0_1.qwen_scored.jsonl` |
| Evaluation report | `.artifacts/session_eval/ramp_session_risk_eval_rjudge_qwen_v0_1.md` |

On R-Judge, `single_turn_max` reaches AUC `0.5471`, while the current
`session_accumulation` score reaches AUC `0.4910`. At the fixed `0.55` threshold,
session accumulation catches only `2` of `286` unsafe sessions and adds no false-negative catches
over the single-turn max baseline.

MHJ was added as an unsafe-only multi-turn jailbreak stress set. It cannot measure FPR or AUC, but
it is useful for unsafe recall and failure mining. At threshold `0.55`, Qwen single-turn max catches
`396` of `496` MHJ sessions (`0.7984` recall), while the current session accumulation formula catches
only `38` of `496` (`0.0766` recall). The top `100` single-turn misses are recorded at
`.artifacts/session_eval/ramp_session_mhj_single_turn_misses_qwen_v0_1.md`.

SafeDialBench was added as an unlabeled multi-turn corpus. It cannot produce metrics until reviewed,
but the top-risk mining artifact is useful for creating a balanced benign/unsafe session review set:

- `.artifacts/session_eval/ramp_session_safedialbench_top_risk_candidates_qwen_v0_1.md`
- `/Users/ratnaditya/Documents/ramp_safedialbench_session_label_review_top_200_v0_1.csv`

Interpretation: the current session formula is useful as a mechanism prototype, but RAMP should not
claim session-level lift from this implementation. The next session experiment should add richer
turn evidence, especially output-risk, role-aware transcript scoring, and tool/action risk. Synthetic
sessions should be kept as smoke tests for severity accumulation, harm drift, and composition, not
as primary evidence.

## Session Classifier v2

The next implementation reframes session risk as compact session-state classification rather than
mathematical aggregation over isolated turn scores.

Runtime shape:

1. Score/store each turn normally.
2. Update a compact safety-specific session state.
3. Select only salient turns and compact evidence.
4. Run cheap deterministic session-state scoring.
5. Optionally call a real classifier on the compact evidence only when session ambiguity warrants it.

The compact state tracks:

- domains and subclusters seen
- highest observed severity
- highest and top-k turn risk
- risk trend
- intent progression
- evasion attempts
- operational-detail requests
- benign cover-story cues
- cross-turn composition
- salient turn snippets and salience reasons

This gives RAMP an optimized classifier input that is much smaller than full transcript input. Current
generated artifacts:

| Artifact | Rows | Mean chars | Max chars |
|---|---:|---:|---:|
| `.artifacts/session_eval/ramp_session_classifier_inputs_rjudge_compact_state_v0_1.jsonl` | 555 | 745.8 | 1,562 |
| `.artifacts/session_eval/ramp_session_classifier_inputs_mhj_compact_state_v0_1.jsonl` | 496 | 950.9 | 1,886 |
| `.artifacts/session_eval/ramp_session_classifier_inputs_safedialbench_compact_state_v0_1.jsonl` | 4,053 | 633.6 | 1,573 |
| `.artifacts/session_eval/ramp_session_classifier_inputs_rjudge_full_transcript_v0_1.jsonl` | 555 | 1,118.9 | 3,753 |
| `.artifacts/session_eval/ramp_session_classifier_inputs_mhj_full_transcript_v0_1.jsonl` | 496 | 1,422.1 | 6,017 |

Deterministic compact scoring is intentionally treated as a baseline, not the final classifier. On
MHJ, it improves over the old accumulator (`0.4234` recall versus `0.0766` at threshold `0.55`), but
it still does not beat single-turn max. The next decisive test is to score the compact session
evidence with Qwen3Guard or another session classifier and compare it with the full-transcript oracle
baseline.
