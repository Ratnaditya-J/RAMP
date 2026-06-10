# RAMP Session Corpus

Session-risk evaluation is multi-turn-first. Each JSONL row is one session or reconstructed
conversation window, not one prompt.

Required fields:

| Field | Meaning |
| --- | --- |
| `session_id` | Stable session/window identifier. |
| `source` | Benchmark or control source, such as `mhj`, `safedialbench`, `rjudge`, or `synthetic_control_v0`. |
| `source_record_id` | Upstream record id or generated control id. |
| `session_label` | `safe`, `unsafe`, or `ambiguous_or_context_needed`. |
| `session_risk_type` | Main session pattern: `harm_drift`, `persistence`, `cross_turn_composition`, `severity_accumulation`, `de_escalation`, or `benign_multi_turn`. |
| `session_boundary` | `explicit_benchmark` or `inferred_sliding_window`. |
| `session_boundary_confidence` | Float from `0..1`. |
| `turns` | Ordered list of turn dictionaries. |

Each turn should contain:

| Field | Meaning |
| --- | --- |
| `turn_id` | Turn id stable within the session. |
| `turn_index` | Zero-based turn index. |
| `role` | Usually `user`; assistant/environment records may be preserved when present. |
| `text` | Turn text. |
| `turn_label` | Optional `safe`, `unsafe`, or `ambiguous_or_context_needed`. |
| `risk_score` | Optional scalar prior. If absent, session evaluation can infer a lexical score. |
| `harm_domain` | Optional mapped RAMP domain. |
| `subcluster_id` | Optional mapped RAMP subcluster. |
| `harm_severity` | Optional `none`, `low`, `medium`, `high`, or `critical`. |

The v0 principle is: use existing multi-turn benchmarks first, then add synthetic controls only for
mechanisms that are hard to isolate in public data.
