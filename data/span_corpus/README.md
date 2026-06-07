# RAMP Span Corpus v0

This directory stores labeled span examples for building and evaluating embedding-risk centroids.

The corpus is span-first, not prompt-first. Each record represents one local evidence span and includes:

- `domain`
- `subcluster_role`
- `subcluster_id`
- `label`
- `source`
- `source_record_id`
- `source_record_hash`
- `span_derivation`
- `raw_prompt_stored`
- `license`
- `safety_redaction`
- `policy_mapping`
- `harm_severity`
- `actionability`
- `intent_confidence`
- `reviewer_notes`

The initial corpus is synthetic and non-instructional. Harmful examples describe request classes at a safe abstraction level rather than providing operational harmful instructions.

This corpus is not final training data. It is a v0 scaffold for validating schema, coverage, and taxonomy alignment before collecting or deriving a larger reviewed span dataset.

## Benchmark-Derived Spans

Use `source_manifest_v0.json` to track benchmark and eval sources for future span extraction.

Recommended extraction workflow:

1. Load benchmark prompts in a restricted research workspace.
2. Extract local spans rather than storing whole prompts when the prompt is operationally harmful.
3. Preserve `source_id`, source split, original record ID or hash, and license notes.
4. Map each span to the RAMP taxonomy: `domain`, `subcluster_role`, `subcluster_id`, `label`, `harm_severity`, `actionability`, and `intent_confidence`.
5. Add hard benign near-neighbors from benchmarks such as XSTest and WildGuardMix so centroid training does not learn topic sensitivity as harm.
6. Keep raw dangerous examples out of the public repository unless they are explicitly safe to redistribute and necessary for reproducibility.

Recommended provenance fields:

| Field | Meaning |
| --- | --- |
| `source_record_id` | Upstream benchmark row ID, behavior ID, or local synthetic ID. |
| `source_record_hash` | SHA-256 hash of the upstream record or stored span. |
| `span_derivation` | `verbatim_safe_span`, `redacted_span`, `abstracted_by_reviewer`, or `synthetic_non_instructional`. |
| `raw_prompt_stored` | Whether the raw benchmark prompt/span is checked into this repository. |
| `license` | Dataset license or local synthetic license note. |
| `safety_redaction` | `none`, `redacted`, `abstracted`, or `restricted`. |
