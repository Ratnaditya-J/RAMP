# External Guard Audit (Option 1) — v0.1 Results

A standalone study applying RAMP's fragility framework to **external, published guards**
rather than RAMP's own signals — the move that turns RAMP from a self-study into a validity
harness you can point at anyone's safety classifier (the same play SIEVE makes with the
published Apollo probes).

## Subjects and testbed

- **Llama Guard 4 12B** (`meta-llama/llama-guard-4-12b`) — a published, dedicated safety
  guard. Binary verdict (no logprobs on OpenRouter), so audited at its fixed operating
  point.
- **Gemini 2.5 Pro as guard** (`google/gemini-2.5-pro`) — a frontier model used as a
  continuous guard (unsafe-probability), so the threshold-transfer analysis applies.
- **Testbed**: RAMP's multi-source public-benchmark slice (wildguardmix, beavertails,
  do_not_answer, harmbench), with an **adaptively-mined hard set** (448 disagreement/error
  rows) and a **blind random set** (500 rows). Labels held to a single provenance — the
  framing-inclusive LLM judge, human-validated at kappa ~0.62 — so the adaptive-vs-blind
  comparison is not confounded by label source. Neither guard had any role in building
  either set, so the audit is non-circular.

## Finding 1 — adaptive vs blind: NO inflation (mildly surprising)

| Guard | set | n | AUROC | recall | FPR | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Llama Guard 4 | adaptive | 365 | 0.859 | 0.799 | 0.081 | 0.837 |
| Llama Guard 4 | blind | 403 | 0.847 | 0.750 | 0.055 | 0.845 |
| Gemini-as-guard | adaptive | 365 | 0.829 | 0.675 | 0.028 | 0.788 |
| Gemini-as-guard | blind | 401 | 0.857 | 0.708 | 0.024 | 0.824 |

Both guards perform about the **same** on adaptively-mined hard cases and on a blind random
sample — F1 is even slightly higher on blind. So, for these published guards with label
provenance held constant, **adaptive eval-set construction does not inflate apparent
performance.**

This is an honest refinement of RAMP's earlier internal-signal result. The large
prompt-only adaptive->blind gap reported earlier (AUROC 0.945 -> 0.850) used human labels on
the adaptive side and judge labels on the blind side; it was therefore partly a
label-provenance effect, not a pure sampling effect. When provenance is held constant, the
sampling effect for these guards is small. RAMP's own framework catching a confound in an
earlier RAMP claim is exactly what a validity harness should do.

## Finding 2 — source shift: the continuous guard's operating point does NOT transfer

Threshold tuned on all-but-one source at target FPR 0.10, applied to the held-out source
(Gemini-as-guard, continuous):

| Held-out source | calib FPR | held-out FPR | calib recall | held-out recall |
| --- | ---: | ---: | ---: | ---: |
| wildguardmix | 0.015 | 0.030 | 0.913 | **0.566** |
| beavertails | 0.039 | **0.102** | 0.632 | 0.947 |

A threshold calibrated on the other sources **collapses recall to 0.57 on held-out
wildguardmix** (it misses ~44% of harms it would have caught in-distribution) and **roughly
triples FPR on held-out beavertails**. So the guard's operating point is source-dependent:
**you cannot tune the threshold on one benchmark and deploy it against another without
recalibration.** (Llama Guard's binary verdict has no tunable threshold, so this axis is not
assessable for it here.)

## Verdicts (Signal Validity Cards)

- **Llama Guard 4 12B: `robust`** on the adaptive-vs-blind axis (source-shift not assessable
  for a binary guard).
- **Gemini-as-guard: `shift_fragile`** — held-out-source recall collapses by up to 0.35.

Cards: `docs/reports/ramp_guard_card_llamaguard4_v0_1.json`,
`docs/reports/ramp_guard_card_gemini_v0_1.json`.

## Honest caveats

- Labels are RAMP framing-inclusive judge labels (human-validated kappa ~0.62), not the
  benchmarks' native labels. A native-label replication would strengthen it.
- The testbed is wildguardmix-heavy; source-shift coverage is limited to the sources with
  both classes present (wildguardmix, beavertails). do_not_answer / harmbench are too small
  or single-class here.
- Llama Guard's binary output (no logprobs via OpenRouter) limits it to operating-point
  analysis; a logprob-capable deployment (local/GPU) would enable AUROC and threshold
  transfer for it too.

## Native-label replication (ToxicChat — fixes the judge-label caveat)

Using the HF token, both guards were re-audited on **ToxicChat** (lmsys/toxic-chat), a
public benchmark with its OWN toxicity labels and two temporal releases (0124, 1123) that
form a built-in distribution shift. 2,000 prompts, ~33% toxic, balanced across releases.
No RAMP labels involved.

| Guard | overall AUROC | overall recall | overall FPR | F1 | threshold transfer across releases |
| --- | ---: | ---: | ---: | ---: | --- |
| Llama Guard 4 12B | 0.748 | **0.545** | 0.050 | 0.662 | n/a (binary) — recall stable 0.57/0.52 across releases |
| Gemini-as-guard | 0.782 | **0.480** | 0.009 | 0.641 | **transfers** (held-out recall 0.56-0.59, FPR stable ~0.01) |

Two native-label findings:

1. **Absolute generalization gap.** On ToxicChat, both published guards catch only roughly
   HALF of the toxic queries (Llama Guard recall 0.55, Gemini-as-guard 0.48) at sensible
   FPR — far below the ~0.9+ such guards report on their own eval distributions. ToxicChat
   is real user-to-Vicuna traffic with subtle toxicity; the guards generalize poorly to it.
   This is on the benchmark's native labels, so it is not a RAMP-labeling artifact.

2. **Threshold transfer depends on shift magnitude.** Across ToxicChat's two temporal
   releases (a MILD shift) the operating point transfers fine (verdict: robust). But across
   distinct benchmarks (the RAMP-corpus wildguardmix<->beavertails shift, a LARGE shift) it
   collapses (recall 0.91->0.57, FPR ~2.6x). So "tune the threshold once, deploy anywhere"
   survives mild temporal drift but breaks under large cross-distribution shift.

## What this establishes

RAMP works as a **validity harness for external published guards**, and it produces a
nuanced, non-trivial, native-label-backed result:

- Published guards are NOT fragile to eval-set construction (adaptive-vs-blind, provenance
  controlled).
- They have a real **absolute generalization gap** (~50% recall on ToxicChat).
- Their operating point transfers under mild shift but is **fragile under large
  cross-benchmark shift**.
- And the harness caught an earlier RAMP label-provenance confound along the way.

That spread of outcomes — robust here, fragile there, with a quantified generalization gap —
is exactly what a validity harness is for, and it is demonstrated on external, published
artifacts with their own labels.
