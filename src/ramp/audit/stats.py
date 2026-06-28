"""Statistical primitives for the robustness audit: AUROC, F1, paired bootstrap.

Ported verbatim (numpy + stdlib only) from
``scripts/evaluate_signal_survival_ladder.py`` so the packaged auditor computes
the *same* numbers the study produced. The bootstrap is seeded through an
explicit ``numpy.random.Generator`` so a card is reproducible from
(bundle, seed) alone.

Design notes carried over from the source:

- ``_fast_auc`` is an average-rank Mann-Whitney AUROC (handles ties), matching
  ``evaluate_reviewed_cumulative_signals.auc``. Returns ``None`` for a
  single-class sample (AUROC undefined).
- ``_f1_at`` thresholds ``score >= threshold`` and matches ``metrics()``.
- ``paired_bootstrap`` resamples per-row (with replacement), drops single-class
  resamples, and reports the 95% percentile CI of the AUROC/F1 delta vs a
  baseline; ``significant`` iff the CI excludes zero.

No model, no GPU, no network. ``numpy`` is the only third-party import.
"""
from __future__ import annotations

import hashlib
from typing import Any


def _seed_int(seed: str) -> int:
    """Deterministic 64-bit seed from a string (sha256 of the string)."""
    return int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:8], "big")


def _fast_auc(np: Any, labels_a: Any, scores_a: Any) -> float | None:
    positives = int(labels_a.sum())
    total = int(labels_a.size)
    negatives = total - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores_a, kind="mergesort")
    s_sorted = scores_a[order]
    ranks_sorted = np.empty(total, dtype=np.float64)
    index = 0
    while index < total:
        end = index
        value = s_sorted[index]
        while end + 1 < total and s_sorted[end + 1] == value:
            end += 1
        ranks_sorted[index : end + 1] = (index + end) / 2.0 + 1.0
        index = end + 1
    ranks = np.empty(total, dtype=np.float64)
    ranks[order] = ranks_sorted
    positive_rank_sum = float(ranks[labels_a == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _f1_at(np: Any, labels_a: Any, scores_a: Any, threshold: Any) -> float:
    predicted = scores_a >= threshold
    tp = int(np.sum(predicted & (labels_a == 1)))
    fp = int(np.sum(predicted & (labels_a == 0)))
    fn = int(np.sum(~predicted & (labels_a == 1)))
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else 0.0


def _delta_summary(np: Any, deltas: list[float], *, point: float | None) -> dict[str, Any]:
    if not deltas:
        return {"delta": point, "ci95": [None, None], "significant": False, "resamples": 0}
    array = np.asarray(deltas, dtype=np.float64)
    low = float(np.percentile(array, 2.5))
    high = float(np.percentile(array, 97.5))
    return {
        "delta": point,
        "ci95": [low, high],
        "significant": bool(low > 0 or high < 0),
        "resamples": len(deltas),
    }


def paired_bootstrap(
    np: Any,
    labels: list[int],
    base_scores: list[float],
    base_threshold: Any,
    candidates: list[tuple[str, list[float], Any]],
    *,
    num_resamples: int,
    seed: str,
) -> dict[str, dict[str, Any]]:
    """Per-row paired bootstrap of AUROC and F1 deltas vs the prompt-only baseline.

    Thresholds may be scalars (blind rung) or per-row arrays (pooled shifted rung, where
    each row was scored by its own leave-one-source-out calibration).
    """
    labels_a = np.asarray(labels, dtype=np.int64)
    base_s = np.asarray(base_scores, dtype=np.float64)
    base_t = np.asarray(base_threshold, dtype=np.float64)
    cand = [
        (name, np.asarray(scores, dtype=np.float64), np.asarray(threshold, dtype=np.float64))
        for name, scores, threshold in candidates
    ]
    total = int(labels_a.size)
    base_auc_point = _fast_auc(np, labels_a, base_s)
    base_f1_point = _f1_at(np, labels_a, base_s, base_t)

    rng = np.random.default_rng(_seed_int(seed))
    draws = rng.integers(0, total, size=(num_resamples, total))
    accumulator: dict[str, dict[str, list[float]]] = {
        name: {"auc": [], "f1": []} for name, _, _ in cand
    }
    used = 0
    for resample in range(num_resamples):
        idx = draws[resample]
        labels_b = labels_a[idx]
        positives = int(labels_b.sum())
        if positives == 0 or positives == total:
            continue
        used += 1
        base_sb = base_s[idx]
        base_tb = base_t[idx] if base_t.ndim else base_t
        base_auc = _fast_auc(np, labels_b, base_sb)
        base_f1 = _f1_at(np, labels_b, base_sb, base_tb)
        for name, scores, threshold in cand:
            cand_sb = scores[idx]
            cand_tb = threshold[idx] if threshold.ndim else threshold
            cand_auc = _fast_auc(np, labels_b, cand_sb)
            if base_auc is not None and cand_auc is not None:
                accumulator[name]["auc"].append(cand_auc - base_auc)
            accumulator[name]["f1"].append(_f1_at(np, labels_b, cand_sb, cand_tb) - base_f1)

    output: dict[str, dict[str, Any]] = {}
    for name, scores, threshold in cand:
        cand_auc_point = _fast_auc(np, labels_a, scores)
        auc_point = (
            None
            if base_auc_point is None or cand_auc_point is None
            else cand_auc_point - base_auc_point
        )
        f1_point = _f1_at(np, labels_a, scores, threshold) - base_f1_point
        output[name] = {
            "auc": _delta_summary(np, accumulator[name]["auc"], point=auc_point),
            "f1": _delta_summary(np, accumulator[name]["f1"], point=f1_point),
            "resamples_used": used,
            "num_rows": total,
        }
    return output
