"""Timeliness metric primitives for SPOT-Bench."""

import math
import re
from typing import List, Tuple


def timeliness_score_scalar(
    tau: float,
    t_s: float,
    t_e: float,
    sigma_early: float,
    sigma_late: float,
) -> float:
    """Timeliness of a prediction at time `tau` against a slot window [t_s, t_e].
    Returns 1.0 inside the window and decays as a Gaussian outside it.
    """
    if sigma_early <= 0 or sigma_late <= 0:
        return 1.0 if (t_s <= tau <= t_e) else 0.0

    if tau < t_s:
        return math.exp(-((tau - t_s) ** 2) / (2.0 * (sigma_early**2)))
    elif tau <= t_e:
        return 1.0
    else:
        return math.exp(-((tau - t_e) ** 2) / (2.0 * (sigma_late**2)))


def greedy_match_timeliness_topK(
    pred_times: List[float],
    slot_t_s: List[float],
    slot_t_e: List[float],
    sigma_early: float,
    sigma_late: float,
    timeliness_threshold: float,
    occupancy_k: int = 5,
    semantics_ok: List[List[bool]] | None = None,
    semantic_ok_fn=None,
) -> Tuple[List[float], List[int], List[bool]]:
    """Match predictions to slots, earliest prediction first.

    A prediction may match a slot only if it clears both `timeliness_threshold`
    and the semantic check. Each slot absorbs at most `occupancy_k` predictions
    before it closes; this is the budget that keeps a model from spamming a slot
    with responses and being rewarded for it. A slot keeps its single best
    Timeliness among the predictions it absorbed.

    Returns `(slot_best_T, slot_best_pred, pred_matched)`, where `slot_best_pred`
    holds -1 for unmatched slots (false negatives) and `pred_matched` is False
    for unmatched predictions (false positives).
    """
    if occupancy_k < 1:
        raise ValueError(f"occupancy_k must be >= 1, got {occupancy_k}")
    if semantics_ok is None and semantic_ok_fn is None:
        raise ValueError("Provide either `semantics_ok` or `semantic_ok_fn`.")

    num_preds = len(pred_times)
    num_slots = len(slot_t_s)

    slot_best_T = [0.0 for _ in range(num_slots)]
    slot_best_pred = [-1 for _ in range(num_slots)]
    pred_matched = [False for _ in range(num_preds)]
    slot_count = [0 for _ in range(num_slots)]
    slot_closed = [False for _ in range(num_slots)]

    pred_order = sorted(range(num_preds), key=lambda i: pred_times[i])
    slot_order = sorted(range(num_slots), key=lambda j: (slot_t_s[j], j))

    for i in pred_order:
        tau = float(pred_times[i])
        for j in slot_order:
            if slot_closed[j]:
                continue

            t_s = float(slot_t_s[j])
            t_e = float(slot_t_e[j])
            T_ij = timeliness_score_scalar(tau, t_s, t_e, sigma_early, sigma_late)
            if T_ij < timeliness_threshold:
                continue

            if semantic_ok_fn is not None:
                sem_ok = bool(semantic_ok_fn(j, i))
            else:
                sem_ok = bool(semantics_ok[j][i])
            if not sem_ok:
                continue

            slot_count[j] += 1
            if T_ij > slot_best_T[j]:
                slot_best_T[j] = T_ij
                slot_best_pred[j] = i
            pred_matched[i] = True

            if slot_count[j] >= occupancy_k:
                slot_closed[j] = True

    return slot_best_T, slot_best_pred, pred_matched


def compute_prf_weighted(TPw: float, FP: int, FN: int, TP: float | None = None):
    """Precision, recall and F1 where true positives are Timeliness-weighted."""
    TP_full = TPw if TP is None else TP

    precision = 0.0 if TP_full + FP == 0 else TPw / (TP_full + FP)
    recall = 0.0 if TP_full + FN == 0 else TPw / (TP_full + FN)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def norm_text(s: str) -> str:
    """Lowercase and strip markdown/punctuation noise from a model response."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace("**", "")
    s = re.sub(r"[\"“”]", "", s)
    s = s.strip(".,!?[];:*")
    s = re.sub(r"\s+", " ", s)
    return s


def ensure_list(x):
    """Coerce a scalar / None / list annotation field into a list."""
    if isinstance(x, list):
        return x
    if x is None:
        return []
    return [x]
