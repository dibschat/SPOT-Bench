"""SPOT-Bench streaming scorer.

Scores a trace against the ground-truth response slots with top-K greedy
timeliness matching. Timeliness is resolved first and semantics lazily, so the
LLM judge only ever sees pairs that are already temporally plausible (cheaper to evaluate).
"""

from collections import defaultdict
from typing import Any, Dict

from tqdm import tqdm

from .config import NO_EVENT_TEXTS, TASK_TIME_CONFIG
from .judge import (
    mentions_end,
    mentions_start,
    semantic_ok_detection,
    semantic_ok_open_ended,
)
from .metrics import (
    compute_prf_weighted,
    ensure_list,
    greedy_match_timeliness_topK,
    norm_text,
)

DETECTION_TASKS = {"PNR", "ABD"}


def _empty_stats():
    return {
        "slots": 0,
        "preds": 0,
        "TPw": 0.0,
        "FP": 0,
        "FN": 0,
        "slot_score_sum": 0.0,
        "latency_sum": 0.0,
        "latency_count": 0,
        "TPw_k1": 0.0,
        "FP_k1": 0,
        "FN_k1": 0,
    }


def _accumulate(dst, src):
    for key in ("slots", "preds", "TPw", "FP", "FN", "slot_score_sum", "latency_sum",
                "latency_count", "TPw_k1", "FP_k1", "FN_k1"):
        dst[key] += src[key]


def _finalize(stats):
    if stats["slots"] == 0:
        return {
            "slots": 0,
            "preds": stats["preds"],
            "TPw": 0.0,
            "FP": stats["FP"],
            "FN": stats["FN"],
            "timeliness_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "timeliness_f1": 0.0,
            "timeliness_f1_at_1": 0.0,
            "avg_latency_s": None,
        }

    timeliness_score = stats["slot_score_sum"] / stats["slots"]
    matched_slots = stats["slots"] - stats["FN"]
    precision, recall, f1 = compute_prf_weighted(
        stats["TPw"], stats["FP"], stats["FN"], TP=matched_slots
    )
    avg_latency = (
        stats["latency_sum"] / stats["latency_count"] if stats["latency_count"] else None
    )
    _, _, f1_at_1 = compute_prf_weighted(
        stats["TPw_k1"], stats["FP_k1"], stats["FN_k1"],
        TP=stats["slots"] - stats["FN_k1"],
    )

    return {
        "slots": stats["slots"],
        "preds": stats["preds"],
        "TPw": stats["TPw"],
        "FP": stats["FP"],
        "FN": stats["FN"],
        "timeliness_score": timeliness_score,
        "precision": precision,
        "recall": recall,
        "timeliness_f1": f1,
        "timeliness_f1_at_1": f1_at_1,
        "avg_latency_s": avg_latency,
    }


def _gold_responses_for_detection(task: str, turn: dict, gt_times: list, gt_resps: list):
    if not gt_resps:
        q = (turn.get("question", "") or "").lower()

        if task == "PNR":
            gt_resps = ["now"] * len(gt_times)
        elif task == "ABD":
            declared = [
                part.strip().lower()
                for part in str(turn.get("type", "") or "").split("|")
                if part.strip()
            ]
            if declared:
                gt_resps = declared
            elif len(gt_times) == 2 and mentions_start(q) and mentions_end(q):
                gt_resps = ["start", "end"]
            elif len(gt_times) == 1:
                gt_resps = (
                    ["end"] if (mentions_end(q) and not mentions_start(q)) else ["start"]
                )
            else:
                gt_resps = ["start"] * len(gt_times)

    if len(gt_resps) == 1 and len(gt_times) > 1:
        gt_resps = gt_resps * len(gt_times)
    elif len(gt_resps) != len(gt_times):
        if len(gt_resps) > len(gt_times):
            gt_resps = gt_resps[: len(gt_times)]
        else:
            gt_resps = gt_resps + [gt_resps[-1]] * (len(gt_times) - len(gt_resps))

    return gt_resps


def score(args, results_by_task: Dict[str, Dict[str, Any]], task_to_family: Dict[str, str]):
    model_name = args.model
    requested_tasks = set(args.tasks)
    occupancy_k = int(args.occupancy_k)
    dump_matches = bool(getattr(args, "dump_matches", False))

    task_stats = {t: _empty_stats() for t in requested_tasks}
    family_stats = defaultdict(_empty_stats)
    overall_stats = _empty_stats()
    match_details: Dict[str, Dict[str, Any]] = {t: {} for t in requested_tasks}

    family_to_tasks = defaultdict(set)
    for task, family in task_to_family.items():
        family_to_tasks[family].add(task)

    for task, videos in results_by_task.items():
        if task not in requested_tasks or task not in TASK_TIME_CONFIG:
            continue

        cfg = TASK_TIME_CONFIG[task]
        offset = cfg["offset"]
        sigma_early = cfg["sigma_early"]
        sigma_late = cfg["sigma_late"]
        t_thresh = cfg["timeliness_threshold"]
        sim_thresh = cfg["sim_threshold"]

        if task not in DETECTION_TASKS and sim_thresh is None:
            raise ValueError(f"sim_threshold not set for open-ended task {task}")

        stats = task_stats[task]

        for vid, entry in tqdm(
            videos.items(), desc=f"[score] {task}", total=len(videos), position=0
        ):
            vid_slots = []
            vid_preds = []

            for turn in entry.get("turns", []):
                gt_times = ensure_list(turn.get("response_time"))
                gt_resps = ensure_list(turn.get("response"))

                if not gt_times:
                    continue

                if task in DETECTION_TASKS:
                    gt_resps = _gold_responses_for_detection(
                        task, turn, gt_times, gt_resps
                    )
                else:
                    assert len(gt_times) == len(
                        gt_resps
                    ), "Time and response count mismatch in annotations."

                question = str(turn.get("question", ""))
                for t_center, gold in zip(gt_times, gt_resps):
                    t_center = float(t_center)
                    vid_slots.append(
                        {
                            "t_s": t_center - offset,
                            "t_e": t_center + offset,
                            "gold_text": str(gold),
                            "question": question,
                        }
                    )

                model_out = turn.get(model_name)
                if not model_out:
                    continue

                pred_times = ensure_list(model_out.get("response_time", []))
                pred_texts = ensure_list(model_out.get("response", []))
                pred_latencies = ensure_list(model_out.get("latency", []))
                if not pred_times or not pred_texts:
                    continue

                n = min(len(pred_times), len(pred_texts))
                for i in range(n):
                    if norm_text(pred_texts[i]) in NO_EVENT_TEXTS:
                        continue
                    latency = pred_latencies[i] if i < len(pred_latencies) else None
                    vid_preds.append(
                        {
                            "time": float(pred_times[i]),
                            "text": str(pred_texts[i]),
                            "latency": float(latency) if latency is not None else None,
                        }
                    )

            if not vid_slots:
                continue

            slot_t_s = [s["t_s"] for s in vid_slots]
            slot_t_e = [s["t_e"] for s in vid_slots]
            slot_golds = [s["gold_text"] for s in vid_slots]
            slot_questions = [s["question"] for s in vid_slots]
            pred_times = [p["time"] for p in vid_preds]
            pred_texts = [p["text"] for p in vid_preds]

            sem_cache: dict[tuple[int, int], bool] = {}

            def _semantic_ok(slot_idx: int, pred_idx: int) -> bool:
                key = (slot_idx, pred_idx)
                if key in sem_cache:
                    return sem_cache[key]

                if task in DETECTION_TASKS:
                    ok = semantic_ok_detection(
                        task, pred_texts[pred_idx], slot_golds[slot_idx]
                    )
                else:
                    ok = semantic_ok_open_ended(
                        task=task,
                        question=slot_questions[slot_idx] if task == "SQA" else "",
                        gold=slot_golds[slot_idx],
                        pred=pred_texts[pred_idx],
                        score_threshold=sim_thresh,
                    )

                sem_cache[key] = ok
                return ok

            slot_best_T, slot_best_pred, pred_matched = greedy_match_timeliness_topK(
                pred_times=pred_times,
                slot_t_s=slot_t_s,
                slot_t_e=slot_t_e,
                sigma_early=sigma_early,
                sigma_late=sigma_late,
                timeliness_threshold=t_thresh,
                occupancy_k=occupancy_k,
                semantic_ok_fn=_semantic_ok,
            )

            if occupancy_k == 1:
                slot_best_T_k1, slot_best_pred_k1, pred_matched_k1 = (
                    slot_best_T, slot_best_pred, pred_matched
                )
            else:
                slot_best_T_k1, slot_best_pred_k1, pred_matched_k1 = (
                    greedy_match_timeliness_topK(
                        pred_times=pred_times,
                        slot_t_s=slot_t_s,
                        slot_t_e=slot_t_e,
                        sigma_early=sigma_early,
                        sigma_late=sigma_late,
                        timeliness_threshold=t_thresh,
                        occupancy_k=1,
                        semantic_ok_fn=_semantic_ok,
                    )
                )

            for j, p_idx in enumerate(slot_best_pred_k1):
                if p_idx == -1:
                    stats["FN_k1"] += 1
                else:
                    stats["TPw_k1"] += float(slot_best_T_k1[j])
            stats["FP_k1"] += sum(1 for m in pred_matched_k1 if not m)

            if dump_matches:
                match_details[task][vid] = {
                    "matched": [
                        {"slot_idx": j, "pred_idx": int(p), "T": float(slot_best_T[j])}
                        for j, p in enumerate(slot_best_pred)
                        if p != -1
                    ],
                    "fn_slot_idxs": [
                        j for j, p in enumerate(slot_best_pred) if p == -1
                    ],
                    "fp_pred_idxs": [
                        i for i, matched in enumerate(pred_matched) if not matched
                    ],
                }

            stats["preds"] += len(vid_preds)
            stats["slots"] += len(vid_slots)

            for j in range(len(vid_slots)):
                Tj = float(slot_best_T[j])
                stats["slot_score_sum"] += Tj

                matched_pred_idx = slot_best_pred[j]
                if matched_pred_idx == -1:
                    stats["FN"] += 1
                    continue

                stats["TPw"] += Tj
                # response latency of only the matched responses
                latency = vid_preds[matched_pred_idx]["latency"]
                if latency is not None:
                    stats["latency_sum"] += latency
                    stats["latency_count"] += 1

            stats["FP"] += sum(1 for matched in pred_matched if not matched)

        family = task_to_family.get(task)
        if family is not None:
            _accumulate(family_stats[family], stats)
        _accumulate(overall_stats, stats)

    per_task = {
        task: _finalize(task_stats[task])
        for task in sorted(requested_tasks)
        if task_stats[task]["slots"] > 0
    }

    # Only meaningful when every task in the family was scored in this run.
    per_family = {}
    for family, fstats in family_stats.items():
        required = family_to_tasks[family]
        if required.issubset(per_task.keys()) and fstats["slots"] > 0:
            per_family[family] = _finalize(fstats)
            per_family[family]["tasks"] = sorted(required)

    overall = {}
    if set(task_to_family).issubset(per_task.keys()) and overall_stats["slots"] > 0:
        overall = _finalize(overall_stats)

    _print_report(per_task, per_family, overall, occupancy_k)

    output = {"per_task": per_task, "per_family": per_family, "overall": overall}
    if dump_matches:
        output["matches"] = match_details
    return output


def _fmt_latency(m):
    return "n/a" if m["avg_latency_s"] is None else f"{m['avg_latency_s']:.3f}s"


def _metrics_line(m, k):
    return (
        f"Slots={m['slots']}, "
        f"T-Score@{k}={m['timeliness_score']*100:.1f}%, "
        f"T-F1@{k}={m['timeliness_f1']*100:.1f}%, "
        f"FP={m['FP']}, FN={m['FN']}, "
        f"T-F1@1={m['timeliness_f1_at_1']*100:.1f}%, "
        f"Latency={_fmt_latency(m)}"
    )


def _print_report(per_task, per_family, overall, occupancy_k):
    for task in sorted(per_task):
        print(f"  Task {task}: {_metrics_line(per_task[task], occupancy_k)}")

    for family in sorted(per_family):
        m = per_family[family]
        print(
            f"  Family {family}: {_metrics_line(m, occupancy_k)} "
            f"(Tasks={','.join(m['tasks'])})"
        )

    if overall:
        print(f"  Overall: {_metrics_line(overall, occupancy_k)}")
