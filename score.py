"""Score SPOT-Bench streaming results.

Open-ended tasks (SQA, SPG, SI, UI) call an LLM judge and therefore need
OPENAI_API_KEY in the environment.
"""

import argparse
import copy
import glob
import json
import os
import sys

from evals import score as score_results
from evals.config import DEFAULT_OCCUPANCY_K
from tasks import TASK_TO_FAMILY, annotation_path, normalize_tasks, task_key
from model_wrappers import MODELS


def args_parser():
    parser = argparse.ArgumentParser(description="Score SPOT-Bench streaming results")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        required=True,
        metavar="{ABD,PNR,SQA,SPG,SI,UI}",
        help="Tasks to score (one or more).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=MODELS,
        help="Registered Model name.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Directory containing the result jsons.",
    )
    parser.add_argument(
        "--annotation_root",
        type=str,
        default="data",
        help="Directory containing SPOT-Bench annotations.",
    )
    parser.add_argument(
        "--annotation_path",
        type=str,
        default=None,
        help="Explicit annotation JSON path. Overrides --annotation_root. Only valid with a single --tasks entry.",
    )
    parser.add_argument(
        "--occupancy_k",
        type=int,
        default=DEFAULT_OCCUPANCY_K,
        help=(
            f"Per-slot occupancy budget K for Timeliness@K matching "
            f"(default: {DEFAULT_OCCUPANCY_K}). K=1 allows one response per slot."
        ),
    )
    parser.add_argument(
        "--dump_matches",
        action="store_true",
        help="Include per-video slot/prediction match details in the score file.",
    )

    args = parser.parse_args()
    args.tasks = normalize_tasks(parser, args.tasks)
    if args.annotation_path is not None and len(args.tasks) != 1:
        parser.error("--annotation_path requires exactly one --tasks entry")
    if args.occupancy_k < 1:
        parser.error("--occupancy_k must be >= 1")
    return args


def _resolve_annotation(args, task: str) -> str:
    if args.annotation_path is not None:
        return args.annotation_path
    return annotation_path(args.annotation_root, task)


def _resolve_result_file(result_dir: str, task: str, model: str):
    """Find the result file for a task/model, preferring a finished run."""
    key = task_key(task)

    final_path = os.path.join(result_dir, f"{key}_{model}.json")
    if os.path.exists(final_path):
        return final_path, "final"

    temp_path = os.path.join(result_dir, f"{key}_{model}.temp.json")
    if os.path.exists(temp_path):
        return temp_path, "temp"

    final_matches = sorted(
        p
        for p in glob.glob(os.path.join(result_dir, f"{key}_{model}_*.json"))
        if not p.endswith(".temp.json")
    )
    if len(final_matches) == 1:
        return final_matches[0], "final"

    temp_matches = sorted(glob.glob(os.path.join(result_dir, f"{key}_{model}_*.temp.json")))
    if len(temp_matches) == 1:
        return temp_matches[0], "temp"

    if len(final_matches) > 1 or len(temp_matches) > 1:
        print(
            f"[WARN] Multiple result files matched task '{task}' and model '{model}'. "
            "Narrow --result_dir or pass the full model variant name.",
            file=sys.stderr,
        )
        for match in final_matches + temp_matches:
            print(f"[WARN]   {match}", file=sys.stderr)

    return None, None


def _ask_time(turn: dict) -> float:
    try:
        return float(turn.get("ask_time", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _route_by_time(response_time: float, turns: list, asked_turn_idxs: set) -> int:
    """Attribute a response to the most recently asked turn that precedes it."""
    asked_prior = [
        idx
        for idx in asked_turn_idxs
        if 0 <= idx < len(turns) and _ask_time(turns[idx]) <= response_time
    ]
    if asked_prior:
        return max(asked_prior, key=lambda idx: _ask_time(turns[idx]))

    all_prior = [
        idx for idx, turn in enumerate(turns) if _ask_time(turn) <= response_time
    ]
    if all_prior:
        return max(all_prior, key=lambda idx: _ask_time(turns[idx]))

    return 0


def _route_question_event(event: dict, turns: list, claimed_turn_idxs: set) -> int:
    """Match a question event to its turn by text, then by ask-time proximity."""
    event_text = str(event.get("value", "")).strip().lower()
    event_time = float(event.get("time", 0.0))

    if event_text:
        for idx, turn in enumerate(turns):
            if idx in claimed_turn_idxs:
                continue
            turn_text = str(turn.get("question", "")).strip().lower()
            if turn_text and turn_text == event_text:
                return idx

    unclaimed = [idx for idx in range(len(turns)) if idx not in claimed_turn_idxs]
    if unclaimed:
        return min(unclaimed, key=lambda idx: abs(_ask_time(turns[idx]) - event_time))

    return _route_by_time(event_time, turns, asked_turn_idxs=set())


def _route_events_to_turns(events, turns: list, model: str) -> list:
    """Fold a video's flat event timeline into per-turn model predictions."""
    events_sorted = (
        sorted(events, key=lambda e: e.get("time", 0.0)) if isinstance(events, list) else []
    )

    predictions = [
        {"response_time": [], "response": [], "latency": []} for _ in turns
    ]
    turn_id_to_idx: dict[str, int] = {}
    asked_turn_idxs: set[int] = set()
    claimed_turn_idxs: set[int] = set()

    for event in events_sorted:
        event_type = event.get("type")
        event_turn_id = event.get("turn_id")
        event_turn_id = event_turn_id.strip() if isinstance(event_turn_id, str) else ""

        if event_type == "question":
            if event_turn_id and event_turn_id in turn_id_to_idx:
                turn_idx = turn_id_to_idx[event_turn_id]
            else:
                turn_idx = _route_question_event(event, turns, claimed_turn_idxs)
                if event_turn_id:
                    turn_id_to_idx[event_turn_id] = turn_idx
                    turn_id_to_idx[event_turn_id.upper()] = turn_idx

            asked_turn_idxs.add(turn_idx)
            claimed_turn_idxs.add(turn_idx)
            continue

        if event_type != "response" or "time" not in event:
            continue

        response_time = float(event["time"])
        turn_idx = None
        if event_turn_id:
            turn_idx = turn_id_to_idx.get(
                event_turn_id, turn_id_to_idx.get(event_turn_id.upper())
            )
        if turn_idx is None:
            turn_idx = _route_by_time(response_time, turns, asked_turn_idxs)

        predictions[turn_idx]["response_time"].append(response_time)
        predictions[turn_idx]["response"].append(str(event.get("value", "")))
        predictions[turn_idx]["latency"].append(event.get("latency"))

    for turn_idx, pred in enumerate(predictions):
        if pred["response_time"]:
            turns[turn_idx][model] = pred

    return turns


def load_results(args):
    """Load result traces and merge them onto the ground-truth turns."""
    results: dict[str, dict] = {}

    for task in sorted(set(args.tasks)):
        results[task] = {}

        fpath, source = _resolve_result_file(args.result_dir, task, args.model)
        if fpath is None:
            print(
                f"[WARN] No results for task '{task}' and model '{args.model}'. Skipping.",
                file=sys.stderr,
            )
            continue
        if source == "temp":
            print(
                f"[WARN] Final result file missing for task '{task}'. "
                f"Scoring the in-progress file instead: {fpath}",
                file=sys.stderr,
            )
        else:
            print(f"[INFO] Loading results from {fpath}")

        with open(fpath, "r", encoding="utf-8") as f:
            model_data = json.load(f)

        anno_path = _resolve_annotation(args, task)
        if not os.path.exists(anno_path):
            print(
                f"[WARN] Annotation file not found for task '{task}': {anno_path}",
                file=sys.stderr,
            )
            continue

        with open(anno_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        for vid, events in model_data.items():
            entry = annotations.get(vid) or annotations.get(vid.split(".")[0])
            if entry is None:
                print(
                    f"[WARN] No annotation for video '{vid}' in task '{task}'. Skipping.",
                    file=sys.stderr,
                )
                continue

            turns = entry.get("turns")
            if not isinstance(turns, list) or not turns:
                turns = [entry]
            turns = copy.deepcopy(turns)

            results[task][vid] = {
                "turns": _route_events_to_turns(events, turns, args.model)
            }

        print(f"[INFO] Task {task}: loaded {len(results[task])} videos")

    return results


def save_scores(args, output):
    tasks_str = "_".join(sorted(set(args.tasks)))
    fpath = os.path.join(args.result_dir, f"{args.model}_{tasks_str}.json")

    os.makedirs(args.result_dir, exist_ok=True)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"[DONE] Saved scores to {fpath}")


def main():
    args = args_parser()
    print(
        f"[INFO]\nTasks: {args.tasks}\nModel: {args.model}\nK: {args.occupancy_k}\n"
    )

    results = load_results(args)
    output = score_results(args, results, TASK_TO_FAMILY)
    save_scores(args, output)


if __name__ == "__main__":
    main()
