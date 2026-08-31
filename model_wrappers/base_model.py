from __future__ import annotations

import abc
import json
import os
from typing import Any, Dict, List

from tqdm import tqdm


def _load_results_from_temp(temp_fp: str, resume: bool) -> Dict[str, Any]:
    if not resume:
        if os.path.exists(temp_fp):
            print(
                f"[INFO] Found temporary file at {temp_fp}, but --resume was not set. "
                "Starting a fresh run."
            )
        return {}

    if not os.path.exists(temp_fp):
        print(
            f"[INFO] --resume set, but no temporary file found at {temp_fp}. "
            "This is the first run."
        )
        return {}

    try:
        with open(temp_fp, "r") as f:
            results = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read temporary file {temp_fp}: {e}. Starting fresh.")
        return {}

    if not isinstance(results, dict):
        print(f"[WARN] Temporary file {temp_fp} is not a JSON object. Starting fresh.")
        return {}

    print(f"[INFO] Resuming from {temp_fp} with {len(results)} completed videos.")
    return results


def _cleanup_temp_file(temp_fp: str, out_fp: str) -> None:
    """Delete the checkpoint once the final file provably contains everything."""
    if not os.path.exists(temp_fp):
        return

    try:
        with open(temp_fp, "r") as f:
            temp_results = json.load(f)
        with open(out_fp, "r") as f:
            main_results = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not verify temporary file cleanup: {e}. Keeping {temp_fp}.")
        return

    temp_keys = set(temp_results.keys()) if isinstance(temp_results, dict) else set()
    main_keys = set(main_results.keys()) if isinstance(main_results, dict) else set()
    missing_in_main = temp_keys - main_keys

    if not missing_in_main:
        os.remove(temp_fp)
        print(f"[INFO] Deleted temporary file: {temp_fp}")
    else:
        print(
            "[WARN] Keeping temporary file: "
            f"{len(missing_in_main)} videos exist in temp but not in main."
        )


def get_solicited_question(task, actions):
    """Render an SI task and its ordered action list as prompt context."""
    actions = "".join([f"{i+1}. {action}\n" for i, action in enumerate(actions)])
    return f"Task: {task}\nActions:\n{actions}"


def get_procedural_question(task, actions):
    """Render an SPG task and its ordered action list as prompt context."""
    actions = "".join([f"{i+1}. {action}\n" for i, action in enumerate(actions)])
    return f"Task: {task}\nActions:\n{actions}"


class ModelStreaming(abc.ABC):
    """Base class for a model evaluated on SPOT-Bench.

    To add a model: subclass ModelStreaming, implement `inference()`, and add one entry
    to `MODEL_REGISTRY` in `model_wrappers/__init__.py`. Checkpointing, resume,
    per-task context and result layout are handled here. `self._active_task` is
    the task being run ("abd", "pnr", "sqa", "spg", "si" or "ui"); prompts belong
    in the subclass, never here.
    """

    def __init__(self, stream_fps: int = 1):
        self.stream_fps = stream_fps
        self._active_task = None

    @abc.abstractmethod
    def inference(
        self, video_path: str, turns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run the video stream and return a flat list of timeline events.

        Each event is a dict:
            {"time": float, "type": "question" | "response", "value": str}

        Response events can also carry `"latency"`: the wall-clock seconds
        from the start of generation to the response being available. Wrappers
        that track which turn a response answers should set `"turn_id"` on both
        the question and the responses to it; the scorer falls back to
        ask-time aware routing when it is absent.
        """

    @staticmethod
    def _enrich_turns_with_context(task: str, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """SI and SPG give the model the task and its ordered action list; UI is
        unsolicited, so it gets a standing monitoring instruction from t=0
        instead of a question.
        """
        turns = entry["turns"]

        if task == "si":
            context_q = get_solicited_question(
                entry.get("scenario", ""), entry.get("actions", [])
            )
            return [dict(t, question=t.get("question", "") or context_q) for t in turns]

        if task == "spg":
            actions = entry.get("turns", [{}])[0].get("response", [])
            context_q = get_procedural_question(entry.get("task", ""), actions)
            return [dict(t, question=t.get("question", "") or context_q) for t in turns]

        if task == "ui":
            ask_time = float(entry.get("ask_time", 0.0))
            return [
                {
                    "question": "Monitor and warn about safety-critical events.", # placeholder
                    "ask_time": ask_time,
                    "response_time": [],
                    "response": [],
                    "concurrent": False,
                    "referential": False,
                }
            ] + list(turns)

        return turns

    def eval(
        self,
        *,
        task: str,
        annotations: Dict[str, Any],
        video_root: str,
        model_name: str,
        result_path: str,
        resume: bool = False,
    ) -> Dict[str, Any]:
        """Stream every video and write `{task}_{model}.json`.

        Progress is checkpointed to `{task}_{model}.temp.json` after each video and
        `resume` continues from it.
        """
        self._active_task = task

        os.makedirs(result_path, exist_ok=True)
        temp_fp = os.path.join(result_path, f"{task}_{model_name}.temp.json")
        out_fp = os.path.join(result_path, f"{task}_{model_name}.json")
        results: Dict[str, Any] = _load_results_from_temp(temp_fp=temp_fp, resume=resume)

        for vid, entry in tqdm(annotations.items(), desc=f"Running {task}"):
            if resume and vid in results:
                print(f"[INFO] Video {vid} already done, skipping.")
                continue

            video_fp = os.path.join(video_root, f"{vid.replace('.mp4', '')}.mp4")
            turns = self._enrich_turns_with_context(task, entry)

            results[vid] = self.inference(video_path=video_fp, turns=turns)

            with open(temp_fp, "w") as f:
                json.dump(results, f, indent=4)

        with open(out_fp, "w") as f:
            json.dump(results, f, indent=4)

        print(f"[DONE] Saved results to {out_fp}")
        _cleanup_temp_file(temp_fp=temp_fp, out_fp=out_fp)
        return results
