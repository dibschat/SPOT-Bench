"""Qwen-VL sliding-window (memoryless) baseline for SPOT-Bench.
"""

import math
import os
import re
from collections import defaultdict, deque
from typing import Any, Dict, List

import torch
import decord

import numpy as np
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForVision2Seq, AutoProcessor

from .base_model import ModelStreaming
from .utils import GenerationTimer, frames_to_base64_images

QWEN_VARIANTS = {
    "2.5": "Qwen/Qwen2.5-VL-7B-Instruct",
    "3": "Qwen/Qwen3-VL-8B-Instruct",
}
DEFAULT_QWEN_VERSION = "2.5"


class QwenVL(ModelStreaming):
    STREAM_WINDOW_SECONDS = 4.0
    MAX_NEW_TOKENS = 512
    DETECTION_MAX_NEW_TOKENS = 8
    DETECTION_TASKS = {"abd", "pnr"}
    FRAME_SHORT_SIDE = 560
    PATCH_FACTOR = 28
    QUERY_ROUNDS = 4

    def __init__(self, args):
        super().__init__(stream_fps=2)
        self.args = args
        self.decision_fps = 1
        if self.stream_fps != 2:
            raise ValueError("QwenVL expects the 2-FPS streaming input rate")

        self.version = self._normalize_version(getattr(args, "qwen_version", None))
        self.window_frames = int(round(self.STREAM_WINDOW_SECONDS * self.stream_fps))
        self._model_init()

    @staticmethod
    def _normalize_version(version: str | None) -> str:
        raw = str(version or DEFAULT_QWEN_VERSION).strip().lower().lstrip("v")
        if raw in {"2.5", "25", "qwen2.5", "qwen2_5"}:
            return "2.5"
        if raw in {"3", "3.0", "qwen3"}:
            return "3"
        raise ValueError(
            f"Unsupported qwen_version={version!r}. Choose one of: "
            f"{', '.join(sorted(QWEN_VARIANTS))}"
        )

    def _model_init(self):
        model_path = getattr(self.args, "model_path", None) or QWEN_VARIANTS[self.version]
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
        except (KeyError, ValueError) as e:
            import transformers

            raise RuntimeError(
                f"Could not load Qwen{self.version}-VL from {model_path} with "
                f"transformers {transformers.__version__}. Qwen3-VL needs >= 4.57."
            ) from e

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = next(self.model.parameters()).device
        print(
            f"[QwenVL] Initialized Qwen{self.version}-VL from {model_path}. "
            f"window={self.window_frames} frames "
            f"({self.STREAM_WINDOW_SECONDS:g}s @ {self.stream_fps} FPS). "
            f"Datatype: {self.model.dtype}"
        )

    PNR_PROMPT = """
You are an online video reasoning system observing a live video stream.
You continuously observe a live video stream, but you must only use information *visible so far* — never guess the future.

At this moment, decide whether the described event in the question has already occurred in the video segment you have just seen.
Answer **for the last frame only**, giving strongest weight to the **latest frames** (≈ last 3s).

Be strict:
- If the event has *not yet happened by the last frame*, respond exactly with **"no"**.
- If the event *has just happened by the last frame* (clearly visible within the last few seconds), respond exactly with **"now"**.
- Do **not** explain or add context. No timestamps, no numbers, no extra words.
- When in doubt, choose "no".

Only output either:
- "no"
- "now"

Question: {question}
"""

    ABD_PROMPT = """
You are an online video reasoning system observing a live video stream.
You continuously observe a live video stream, but you must only use information *visible so far* — never guess the future.

At this moment, decide whether the described event in the question has already occurred in the video segment you have just seen.
Answer **for the last frame only**, giving strongest weight to the **latest frames** (≈ last 3s).

Definitions:
{definitions}

Be strict:
- If the event has *not yet happened by the last frame*, respond exactly with **"no"**.
{rules}
- Do **not** explain or add context. No timestamps, no numbers, no extra words.
- When in doubt, choose "no".

Only output exactly one of:
{outputs}

Question: {question}
"""

    SQA_PROMPT = """
You are an online video reasoning system observing a live video stream.
You continuously observe a live video stream, but you must only use information *visible so far* — never guess the future.

At this moment, decide whether the existing visual content, especially the most recent frames near the end of this segment,
provides enough information to confidently answer the question.
Answer **for the last frame only**, giving strongest weight to the **latest frames** (≈ last 3s).

Be strict:
- If there is not enough evidence yet in the last frame, respond exactly with **"no"**.
- If the evidence is visible and sufficient in the last frame, answer the question directly and concisely.
- Do **not** explain or add context. No timestamps, no numbers, no extra words.
- When in doubt, choose 'no'.

Only output either:
- 'no', or
- a single short answer to the question (≤ 12 words)

Question: {question}
"""

    SPG_PROMPT = """
You are an online visual assistant observing a live video feed of a person performing a known task.
You continuously observe a live video stream, but you must only use information *visible so far* — never guess the future.

The task and its steps, in order:
{question}

Decide, from these frames alone, whether a step has just been completed.

Rules:
- Use only the most recent frames (≈ last 3s), and answer for the last frame only.
- If these frames show one of the listed steps being completed, reply with the **next**
  step in the list, copied as written.
- If the video has only just begun and no step has started yet, reply with step 1.
- Otherwise reply exactly "no".
- Do not announce a step early: reply only at the moment its predecessor finishes.
- Reply with the step text itself, never its number, and add nothing else.

Reply with exactly one line: either "no", or one step copied from the list above.
"""

    SI_PROMPT = """
You are an online visual assistant observing a live egocentric video feed of a person performing a known task.
You continuously observe a live video stream, but you must only use information *visible so far* — never guess the future.

You know the task and its ordered actions:
{question}

Your goal is to decide whether you should **intervene now** to help the person proceed correctly.
Answer **for the last frame only**, giving strongest weight to the **latest frames** (≈ last 3s).

Guidelines:
- If the person is **progressing normally by the last frame** or **has just completed a step successfully**, respond exactly with **"no"**.
- If the person **appears stuck/hesitant/incorrect by the last frame**, respond with **one short instruction** that guides them to the correct next action.
- Keep the guidance concise and action-specific. No narration, no timestamps, no numbers, no extra words.
- When in doubt, choose 'no'.

Only output either:
- "no", or
- one short imperative instruction (≤ 20 words)
"""

    UI_PROMPT = """
You are an online visual assistant observing a live egocentric video feed from a blind user’s perspective.
You continuously observe a live video stream, but you must only use information *visible so far* — never guess the future.

Your goal is to decide whether you should **intervene now** with a short, helpful spoken warning or guidance.
Answer **for the last frame only**, giving strongest weight to the **latest frames** (≈ last 3s).

Guidelines:
- If **no immediate risk or obstacle is visible by the last frame**, respond exactly with **"no"**.
- If **a clear, time-sensitive risk is visible by the last frame**, respond with **one concise spoken warning**.
- Be factual and specific to what is visible. No explanations, no timestamps, no numbers, no extra words.
- When in doubt, choose "no".

Only output either:
- "no", or
- one short warning sentence (≤ 20 words)

Your response should sound like a real-time verbal cue.
"""

    _ABD_RESPOND_WORDS_RE = re.compile(
        r"\bRespond\b.*?\bwords?\s*:\s*(?P<words>.+?)\s*$", re.IGNORECASE
    )
    _ABD_START_WORD_RE = re.compile(r"^(?:start|begin|onset|commence)\w*$")

    @staticmethod
    def _strip_trailing_respond_sentence(question: str) -> str:
        text = " ".join(str(question or "").split()).strip()
        if not text:
            return ""
        return re.sub(r"([.!?])\s*Respond\b.*$", r"\1", text, flags=re.IGNORECASE).strip()

    @classmethod
    def _abd_labels(cls, raw_question: str) -> list[tuple[str, str]]:
        """`[(role, word)]` in query order; `word` is the literal label it names."""
        match = cls._ABD_RESPOND_WORDS_RE.search(raw_question)
        if not match:
            return [("start", "start"), ("end", "end")]

        labels: list[tuple[str, str]] = []
        for chunk in re.split(r"\s*(?:,|\band\b|\bor\b|/)\s*", match.group("words")):
            word = chunk.strip().strip(".,;:").lower()
            if not word or any(word == seen for _, seen in labels):
                continue
            role = "start" if cls._ABD_START_WORD_RE.match(word) else "end"
            labels.append((role, word))

        return labels or [("start", "start"), ("end", "end")]

    @classmethod
    def _abd_prompt(cls, raw_questions: List[str], question_text: str) -> str:
        """Concurrent turns can name different boundaries, so the shared blocks
        carry the union of their labels in the order first seen."""
        labels: list[tuple[str, str]] = []
        for raw in raw_questions:
            for role, word in cls._abd_labels(raw):
                if not any(word == seen for _, seen in labels):
                    labels.append((role, word))
        if not labels:
            labels = [("start", "start"), ("end", "end")]
        width = max(len(word) for _, word in labels)

        definition_text = {
            "start": "the first intentional motion that commits to the step (earliest unique onset).",
            "end": "the completion or achieved goal-state (state flip / stopping point / fully done).",
        }
        definitions = "\n".join(
            f'- "{word}"{" " * (width - len(word))} = {definition_text[role]}'
            for role, word in labels
        )

        rules = []
        for role, word in sorted(labels, key=lambda rw: rw[0] != "end"):
            if role == "end":
                rules.append(
                    f"- If **completion is visible by the last frame** (within the "
                    f'recency window), respond exactly with **"{word}"**.'
                )
            elif rules:
                rules.append(
                    f"- Otherwise, if **onset is visible by the last frame**, "
                    f'respond exactly with **"{word}"**.'
                )
            else:
                rules.append(
                    f"- If **onset is visible by the last frame**, "
                    f'respond exactly with **"{word}"**.'
                )
        if len(labels) > 1:
            order = " > ".join([word for _, word in labels][::-1] + ["no"])
            rules.append(f"- Tie-break at the last frame: **{order}**.")

        outputs = "\n".join(f'- "{word}"' for _, word in labels) + '\n- "no"'

        return cls.ABD_PROMPT.format(
            definitions=definitions,
            rules="\n".join(rules),
            outputs=outputs,
            question=question_text,
        )

    TAGGED_TASKS = {"abd", "sqa"}
    UNTAGGED_TASKS = {"si", "spg", "ui"}

    def _query_lines(self, task: str, active: List[Dict[str, Any]]) -> List[str]:
        lines = []
        for item in active:
            question = item["question"]
            if task in self.UNTAGGED_TASKS:
                lines.append(question)
            elif task in self.TAGGED_TASKS:
                lines.append(f"{item['turn_id']}: {question}")
            else:
                lines.append(question)
        return lines

    def _build_prompt(self, task: str, active: List[Dict[str, Any]]) -> str:
        task = (task or "").lower()
        if task == "ui":
            return self.UI_PROMPT

        raw = [item["question"] for item in active]
        if task == "abd":
            question_text = "\n".join(
                f"{item['turn_id']}: {self._strip_trailing_respond_sentence(item['question'])}"
                for item in active
            )
            return self._abd_prompt(raw, question_text)

        question_text = "\n".join(self._query_lines(task, active))
        if task == "sqa":
            return self.SQA_PROMPT.format(question=question_text)
        if task == "pnr":
            return self.PNR_PROMPT.format(
                question="\n".join(
                    self._strip_trailing_respond_sentence(q) for q in raw
                )
            )
        if task == "spg":
            return self.SPG_PROMPT.format(question=question_text)
        if task == "si":
            return self.SI_PROMPT.format(question=question_text)
        raise NotImplementedError(f"Unsupported task: {task}")

    _RESPONSE_TAG_RE = re.compile(r"^\s*(?P<tag>Q\d+)\s*[:.\)-]?\s*(?P<value>.*)$")

    def _split_responses(
        self, text: str, task: str, active: List[Dict[str, Any]]
    ) -> List[tuple[str, str]]:
        """Route a generation back to turn ids. A tagged task may answer several
        turns at once, one per line; unrecognised tags fall back to the latest."""
        fallback = active[-1]["turn_id"] if active else ""
        raw = str(text or "").strip()
        if not raw:
            return []

        if task not in self.TAGGED_TASKS or len(active) == 1:
            return [(fallback, raw)]

        known = {item["turn_id"].upper(): item["turn_id"] for item in active}
        out: List[tuple[str, str]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            match = self._RESPONSE_TAG_RE.match(line)
            if match and match.group("tag").upper() in known:
                out.append((known[match.group("tag").upper()], match.group("value").strip()))
            else:
                out.append((fallback, line))
        return out or [(fallback, raw)]

    @staticmethod
    def _append_question_event(
        events: List[Dict[str, Any]], ask_time: float, question: str, turn_id: str
    ):
        events.append(
            {
                "time": float(round(ask_time, 3)),
                "type": "question",
                "value": question,
                "turn_id": turn_id,
            }
        )

    @classmethod
    def _append_response_event(
        cls,
        events: List[Dict[str, Any]],
        t: float,
        value: str,
        turn_id: str,
        latency_s: float | None = None,
    ):
        event = {
            "time": float(round(t, 3)),
            "type": "response",
            "value": value,
            "turn_id": turn_id,
        }
        if latency_s is not None and not cls._is_silent(value):
            event["latency"] = float(round(latency_s, 4))
        events.append(event)

    @staticmethod
    def _is_silent(text: str) -> bool:
        s = str(text).strip().lower()
        if not s or not any(ch.isalnum() for ch in s):
            return True
        return s in {"no", "none", "nothing", "no answer", "no response"}

    def _group_questions_by_tick(self, turns: List[Dict[str, Any]]):
        grouped = defaultdict(list)
        q_idx = 0
        for turn in sorted(turns, key=lambda tn: float(tn.get("ask_time", 0.0) or 0.0)):
            question = str(turn.get("question", "") or "").strip()
            if not question:
                continue
            ask_time = float(turn.get("ask_time", 0.0) or 0.0)
            grouped[max(0, int(ask_time * self.stream_fps))].append(
                {
                    "turn_id": f"Q{q_idx + 1}",
                    "ask_time": ask_time,
                    "question": question,
                }
            )
            q_idx += 1
        return grouped

    @classmethod
    def _resized_dims(cls, height: int, width: int) -> tuple[int, int]:
        """Short side capped at 560, both sides snapped to the 28-px patch grid."""
        short_side = min(height, width)
        scale = cls.FRAME_SHORT_SIDE / short_side if short_side > cls.FRAME_SHORT_SIDE else 1.0
        snap = lambda x: max(cls.PATCH_FACTOR, int(round(x * scale / cls.PATCH_FACTOR)) * cls.PATCH_FACTOR)
        return snap(height), snap(width)

    @staticmethod
    def _resize_frame(frame_rgb: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        """Resize here, not in the processor: frames reach `process_vision_info` as
        base64 PNGs, so a 4K frame costs 199 MB of arrays and 266 MB of base64 per
        tick against 14 MB / 18 MB once resized."""
        height, width = target_hw
        if frame_rgb.shape[0] == height and frame_rgb.shape[1] == width:
            return frame_rgb
        return np.asarray(
            Image.fromarray(frame_rgb).resize((width, height), Image.BICUBIC)
        )

    def _build_messages(self, prompt: str, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        frame_array = np.stack(frames, axis=0)
        resized_height, resized_width = (int(x) for x in frame_array.shape[1:3])
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames_to_base64_images(frame_array),
                        "resized_height": resized_height,
                        "resized_width": resized_width,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _generate_response(self, prompt: str, frames: List[np.ndarray]) -> str:
        messages = self._build_messages(prompt, frames)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        is_detection = (self._active_task or "").lower() in self.DETECTION_TASKS
        gen_kwargs: dict = {
            "max_new_tokens": (
                self.DETECTION_MAX_NEW_TOKENS if is_detection else self.MAX_NEW_TOKENS
            )
        }
        if is_detection:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def inference(self, video_path: str, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not os.path.exists(video_path):
            print(f"[QwenVL] Missing video: {video_path}")
            return []

        try:
            frames = decord.VideoReader(video_path, num_threads=2)
        except Exception as e:
            print(f"[QwenVL] Failed to read video {video_path}: {e}")
            return []

        total_frames = int(len(frames))
        if total_frames == 0:
            return []

        src_fps = float(frames.get_avg_fps() or 0.0)
        if src_fps <= 0:
            src_fps = float(self.stream_fps)
        duration = float(total_frames / src_fps)
        if duration <= 0:
            return []

        task = (self._active_task or "").lower()
        step = 1.0 / float(self.stream_fps)
        num_ticks = max(1, int(math.ceil(duration * self.stream_fps)))
        decision_every = max(1, int(round(self.stream_fps / max(self.decision_fps, 1))))
        questions_by_tick = self._group_questions_by_tick(turns)

        recent_frames: deque[np.ndarray] = deque(maxlen=self.window_frames)
        target_hw: tuple[int, int] | None = None
        active: deque[Dict[str, Any]] = deque(maxlen=self.QUERY_ROUNDS)
        events: List[Dict[str, Any]] = []

        for tick in range(num_ticks):
            tick_time = tick * step
            frame_idx = min(total_frames - 1, max(0, int(round(tick_time * src_fps))))
            frame_rgb = frames[frame_idx].asnumpy()
            if target_hw is None:
                target_hw = self._resized_dims(frame_rgb.shape[0], frame_rgb.shape[1])
            recent_frames.append(self._resize_frame(frame_rgb, target_hw))

            for item in questions_by_tick.get(tick, []):
                active.append(item)
                self._append_question_event(
                    events,
                    ask_time=float(item["ask_time"]),
                    question=item["question"],
                    turn_id=item["turn_id"],
                )

            if (tick + 1) % decision_every != 0:
                continue
            if not active or not recent_frames:
                continue

            active_now = list(active)
            prompt = self._build_prompt(task, active_now)
            timer = GenerationTimer(torch).start()
            try:
                response = self._generate_response(prompt, list(recent_frames))
            except Exception as e:
                raise RuntimeError(
                    f"[QwenVL] Generation failed at t={tick_time:.1f}s in "
                    f"{video_path}. Refusing to write a partial trace; re-run "
                    "with --resume to continue from the last completed video."
                ) from e
            latency_s = timer.finish()

            response_time = min(duration, (tick + 1) * step)
            for turn_id, value in self._split_responses(response, task, active_now):
                self._append_response_event(
                    events,
                    t=response_time,
                    value=value,
                    turn_id=turn_id,
                    latency_s=latency_s,
                )

        events.sort(
            key=lambda e: (e.get("time", 0.0), 0 if e.get("type") == "question" else 1)
        )
        return events
