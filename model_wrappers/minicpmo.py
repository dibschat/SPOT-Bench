from __future__ import annotations

import math
import os
import re
import time
import types
from typing import Any, Dict, List

import torch
import decord
from PIL import Image
from transformers import AutoModel

from .base_model import ModelStreaming

DEFAULT_MODEL_PATH = "openbmb/MiniCPM-o-4_5"

_THINK_RE = re.compile(r"<think>.*?</think>|</?think>", re.DOTALL)
_SENT_END_RE = re.compile(r"[.!?。！？]+(?=\s|$)")


class MiniCPM_o(ModelStreaming):
    SYSTEM = "Streaming Omni Conversation."
    STREAM_FPS = 2
    MAX_PIXELS = 262144
    CONTEXT_MAX_UNITS = 8
    TEMPERATURE = 0.7
    TOP_K = 100
    TOP_P = 0.8

    WAIT = "Do not reply right away. Keep watching, and reply only at the moment it happens."
    UI_PROMPT = (
        "Describe the video as concise live guidance for a blind person. Mention "
        "hazards, obstacles, traffic, steps, edges, or needed actions when visible."
    )
    SI_PROMPT = (
        "Watch the person perform the ordered task below. Stay silent while they "
        "proceed correctly. Speak only when the current video shows a mistake or "
        "hesitation. Then give one short imperative instruction correcting that "
        "specific error. Do not announce routine next steps."
    )
    SPG_PROMPT = (
        "Track progress through the ordered task below. Speak only when the video "
        "shows that the current step is complete and the next listed action is due. "
        "Output one short imperative instruction containing that next action. "
        "Do not announce future steps early."
    )

    def __init__(self, args):
        super().__init__(stream_fps=self.STREAM_FPS)
        model = AutoModel.from_pretrained(
            getattr(args, "model_path", None) or DEFAULT_MODEL_PATH,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )
        self.model = model.eval().cuda().as_duplex(
            generate_audio=False,
            sliding_window_mode="context",
            context_max_units=self.CONTEXT_MAX_UNITS,
        )
        self._keep_prompts_in_previous()

    def _keep_prompts_in_previous(self):
        decoder = self.model.decoder
        native_extract = decoder._extract_generated_text

        def extract(decoder_self, units):
            gen_text, gen_tokens = native_extract(units)
            parts = [str(u.get("input_text") or "").strip() for u in units]
            parts = [p for p in parts if p]
            if not parts:
                return gen_text, gen_tokens
            text = "\n".join(parts) + "\n"
            tokens = decoder_self.tokenizer.encode(text, add_special_tokens=False)
            return text + gen_text, tokens + gen_tokens

        decoder._extract_generated_text = types.MethodType(extract, decoder)

    def _prompt(self, task: str, question: str) -> str:
        if task == "ui":
            return self.UI_PROMPT
        if task == "si":
            return f"{self.SI_PROMPT}\n{question}"
        if task == "spg":
            return f"{self.SPG_PROMPT} {self.WAIT}\n{question}"
        return f"{question} {self.WAIT}"

    @staticmethod
    def _safe_ask_time(tn: Dict[str, Any]) -> float:
        try:
            return float(tn.get("ask_time", 0.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _join(prev: str, piece: str) -> str:
        if not prev or not piece:
            return prev + piece
        if piece[0].isspace() or prev[-1].isspace() or piece[0] in ".,!?;:'\")]}%":
            return prev + piece
        return prev + " " + piece

    @staticmethod
    def _clean(text: str) -> str:
        s = " ".join(_THINK_RE.sub("", text or "").split())
        return s if any(ch.isalnum() for ch in s) else ""

    def _frame(self, reader, idx: int) -> Image.Image:
        img = Image.fromarray(reader[idx].asnumpy())
        w, h = img.size
        if w * h > self.MAX_PIXELS:
            s = math.sqrt(self.MAX_PIXELS / float(w * h))
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
        return img

    def inference(self, video_path: str, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not os.path.exists(video_path):
            print(f"[MiniCPM-o] Missing video: {video_path}")
            return []
        try:
            reader = decord.VideoReader(video_path, num_threads=2)
        except Exception as e:
            print(f"[MiniCPM-o] Failed to read video {video_path}: {e}")
            return []
        total_frames = len(reader)
        if total_frames == 0:
            return []
        src_fps = float(reader.get_avg_fps() or 0.0) or float(self.stream_fps)
        duration = total_frames / src_fps

        task = (self._active_task or "").lower()
        is_turn_qa = task in {"pnr", "abd", "sqa"}

        events: List[Dict[str, Any]] = []
        turn_meta: List[Dict[str, Any]] = []
        for tn in sorted(turns, key=self._safe_ask_time):
            question = str(tn.get("question") or "").strip()
            if not question:
                continue
            turn_id = f"Q{len(turn_meta) + 1}"
            ask_time = self._safe_ask_time(tn)
            turn_meta.append({"turn_id": turn_id, "ask_time": ask_time, "question": question})
            if is_turn_qa:
                events.append({"time": float(round(ask_time, 3)), "type": "question",
                               "value": question, "turn_id": turn_id})

        torch.cuda.empty_cache()
        self.model.prepare(prefix_system_prompt=self.SYSTEM)
        if task in {"pnr", "abd"}:
            gen_kwargs = {"decode_mode": "greedy"}
        else:
            gen_kwargs = {"decode_mode": "sampling", "temperature": self.TEMPERATURE,
                          "top_k": self.TOP_K, "top_p": self.TOP_P}

        buf: Dict[str, Any] | None = None
        last_turn_id: str | None = None

        def emit(text: str, t: float, latency: float, tid: str | None):
            value = self._clean(text)
            if not value or tid is None:
                return
            ev = {"time": t, "type": "response", "value": value, "raw_text": text,
                  "latency": float(round(latency, 4))}
            if is_turn_qa:
                ev["turn_id"] = tid
            events.append(ev)

        def flush():
            nonlocal buf
            if buf is not None:
                emit(buf["text"], buf["t"], buf["lat"], buf["tid"])
            buf = None

        for k in range(max(1, math.ceil(duration))):
            t0 = time.perf_counter()
            frames = []
            for j in range(self.stream_fps):
                sub_t = k + j / float(self.stream_fps)
                if j > 0 and sub_t >= duration:
                    break
                frames.append(self._frame(reader, min(total_frames - 1, int(round(sub_t * src_fps)))))

            active = [m for m in turn_meta if m["ask_time"] <= k]
            turn_id = active[-1]["turn_id"] if active else None
            text_list = None
            if active and turn_id != last_turn_id:
                text_list = [self._prompt(task, active[-1]["question"])]
                last_turn_id = turn_id

            self.model.streaming_prefill(frame_list=frames, text_list=text_list, batch_vision_feed=True)
            result = self.model.streaming_generate(**gen_kwargs)
            if text_list and self.model.decoder._unit_history:
                self.model.decoder._unit_history[-1]["input_text"] = text_list[0]
            compute = time.perf_counter() - t0
            t_dec = float(round(min(duration, result["current_time"]), 3))

            if result["is_listen"]:
                flush()
                continue
            if buf is None:
                buf = {"text": "", "t": t_dec, "lat": compute, "tid": turn_id}
            buf["text"] = self._join(buf["text"], result["text"])
            while True:
                m = _SENT_END_RE.search(buf["text"])
                if not m:
                    break
                emit(buf["text"][: m.end()], buf["t"], buf["lat"], buf["tid"])
                buf = {"text": buf["text"][m.end():].strip(), "t": t_dec, "lat": compute, "tid": turn_id}
            if not buf["text"]:
                buf = None
            if result["end_of_turn"]:
                flush()
        flush()

        events.sort(key=lambda x: (x.get("time", 0.0), 0 if x.get("type") == "question" else 1))
        return events
