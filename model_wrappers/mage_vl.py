"""Mage-VL baselines for SPOT-Bench.

MageVL       Same protocol, prompts and decision loop as the QwenVL baseline (4s window
             @ 2 FPS, one decision per second); only model loading and generation differ.
MageVLCodec  Same prompts, active-query bookkeeping and 1 Hz decisions, but each decision
             at time t sees the last MAGE_CODEC_WINDOW seconds [t - W, t] through
             Mage-VL's codec-native (HEVC) path, which packs many more frames into fewer
             visual tokens. Windows are cut from the causal H.264 copies written by
             scripts/transcode_causal.sh, so no input frame is later than t.
"""

import math
import os
import shutil
import subprocess
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import decord
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from .qwen_vl import QwenVL
from .utils import GenerationTimer

MAGE_VL_REPO = "microsoft/Mage-VL"
MAGE_VL_REVISION = "d88b153285f1633a61b2f693c59c8576693af185"


class MageVL(QwenVL):
    PATCH_FACTOR = 32  # Mage-ViT 16-px patches, 2x2 spatial merge

    def _model_init(self):
        model_path = getattr(self.args, "model_path", None) or MAGE_VL_REPO
        revision = None if model_path != MAGE_VL_REPO else MAGE_VL_REVISION
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            revision=revision,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, revision=revision, trust_remote_code=True
        )
        self.device = next(self.model.parameters()).device
        print(
            f"[MageVL] Initialized Mage-VL from {model_path}. "
            f"window={self.window_frames} frames "
            f"({self.STREAM_WINDOW_SECONDS:g}s @ {self.stream_fps} FPS). "
            f"Datatype: {self.model.dtype}"
        )

    def _chat_text(self, prompt: str) -> str:
        messages = [
            {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}
        ]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _generate(self, inputs) -> str:
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

        is_detection = (self._active_task or "").lower() in self.DETECTION_TASKS
        max_new_tokens = self.DETECTION_MAX_NEW_TOKENS if is_detection else self.MAX_NEW_TOKENS
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        return self.processor.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

    def _generate_response(self, prompt: str, frames: List[np.ndarray]) -> str:
        # Frames are already resized; pre-decoded frames are stamped <0.0 seconds>,
        # <1.0 seconds>, ... by the processor.
        inputs = self.processor(
            text=[self._chat_text(prompt)], videos=[list(frames)], return_tensors="pt", padding=True
        )
        return self._generate(inputs)


class MageVLCodec(MageVL):
    CAUSAL_VIDEO_ROOT = os.environ.get("MAGE_CODEC_VIDEOS", "data/videos_causal")
    WINDOW_SECONDS = float(os.environ.get("MAGE_CODEC_WINDOW", 8))
    TARGET_CANVAS = int(os.environ.get("MAGE_CODEC_CANVAS", 8))
    MAX_PIXELS = 150000
    PREFETCH = int(os.environ.get("MAGE_CODEC_PREFETCH", 8))

    def _model_init(self):
        super()._model_init()
        self.tmp_root = Path(tempfile.mkdtemp(prefix="mage_codec_"))
        self.pool = ThreadPoolExecutor(self.PREFETCH)
        self.codec_failures = 0
        print(
            f"[MageVLCodec] window={self.WINDOW_SECONDS:g}s, target_canvas={self.TARGET_CANVAS} "
            f"(<= {self.TARGET_CANVAS // 4 * 32} sampled frames), max_pixels={self.MAX_PIXELS}"
        )

    def _codec_inputs(self, video_fp: str, t_end: float, prompt: str):
        """Cut [t_end - W, t_end] (the cut can start up to one keyframe earlier, never
        later than t_end) and run it through the codec processor."""
        work = Path(tempfile.mkdtemp(dir=self.tmp_root))
        try:
            clip = work / "clip.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                    "-ss", f"{max(0.0, t_end - self.WINDOW_SECONDS):.3f}", "-to", f"{t_end:.3f}",
                    "-i", video_fp, "-c", "copy", "-avoid_negative_ts", "make_zero", str(clip),
                ],
                check=True,
                timeout=120,
            )
            return self.processor(
                text=[self._chat_text(prompt)],
                videos=[str(clip)],
                video_backend="codec",
                max_pixels=self.MAX_PIXELS,
                codec_config={"patch": 16, "target_canvas": self.TARGET_CANVAS, "cache_root": work},
                return_tensors="pt",
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _decisions(self, turns: List[Dict[str, Any]], duration: float):
        """(response_time, prompt, active turns) for every decision, with the QwenVL
        tick/TTL bookkeeping; a decision only sees questions asked by then."""
        task = (self._active_task or "").lower()
        step = 1.0 / float(self.stream_fps)
        num_ticks = max(1, int(math.ceil(duration * self.stream_fps)))
        decision_every = max(1, int(round(self.stream_fps / max(self.decision_fps, 1))))
        questions_by_tick = self._group_questions_by_tick(turns)
        active: deque = deque(maxlen=self.QUERY_ROUNDS)
        for tick in range(num_ticks):
            tick_time = tick * step
            new = questions_by_tick.get(tick, [])
            active.extend(new)
            if task in self.TTL_TASKS:
                active = deque(
                    (q for q in active if tick_time - q["ask_time"] < self.QUERY_TTL_SECONDS),
                    maxlen=self.QUERY_ROUNDS,
                )
            if (tick + 1) % decision_every != 0 or not active:
                yield tick, new, None
                continue
            active_now = list(active)
            yield tick, new, (min(duration, (tick + 1) * step), self._build_prompt(task, active_now), active_now)

    def inference(self, video_path: str, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        video_fp = os.path.join(self.CAUSAL_VIDEO_ROOT, os.path.basename(video_path))
        if not os.path.exists(video_fp):
            print(f"[MageVLCodec] Missing causal video: {video_fp}")
            return []
        reader = decord.VideoReader(video_fp, num_threads=1)
        duration = len(reader) / float(reader.get_avg_fps() or self.stream_fps)
        del reader
        if duration <= 0:
            return []

        task = (self._active_task or "").lower()
        events: List[Dict[str, Any]] = []
        steps = list(self._decisions(turns, duration))
        for _, new, _ in steps:
            for item in new:
                self._append_question_event(
                    events, ask_time=float(item["ask_time"]), question=item["question"], turn_id=item["turn_id"]
                )

        decisions = [d for _, _, d in steps if d is not None]
        last_reply: Dict[str, str] = {}
        pending: deque = deque()
        submitted = 0
        for response_time, _, active_now in decisions:
            while submitted < len(decisions) and len(pending) < 2 * self.PREFETCH:
                t_end, prompt, _ = decisions[submitted]
                pending.append(self.pool.submit(self._codec_inputs, video_fp, t_end, prompt))
                submitted += 1
            future = pending.popleft()
            try:
                inputs = future.result()
            except Exception as e:  # counted and reported; the decision stays silent
                self.codec_failures += 1
                print(f"[MageVLCodec] codec failed at t={response_time:.1f}s in {video_fp}: {e}")
                continue

            timer = GenerationTimer(torch).start()
            response = self._generate(inputs)
            latency_s = timer.finish()
            for turn_id, value in self._split_responses(response, task, active_now):
                repeat = last_reply.get(turn_id) == value.strip().lower()
                last_reply[turn_id] = value.strip().lower()
                if self.no_repeat and repeat:
                    continue
                self._append_response_event(
                    events, t=response_time, value=value, turn_id=turn_id, latency_s=latency_s
                )

        if self.codec_failures:
            print(f"[MageVLCodec] codec failures so far: {self.codec_failures}")
        events.sort(key=lambda e: (e.get("time", 0.0), 0 if e.get("type") == "question" else 1))
        return events
