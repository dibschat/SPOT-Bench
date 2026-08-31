from __future__ import annotations

import base64
import io
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import decord
import requests
from PIL import Image

from model_wrappers.base_model import ModelStreaming


SILENCE_TOKEN = "</silence>"
RESPONSE_TOKEN = "</response>"
DETECTION_GENERATION_PARAMS = {
    "max_tokens": 4,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}


class JoyAI_VL(ModelStreaming):
    def __init__(self, args, stream_fps: int | None = None):
        self.args = args
        self.api_base = (
            getattr(args, "api_base", None)
            or os.environ.get("JOYAI_VL_API_BASE", "http://127.0.0.1:8070/v1")
        ).rstrip("/")
        self.model = (
            getattr(args, "model_name", None)
            or os.environ.get("JOYAI_VL_MODEL", "JoyAI-VL-Interaction")
        )
        self.max_pixels = int(getattr(args, "max_pixels", None) or 262144)
        self.jpeg_quality = int(getattr(args, "jpeg_quality", None) or 90)
        self.request_timeout = float(getattr(args, "request_timeout", None) or 300.0)

        fps = (
            stream_fps
            if stream_fps is not None
            else int(os.environ.get("JOYAI_VL_STREAM_FPS", "2"))
        )
        super().__init__(stream_fps=fps)
        self.decision_fps = 1
        if self.stream_fps != 2:
            raise ValueError("JoyAI latency protocol requires input=2 FPS")

        self._session = requests.Session()

        self._ui_user_prompt = (
            "Describe the video as concise live guidance for a blind person. Mention "
            "hazards, obstacles, traffic, steps, edges, or needed actions when visible."
        )
        self._si_user_prompt = (
            "Watch the person perform the ordered task below. Stay silent while they "
            "proceed correctly. Speak only when the current video shows a mistake or "
            "hesitation. Then give one short imperative instruction correcting that "
            "specific error. Do not announce routine next steps."
        )
        self._spg_user_prompt = (
            "Track progress through the ordered task below. Speak only when the video "
            "shows that the current step is complete and the next listed action is due. "
            "Output one short imperative instruction containing that next action. "
            "Do not announce future steps early."
        )

    def _append_question_event(
        self,
        events: List[Dict[str, Any]],
        ask_time: float,
        question: str,
        turn_id: str | None = None,
    ):
        event = {
            "time": float(round(ask_time, 3)),
            "type": "question",
            "value": question,
        }
        if turn_id is not None:
            event["turn_id"] = turn_id
        events.append(event)

    def _append_response_event(
        self,
        events: List[Dict[str, Any]],
        t: float,
        value: str,
        turn_id: str | None = None,
        raw_text: str | None = None,
        latency_s: float | None = None,
    ):
        event = {
            "time": float(round(t, 3)),
            "type": "response",
            "value": value,
        }
        if turn_id is not None:
            event["turn_id"] = turn_id
        if raw_text is not None:
            event["raw_text"] = raw_text
        if latency_s is not None and str(value).strip().lower() != "no":
            event["latency"] = float(round(latency_s, 4))
        events.append(event)

    def _new_turn_id(self, idx: int) -> str:
        return f"Q{idx + 1}"

    @staticmethod
    def _safe_ask_time(tn: Dict[str, Any]) -> float:
        """ask_time may be missing or None (e.g. UI's appended real turns)."""
        try:
            return float(tn.get("ask_time", 0.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _normalize_silence_output(text: str) -> str:
        """Mirror streamingvlm._normalize_silence_output: empty / punctuation-only
        collapses to the canonical 'no' (which the scorer filters)."""
        s = str(text).strip()
        if not s:
            return "no"
        if not any(ch.isalnum() for ch in s):
            return "no"
        return s

    def _question_for_prompt(self, question: str, turn_id: str | None = None) -> str:
        """Keep task cues in the user prompt; the native system prompt is untouched."""
        task = (self._active_task or "").lower()
        if task == "pnr":
            return question
        if task == "sqa":
            return question
        if task == "abd":
            return question
        if task == "ui":
            return self._ui_user_prompt
        if task == "si":
            return f"{self._si_user_prompt}\n{question}" if question else self._si_user_prompt
        if task == "spg":
            return (
                f"{self._spg_user_prompt}\n{question}"
                if question
                else self._spg_user_prompt
            )
        return question

    def _generation_params(self) -> Dict[str, float | int]:
        task = (self._active_task or "").lower()
        if task in {"pnr", "abd"}:
            return dict(DETECTION_GENERATION_PARAMS)
        return {"temperature": 0.8, "top_p": 0.9}

    @staticmethod
    def _parse_action(content: str) -> Tuple[bool, str]:
        """Return (spoke, reply_text). Relies on adapter normalize_output=True:
        content is either '</silence>' or '</response> <text>'."""
        c = (content or "").strip()
        if c.startswith(RESPONSE_TOKEN):
            reply = c[len(RESPONSE_TOKEN):].strip()
            reply = reply.split("</delegation>")[0].split("<delegation>")[0].strip()
            reply = " ".join(reply.splitlines()[0].split()) if reply else ""
            return True, reply
        return False, ""

    def _session_id(self, video_path: str) -> str:
        vid = os.path.splitext(os.path.basename(video_path))[0]
        return f"spotbench_{self._active_task}_{vid}"

    def _reset(self, session_id: str, strict: bool = False) -> None:
        """Reset a streaming session. `strict=True` (initial reset) fails fast so a
        dead adapter cannot masquerade as a valid all-silent run; the final cleanup
        reset stays best-effort."""
        try:
            resp = self._session.post(
                f"{self.api_base}/streaming/reset",
                json={},
                headers={"x-streaming-session": session_id},
                timeout=60,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            if strict:
                raise RuntimeError(
                    f"[JoyAI_VL] initial reset failed for {session_id}: {e}. "
                    "Is the webinfer adapter up on :8070? (scripts/joyaivl_serve.sh)"
                ) from e
            print(f"[JoyAI_VL] cleanup reset failed for {session_id}: {e}")

    def _downscale(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w * h <= self.max_pixels or w == 0 or h == 0:
            return img
        scale = math.sqrt(self.max_pixels / float(w * h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return img.resize((new_w, new_h), Image.BILINEAR)

    def _frame_to_data_url(self, frame_rgb) -> str:
        img = Image.fromarray(frame_rgb)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = self._downscale(img)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _step(
        self,
        session_id: str,
        data_url: str,
        prompt_text: str,
        t: float,
    ) -> str:
        content: List[Dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": data_url}}
        ]
        if prompt_text:
            content.append({"type": "text", "text": prompt_text})
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "frame_time_ranges": [f"{t:.1f} seconds"],
        }
        body.update(self._generation_params())
        resp = self._session.post(
            f"{self.api_base}/chat/completions",
            json=body,
            headers={"x-streaming-session": session_id},
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def inference(self, video_path: str, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not os.path.exists(video_path):
            print(f"[JoyAI_VL] Missing video: {video_path}")
            return []

        try:
            frames = decord.VideoReader(video_path, num_threads=2)
        except Exception as e:
            print(f"[JoyAI_VL] Failed to read video {video_path}: {e}")
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

        step = 1.0 / float(self.stream_fps)
        num_ticks = max(1, int(math.ceil(duration * self.stream_fps)))
        decision_every = max(
            1, int(round(self.stream_fps / max(self.decision_fps, 1)))
        )

        sorted_turns = sorted(turns, key=self._safe_ask_time)
        turn_meta: List[Dict[str, Any]] = []
        events: List[Dict[str, Any]] = []
        is_turn_qa = (self._active_task or "").lower() in {"pnr", "abd", "sqa"}
        q_idx = 0
        for tn in sorted_turns:
            question = str(tn.get("question") or "").strip()
            if not question:
                continue
            ask_time = self._safe_ask_time(tn)
            turn_id = self._new_turn_id(q_idx)
            q_idx += 1
            turn_meta.append(
                {"turn_id": turn_id, "ask_time": ask_time, "question": question}
            )
            if is_turn_qa:
                self._append_question_event(
                    events, ask_time=ask_time, question=question, turn_id=turn_id
                )

        session_id = self._session_id(video_path)
        self._reset(session_id, strict=True)  # fail fast if adapter is down
        last_prompt_turn_id: str | None = None
        pending_open_responses: List[Tuple[str, str, float, str | None]] = []
        pending_pnr_response: Tuple[str, str, float, str | None] | None = None
        open_tasks = {"sqa", "spg", "si", "ui"}

        try:
            for i in range(num_ticks):
                t = i * step

                is_decision = (i + 1) % decision_every == 0
                frame_idx = min(total_frames - 1, int(round(t * src_fps)))
                frame_rgb = frames[frame_idx].asnumpy()
                response_start = time.perf_counter()
                data_url = self._frame_to_data_url(frame_rgb)

                # Latest active turn (single active query -> most recent one)
                active = [m for m in turn_meta if m["ask_time"] <= t]
                if active:
                    m = active[-1]
                    turn_id = m["turn_id"]
                    if turn_id != last_prompt_turn_id:
                        prompt_text = self._question_for_prompt(m["question"], turn_id)
                        last_prompt_turn_id = turn_id
                    else:
                        prompt_text = ""
                else:
                    prompt_text, turn_id = "", None

                content = self._step(session_id, data_url, prompt_text, t)
                response_latency_s = time.perf_counter() - response_start

                # Preserve both open-ended half-tick responses for the 1-Hz readout.
                task = (self._active_task or "").lower()
                if task in open_tasks:
                    frame_spoke, frame_reply = self._parse_action(content)
                    if frame_spoke:
                        pending_open_responses.append(
                            (frame_reply, content, response_latency_s, turn_id)
                        )
                elif task == "pnr":
                    frame_spoke, frame_reply = self._parse_action(content)
                    if frame_spoke:
                        pending_pnr_response = (
                            frame_reply,
                            content,
                            response_latency_s,
                            turn_id,
                        )

                if not is_decision:
                    continue

                if turn_id is None:
                    continue

                if task in open_tasks:
                    current_responses = [
                        item for item in pending_open_responses if item[3] == turn_id
                    ]
                    pending_open_responses = []
                    if not current_responses:
                        spoke, reply = False, ""
                    else:
                        reply = "; ".join(item[0] for item in current_responses)
                        content = "; ".join(item[1] for item in current_responses)
                        response_latency_s = max(item[2] for item in current_responses)
                        spoke = True
                elif task == "pnr":
                    if (
                        pending_pnr_response is None
                        or pending_pnr_response[3] != turn_id
                    ):
                        spoke, reply = False, ""
                    else:
                        reply, content, response_latency_s, _ = pending_pnr_response
                        spoke = True
                    pending_pnr_response = None
                else:
                    spoke, reply = self._parse_action(content)
                value = self._normalize_silence_output(reply) if spoke else "no"
                self._append_response_event(
                    events,
                    t=min(duration, (i + 1) * step),
                    value=value,
                    turn_id=turn_id if is_turn_qa else None,
                    raw_text=content,
                    latency_s=response_latency_s,
                )
        finally:
            self._reset(session_id)

        events.sort(
            key=lambda x: (x.get("time", 0.0), 0 if x.get("type") == "question" else 1)
        )
        return events
