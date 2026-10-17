from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as q25

from .base_model import ModelStreaming
from .utils import GenerationTimer, verbose_log

DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

_ORIG_APPLY_ROTARY = q25.apply_multimodal_rotary_pos_emb
_CAPTURE: Dict[str, Any] = {"on": False, "keys": []}


def _capturing_apply_rotary(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    if _CAPTURE["on"]:
        _CAPTURE["keys"].append(k)
    return _ORIG_APPLY_ROTARY(q, k, cos, sin, mrope_section, unsqueeze_dim)


q25.apply_multimodal_rotary_pos_emb = _capturing_apply_rotary


@dataclass
class Segment:
    ids: torch.Tensor
    keys: List[torch.Tensor]
    values: List[torch.Tensor]
    grid: Optional[torch.Tensor] = None
    time: float = 0.0
    emb: Optional[torch.Tensor] = None
    uid: int = -1

    def __len__(self) -> int:
        return int(self.ids.numel())

    def to(self, device) -> "Segment":
        return Segment(
            ids=self.ids,
            keys=[k.to(device, non_blocking=True) for k in self.keys],
            values=[v.to(device, non_blocking=True) for v in self.values],
            grid=self.grid,
            time=self.time,
            emb=self.emb,
            uid=self.uid,
        )


@dataclass
class Query:
    tag: str
    question: str
    ask_time: float
    retrieved: List[Segment]
    answer: Optional[Segment] = None
    last_reply: Optional[Tuple[str, float]] = None


class CtrlSPOT(ModelStreaming):
    STREAM_FPS = 2
    DECISION_HZ = 1
    SHORT_TERM_FRAMES = 16
    LONG_TERM_FRAMES = 256
    RETRIEVE_TOP_K = 2
    MAX_SUMMARIES = 5
    SUMMARY_WINDOW_SECONDS = 16
    QUERY_TTL_SECONDS = 120.0
    COMPRESS_SIMILARITY = 0.95

    FRAME_SHORT_SIDE = 336
    FRAME_BLOCK = 28
    SUMMARY_MAX_NEW_TOKENS = 40
    TOKENS_PER_QUERY = 8
    MAX_NEW_TOKENS = 192

    TASKS = {"abd", "pnr"}

    SYSTEM_BASE = (
        "You are a real-time assistant watching a live video stream at two frames "
        "per second. Each frame is preceded by its timestamp, for example "
        "<12.0 seconds>.\n"
        "Questions arrive while the video plays. Each is tagged Q1, Q2, ... and "
        "shown with the time it was asked.\n"
        "Lines starting with 'You replied to' are replies you already gave. "
        "Lines starting with 'Summary of' describe earlier parts of the video.\n"
    )

    OUTPUT_RULES = {
        "abd": (
            "After each new frame, decide for every active question whether it needs a "
            "reply at this exact moment. If none does, output exactly: no\n"
            "Otherwise output one line per question that needs a reply, formatted as "
            "'Qn: reply', and nothing else."
        ),
        "pnr": (
            "After each new frame, answer every active question on its own line, "
            "formatted as 'Qn: reply', where the reply is 'no' if that question needs "
            "no reply at this exact moment. Output nothing else."
        ),
    }

    TASK_RULES = {
        "abd": (
            "Each question asks for the start and/or the end of an action step and "
            "names the word to use for each. Reply with the start word (for example "
            "'Qn: start') when the newest frame shows the step beginning (the first "
            "motion of that step), and with the end word (for example 'Qn: finish') "
            "when the newest frame shows the step being completed. Report each "
            "boundary once; never repeat a boundary you already replied to. When "
            "unsure, output no."
        ),
        "pnr": (
            "Each question asks for the moment an action happens. Almost every frame "
            "does NOT show that moment. Reply 'Qn: now' when the newest frame shows "
            "the action taking place (the moment the object's state changes). Do not "
            "reply again for an occurrence you already replied to."
        ),
    }

    TICK_QUESTION = {
        "abd": (
            "Does the newest frame show the start or end moment asked about in a "
            "question, which you have not already replied to? If so, reply with that "
            "word, for example 'Qn: start'. If not, output no."
        ),
        "pnr": (
            "For each question, is the newest frame the instant of a new occurrence "
            "that you have not replied to yet? Answer every question with one line: "
            "'Qn: now' or 'Qn: no'."
        ),
    }

    SUMMARY_PROMPT = "Summarize what happened in these frames in one short sentence."

    _REPLY_RE = re.compile(r"^\s*\**\s*(?P<tag>Q\d+)\s*\**\s*[:.)\-]\s*(?P<value>.*)$", re.I)
    _RESPOND_CLAUSE_RE = re.compile(r"([.!?])\s*Respond\b.*$", re.I)

    def __init__(self, args):
        super().__init__(stream_fps=self.STREAM_FPS)
        self.decision_every = max(1, self.STREAM_FPS // self.DECISION_HZ)
        self.args = args
        self.model_path = getattr(args, "model_path", None) or DEFAULT_MODEL_PATH
        self._model_init()
        self._system_cache: Dict[str, Segment] = {}

    def _model_init(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="sdpa",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer
        self.device = next(self.model.parameters()).device

        cfg = self.model.config
        self.num_layers = int(cfg.text_config.num_hidden_layers)
        self.video_token_id = int(cfg.video_token_id)
        self.vision_start_id = int(cfg.vision_start_token_id)
        self.vision_end_id = int(cfg.vision_end_token_id)
        self.merge = int(cfg.vision_config.spatial_merge_size)
        self.rotary = self.model.model.language_model.rotary_emb
        self.mrope_section = list(cfg.text_config.rope_scaling["mrope_section"])

        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.stop_ids = {int(im_end)}
        if self.tokenizer.eos_token_id is not None:
            self.stop_ids.add(int(self.tokenizer.eos_token_id))
        self.no_ids = set()
        for word in ("no", "No", " no", " No", "NO"):
            ids = self.tokenizer(word, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                self.no_ids.add(int(ids[0]))

        self._last_visual = None

        def _hook(_module, _inp, out):
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            self._last_visual = tensor.detach() if torch.is_tensor(tensor) else None

        self.model.model.visual.register_forward_hook(_hook)
        print(f"[CtrlSPOT] Qwen2.5-VL from {self.model_path} on {self.device}")

    def _positions(self, ids: torch.Tensor, grids: List[torch.Tensor]) -> torch.Tensor:
        ids = ids.to(self.device).unsqueeze(0)
        grid = torch.cat(grids, dim=0).to(self.device) if grids else None
        pos, _ = self.model.model.get_rope_index(ids, None, grid, attention_mask=None)
        return pos

    def _rotate(self, keys: List[torch.Tensor], pos: torch.Tensor) -> List[torch.Tensor]:
        cos, sin = self.rotary(keys[0], pos)
        return [_ORIG_APPLY_ROTARY(k, k, cos, sin, self.mrope_section)[1] for k in keys]

    def _forward(
        self,
        prefix: List[Segment],
        chunk_ids: List[int],
        pixel_values: Optional[torch.Tensor] = None,
        grid: Optional[torch.Tensor] = None,
    ):
        chunk = torch.tensor(chunk_ids, dtype=torch.long)
        P = sum(len(s) for s in prefix)
        C = int(chunk.numel())
        ids_all = torch.cat([s.ids for s in prefix] + [chunk])
        grids = [s.grid for s in prefix if s.grid is not None]
        if grid is not None:
            grids.append(grid)
        pos = self._positions(ids_all, grids)

        cache = DynamicCache()
        if P > 0:
            keys = [
                torch.cat([s.keys[l].to(self.device) for s in prefix], dim=2)
                for l in range(self.num_layers)
            ]
            rotated = self._rotate(keys, pos[:, :, :P])
            del keys
            for l in range(self.num_layers):
                values = torch.cat([s.values[l].to(self.device) for s in prefix], dim=2)
                cache.update(rotated[l], values, l)
            del rotated

        inputs = {
            "input_ids": chunk.unsqueeze(0).to(self.device),
            "attention_mask": torch.ones((1, P + C), dtype=torch.long, device=self.device),
            "position_ids": pos[:, :, P:],
            "cache_position": torch.arange(P, P + C, device=self.device),
        }
        if pixel_values is not None:
            inputs["pixel_values_videos"] = pixel_values.to(self.device, self.model.dtype)
            inputs["video_grid_thw"] = grid.to(self.device)

        _CAPTURE["on"], _CAPTURE["keys"] = True, []
        try:
            with torch.no_grad():
                out = self.model(
                    **inputs, past_key_values=cache, use_cache=True, logits_to_keep=1
                )
        finally:
            captured = _CAPTURE["keys"]
            _CAPTURE["on"], _CAPTURE["keys"] = False, []
        if len(captured) != self.num_layers:
            raise RuntimeError(
                f"[CtrlSPOT] captured {len(captured)} key tensors, expected {self.num_layers}"
            )
        return out, out.past_key_values, captured, pos

    def _cut_segment(self, cache, captured, P: int, start: int, length: int, ids) -> Segment:
        keys = [k[:, :, start : start + length].clone() for k in captured]
        values = [
            cache.layers[l].values[:, :, P + start : P + start + length].clone()
            for l in range(self.num_layers)
        ]
        return Segment(ids=torch.tensor(ids, dtype=torch.long), keys=keys, values=values)

    def _greedy(self, out, cache, pos, max_new_tokens: int, stop_on_no: bool) -> str:
        token = int(out.logits[0, -1].argmax())
        generated = [token]
        if stop_on_no and token in self.no_ids:
            return "no"
        cursor = cache.get_seq_length()
        next_pos = int(pos.max()) + 1
        for step in range(max_new_tokens - 1):
            if token in self.stop_ids:
                break
            with torch.no_grad():
                step_out = self.model(
                    input_ids=torch.tensor([[token]], device=self.device),
                    attention_mask=torch.ones((1, cursor + 1), dtype=torch.long, device=self.device),
                    position_ids=torch.full((3, 1, 1), next_pos + step, device=self.device),
                    cache_position=torch.tensor([cursor], device=self.device),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
            cache = step_out.past_key_values
            cursor += 1
            token = int(step_out.logits[0, -1].argmax())
            generated.append(token)
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _text_segment(self, text: str) -> Segment:
        sys_seg = self._system_segment()
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        out, cache, captured, _ = self._forward([sys_seg], ids)
        seg = self._cut_segment(cache, captured, len(sys_seg), 0, len(ids), ids)
        del out, cache
        return seg

    def _system_segment(self) -> Segment:
        task = self._active_task
        if task not in self._system_cache:
            text = (
                f"<|im_start|>system\n{self.SYSTEM_BASE}{self.OUTPUT_RULES[task]}\n"
                f"{self.TASK_RULES[task]}<|im_end|>\n<|im_start|>user\n"
            )
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            out, cache, captured, _ = self._forward([], ids)
            self._system_cache[task] = self._cut_segment(cache, captured, 0, 0, len(ids), ids)
            del out, cache
        return self._system_cache[task]

    def _resize(self, frame: np.ndarray) -> Image.Image:
        h, w = frame.shape[:2]
        scale = min(1.0, self.FRAME_SHORT_SIDE / float(min(h, w)))
        b = self.FRAME_BLOCK
        rh = max(b, int(round(h * scale / b)) * b)
        rw = max(b, int(round(w * scale / b)) * b)
        return Image.fromarray(frame).convert("RGB").resize((rw, rh), Image.BICUBIC)

    def _encode_frame(self, image: Image.Image):
        enc = self.processor.video_processor(videos=[[image, image]], return_tensors="pt")
        grid = enc["video_grid_thw"]
        if int(grid[0, 0]) != 1:
            raise RuntimeError(f"[CtrlSPOT] expected a single temporal patch, got {grid.tolist()}")
        n_tokens = int(grid[0, 1] * grid[0, 2]) // (self.merge * self.merge)
        return enc["pixel_values_videos"], grid, n_tokens

    def _frame_embedding(self) -> torch.Tensor:
        emb = self._last_visual.float().mean(dim=0)
        return F.normalize(emb, dim=0).cpu()

    def _query_embedding(self, text: str) -> torch.Tensor:
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            emb = self.model.get_input_embeddings()(ids.to(self.device)).float().mean(dim=1)[0]
        return F.normalize(emb, dim=0).cpu()

    def _reset_stream(self):
        self.short_term: List[Segment] = []
        self.long_term: List[Segment] = []
        self.summaries: List[Segment] = []
        self.active: List[Query] = []
        self._uid = 0
        self._window: List[Segment] = []

    def _drop_most_similar(self, threshold: Optional[float]) -> bool:
        if len(self.long_term) < 2:
            return False
        embs = torch.stack([s.emb for s in self.long_term])
        sims = (embs[:-1] * embs[1:]).sum(dim=1)
        i = int(sims.argmax())
        if threshold is not None and float(sims[i]) < threshold:
            return False
        del self.long_term[i]
        return True

    def _push_frame(self, seg: Segment):
        self.short_term.append(seg)
        while len(self.short_term) > self.SHORT_TERM_FRAMES - 1:
            self.long_term.append(self.short_term.pop(0).to("cpu"))
        while len(self.long_term) > self.LONG_TERM_FRAMES:
            self._drop_most_similar(threshold=None)

    def _retrieve(self, question: str) -> List[Segment]:
        if not self.long_term:
            return []
        q = self._query_embedding(question)
        sims = torch.stack([s.emb for s in self.long_term]) @ q
        top = torch.topk(sims, k=min(self.RETRIEVE_TOP_K, len(self.long_term))).indices
        picked = sorted((self.long_term[int(i)] for i in top), key=lambda s: s.time)
        return [s.to(self.device) for s in picked]

    def _summarize(self, frames: List[Segment]):
        t0, t1 = frames[0].time, frames[-1].time
        ids = self.tokenizer(
            f"\n{self.SUMMARY_PROMPT}<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )["input_ids"]
        out, cache, _, pos = self._forward([self._system_segment()] + frames, ids)
        text = self._greedy(out, cache, pos, self.SUMMARY_MAX_NEW_TOKENS, stop_on_no=False)
        del out, cache
        text = " ".join(text.split()) or "Nothing notable happened."
        line = f"Summary of {t0:.1f}-{t1:.1f} seconds: {text}\n"
        self.summaries.append(self._text_segment(line))
        del self.summaries[: -self.MAX_SUMMARIES]
        verbose_log(f"[CtrlSPOT] {line.strip()}")

    def _set_answer(self, query: Query, value: str, t: float):
        query.answer = self._text_segment(f"You replied to {query.tag} at {t:.1f} seconds: {value}\n")
        query.last_reply = (value, t)

    def _prefix(self) -> List[Segment]:
        retrieved: Dict[int, Segment] = {}
        for q in self.active:
            for s in q.retrieved:
                retrieved.setdefault(s.uid, s)
        prefix = [self._system_segment()] + list(self.summaries)
        prefix += sorted(retrieved.values(), key=lambda s: s.time)
        prefix += [q.answer for q in self.active if q.answer is not None]
        prefix += self.short_term
        return prefix

    @classmethod
    def _retrieval_text(cls, question: str) -> str:
        return cls._RESPOND_CLAUSE_RE.sub(r"\1", " ".join(question.split())).strip()

    def _query_block(self, t: float) -> str:
        lines = []
        for q in self.active:
            line = f"{q.tag} (asked {t - q.ask_time:.0f} seconds ago): {q.question}"
            if q.last_reply is not None:
                value, when = q.last_reply
                line += f" (you replied '{value}' {t - when:.0f} seconds ago)"
            lines.append(line)
        return (
            "\nActive questions:\n" + "\n".join(lines) + f"\nNewest frame: {t:.1f} seconds. "
            + self.TICK_QUESTION[self._active_task]
            + "<|im_end|>\n<|im_start|>assistant\n"
        )

    def _max_new_tokens(self) -> int:
        return min(self.MAX_NEW_TOKENS, self.TOKENS_PER_QUERY * len(self.active) + 2)

    def _parse_replies(self, text: str) -> List[tuple]:
        tags = {q.tag.upper(): q for q in self.active}
        out = []
        lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
        for line in lines:
            m = self._REPLY_RE.match(line)
            if m and m.group("tag").upper() in tags:
                query, value = tags[m.group("tag").upper()], m.group("value").strip()
            elif len(lines) == 1 and self.active:
                query, value = self.active[-1], (m.group("value") if m else line)
            else:
                continue
            value = value.strip().strip("*").strip()
            if self._is_silent(value):
                continue
            out.append((query, value))
        return out

    @staticmethod
    def _is_silent(text: str) -> bool:
        s = str(text).strip().lower().rstrip(".!")
        return (not s) or (not any(ch.isalnum() for ch in s)) or s in {
            "no", "none", "nothing", "no answer", "no response", "no reply",
        }

    def _schedule(self, turns: List[Dict[str, Any]]):
        by_tick = defaultdict(list)
        n = 0
        for turn in sorted(turns, key=lambda tn: float(tn.get("ask_time", 0.0) or 0.0)):
            question = str(turn.get("question", "") or "").strip()
            if not question:
                continue
            n += 1
            ask = float(turn.get("ask_time", 0.0) or 0.0)
            by_tick[max(0, int(math.floor(ask * self.stream_fps)))].append((f"Q{n}", question, ask))
        return by_tick

    def inference(self, video_path: str, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self._active_task not in self.TASKS:
            raise ValueError(f"[CtrlSPOT] supports {sorted(self.TASKS)}, not {self._active_task!r}")
        if not os.path.exists(video_path):
            print(f"[CtrlSPOT] Missing video: {video_path}")
            return []
        import decord

        try:
            vr = decord.VideoReader(video_path, num_threads=2)
        except Exception as e:
            print(f"[CtrlSPOT] Failed to read video {video_path}: {e}")
            return []
        total = len(vr)
        src_fps = float(vr.get_avg_fps() or 0.0) or float(self.stream_fps)
        duration = total / src_fps
        if total == 0 or duration <= 0:
            return []

        by_tick = self._schedule(turns)
        self._reset_stream()
        events: List[Dict[str, Any]] = []
        num_ticks = max(1, int(math.ceil(duration * self.stream_fps)))
        W = int(round(self.SUMMARY_WINDOW_SECONDS * self.stream_fps))

        for tick in range(num_ticks):
            t = tick / float(self.stream_fps)
            frame = vr[min(total - 1, int(round(t * src_fps)))].asnumpy()

            for tag, question, ask in by_tick.get(tick, []):
                self.active.append(Query(tag, question, ask, self._retrieve(self._retrieval_text(question))))
                events.append({"time": round(ask, 3), "type": "question", "value": question, "turn_id": tag})
            self.active = [q for q in self.active if t - q.ask_time < self.QUERY_TTL_SECONDS]

            pixels, grid, n_tokens = self._encode_frame(self._resize(frame))
            stamp = self.tokenizer(f"<{t:.1f} seconds>", add_special_tokens=False)["input_ids"]
            frame_ids = stamp + [self.vision_start_id] + [self.video_token_id] * n_tokens + [self.vision_end_id]
            on_second = (tick + 1) % self.decision_every == 0
            decide = on_second and bool(self.active)
            t_resp = min(duration, (tick + 1) / float(self.stream_fps))
            tail = (
                self.tokenizer(self._query_block(t), add_special_tokens=False)["input_ids"]
                if decide
                else []
            )

            prefix = self._prefix()
            P = sum(len(s) for s in prefix)
            timer = GenerationTimer(torch).start()
            out, cache, captured, pos = self._forward(prefix, frame_ids + tail, pixels, grid)
            frame_seg = self._cut_segment(cache, captured, P, 0, len(frame_ids), frame_ids)
            frame_seg.grid, frame_seg.time, frame_seg.uid = grid.cpu(), t, self._uid
            frame_seg.emb = self._frame_embedding()
            self._uid += 1

            if decide:
                text = self._greedy(out, cache, pos, self._max_new_tokens(), stop_on_no=True)
                latency = timer.finish()
                del out, cache, captured
                if text != "no":
                    verbose_log(f"[CtrlSPOT] t={t:.1f} -> {text!r}")
                for query, value in self._parse_replies(text):
                    events.append({
                        "time": round(t_resp, 3),
                        "type": "response",
                        "value": value,
                        "turn_id": query.tag,
                        "latency": round(float(latency), 4),
                    })
                    self._set_answer(query, value, t)
            else:
                del out, cache, captured

            self._window.append(frame_seg.to("cpu"))
            self._push_frame(frame_seg)
            window_ends = (tick + 1) % W == 0

            if not self.active:
                if window_ends and len(self._window) == W:
                    self._summarize(self._window)
                if on_second:
                    self._drop_most_similar(threshold=self.COMPRESS_SIMILARITY)
            if window_ends:
                self._window = []

        events.sort(key=lambda e: (e["time"], 0 if e["type"] == "question" else 1))
        return events
