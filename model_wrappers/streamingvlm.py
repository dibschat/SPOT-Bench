import os
import re
import sys
from collections import defaultdict
from typing import Any

import torch

from .base_model import ModelStreaming
from .utils import GenerationTimer, verbose_log

DEFAULT_MODEL_PATH = "mit-han-lab/StreamingVLM"

BASELINE_ROOT = os.environ.get("STREAMING_VLM_ROOT", "")
LIVECC_SRC = (
    os.path.join(BASELINE_ROOT, "streaming_vlm", "livecc_utils", "src")
    if BASELINE_ROOT
    else ""
)

for path in (BASELINE_ROOT, LIVECC_SRC):
    if path and path not in sys.path:
        sys.path.insert(0, path)

try:
    from streaming_vlm.inference import inference as baseline_inf
except ImportError as e:
    raise ImportError(
        "Could not import the StreamingVLM baseline. Clone "
        "https://github.com/mit-han-lab/streaming-vlm and set STREAMING_VLM_ROOT to "
        f"it (currently {BASELINE_ROOT or 'unset'})."
    ) from e


class _StreamingVLMBase(ModelStreaming):
    def __init__(self, args, stream_fps: int | None = None):
        self.args = args

        self.model_path = getattr(args, "model_path", None) or DEFAULT_MODEL_PATH
        self.model_base = "Qwen2_5"

        self.chunk_duration = baseline_inf.DEFAULT_CHUNK_DURATION
        self.window_size = baseline_inf.DEFAULT_WINDOW_SIZE
        self.text_round = baseline_inf.DEFAULT_TEXT_ROUND
        self.text_sink = baseline_inf.DEFAULT_TEXT_SINK
        self.text_sliding_window = (
            baseline_inf.DEFAULT_TEXT_SLIDING_WINDOW
        )
        self.temperature = baseline_inf.DEFAULT_TEMPERATURE
        self.repetition_penalty = (
            baseline_inf.DEFAULT_REPETITION_PENALTY
        )
        self.max_new_tokens = (
            baseline_inf.MAX_TOKEN_PER_DURATION
        )
        self.pos_mode = "shrink"

        fps = int(round(float(baseline_inf.FPS)))
        super().__init__(stream_fps=stream_fps if stream_fps is not None else fps)
        self._model_init()

    def _log(self, msg: str):
        verbose_log(msg)

    def _model_init(self):
        self.model, self.processor = baseline_inf.load_model_and_processor(
            self.model_path, self.model_base
        )
        self.device = self.model.device
        self.stream_fps = int(round(float(baseline_inf.FPS)))
        self._log(
            f"[{self.__class__.__name__}] Loaded baseline pipeline from {self.model_path} "
            f"with stream_fps={self.stream_fps}"
        )

    def _group_questions_by_chunk(
        self, turns: list[dict[str, Any]]
    ) -> dict[int, list[dict[str, Any]]]:
        q_by_chunk: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for turn_idx, turn in enumerate(turns):
            ask_time = float(turn.get("ask_time", 0.0))
            question = str(turn.get("question", "")).strip()
            chunk_idx = max(0, int(ask_time // self.chunk_duration))
            q_by_chunk[chunk_idx].append(
                {
                    "turn_idx": turn_idx,
                    "ask_time": ask_time,
                    "question": question,
                }
            )
        for chunk_idx in q_by_chunk:
            q_by_chunk[chunk_idx].sort(key=lambda x: (x["ask_time"], x["turn_idx"]))
        return q_by_chunk

    def _sanitize_response(self, decoded: str) -> str:
        response = str(decoded).strip()
        if response.endswith(" ..."):
            response = response[:-4].strip()
        return response

    def _assistant_history_text(self, decoded: str, response_text: str) -> str:
        d = str(decoded).strip()
        if d.endswith(" ..."):
            return d
        if response_text:
            return f"{response_text} ..."
        return " ..."

    def _append_question_event(
        self,
        events: list[dict[str, Any]],
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
        events: list[dict[str, Any]],
        t: float,
        value: str,
        turn_id: str | None = None,
        raw_text: str | None = None,
        parsed_label: str | None = None,
        latency: float | None = None,
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
        if parsed_label is not None:
            event["parsed_label"] = parsed_label

        if latency is not None and not self._is_silent_or_no(value):
            event["latency"] = float(round(latency, 4))
        events.append(event)

    @staticmethod
    def _is_silent_or_no(text: str) -> bool:
        s = str(text).strip().lower()
        if not s:
            return True
        if s in {"no", "none", "nothing", "no answer", "no response"}:
            return True
        if not any(ch.isalnum() for ch in s):
            return True
        return False

    def _new_state(self) -> dict[str, Any]:
        return {}

    def _on_chunk_questions(
        self,
        chunk_questions: list[dict[str, Any]],
        events: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> list[str]:
        raise NotImplementedError

    def _extra_prompt_text(self, chunk_idx: int, state: dict[str, Any]) -> str:
        return ""

    def _initial_query_text(
        self,
        chunk_question_texts: list[str],
        state: dict[str, Any],
    ) -> str:
        if chunk_question_texts:
            return "\n".join(chunk_question_texts)
        return ""

    def _handle_chunk_response(
        self,
        response_text: str,
        raw_text: str,
        end_time: float,
        events: list[dict[str, Any]],
        state: dict[str, Any],
        latency: float | None,
    ):
        raise NotImplementedError

    def _generation_overrides(
        self, chunk_idx: int, state: dict[str, Any]
    ) -> dict[str, Any]:
        return {}

    def _get_system_prompt(self, state: dict[str, Any]) -> str:
        _ = state
        return ""

    def _process_past_kv(
        self,
        past_key_values: Any,
        chunk_idx: int,
        full_conversation_history: list[dict[str, Any]],
        prev_generated_ids: torch.Tensor | None,
        assistant_start_bias: int,
        assistant_end_bias: int,
        recent_video_window_clips: list[Any],
        recent_pixel_values_videos: list[Any],
        state: dict[str, Any],
    ) -> tuple[Any, torch.Tensor | None, list[Any], list[Any]]:
        return baseline_inf.process_past_kv(
            past_key_values,
            chunk_idx,
            text_round=self.text_round,
            visual_round=self.window_size,
            full_conversation_history=full_conversation_history,
            prev_generated_ids=prev_generated_ids,
            assistant_start_bias=assistant_start_bias,
            assistant_end_bias=assistant_end_bias,
            recent_video_window_clips=recent_video_window_clips,
            recent_pixel_values_videos=recent_pixel_values_videos,
            text_sink=self.text_sink,
            text_sliding_window=self.text_sliding_window,
        )

    def _previous_text_seed(self, system_prompt: str, state: dict[str, Any]) -> str:
        return ""

    def _append_system_prompt_to_user(self) -> bool:
        return True

    def _post_generation_context(
        self,
        *,
        generated_ids: torch.Tensor,
        current_input_len: int,
        past_key_values: Any,
        response_text: str,
        raw_text: str,
        state: dict[str, Any],
    ) -> tuple[Any, torch.Tensor, str]:
        assistant_history = self._assistant_history_text(raw_text, response_text)
        return past_key_values, generated_ids.clone(), assistant_history

    def inference(self, video_path: str, turns: list) -> list:
        if not os.path.exists(video_path):
            print(f"[{self.__class__.__name__}] Missing video: {video_path}")
            return []

        video_reader, _, _ = baseline_inf.get_smart_resized_video_reader(
            video_path, baseline_inf.MAX_PIXELS
        )
        video_fps = float(video_reader.get_avg_fps())
        if video_fps <= 0:
            return []

        duration = float(len(video_reader) / video_fps)
        num_chunks = int((duration + self.chunk_duration - 1) // self.chunk_duration)
        if num_chunks <= 0:
            return []

        assistant_start_bias = len(
            self.processor(text="<|im_start|>assistant\n")["input_ids"][0]
        )
        assistant_end_bias = len(self.processor(text=" ...<|im_end|>")["input_ids"][0])

        q_by_chunk = self._group_questions_by_chunk(turns)
        events: list[dict[str, Any]] = []
        state = self._new_state()

        past_key_values = None
        full_conversation_history = []
        prev_generated_ids = None
        recent_video_window_clips = []
        recent_pixel_values_videos = []
        streaming_args = baseline_inf.StreamingArgs(
            pos_mode=self.pos_mode, all_text=False
        )

        system_prompt = self._get_system_prompt(state)

        for chunk_idx in range(num_chunks):
            start_time = chunk_idx * self.chunk_duration
            end_time = min(duration, start_time + self.chunk_duration)
            (
                past_key_values,
                prev_generated_ids,
                recent_video_window_clips,
                recent_pixel_values_videos,
            ) = self._process_past_kv(
                past_key_values,
                chunk_idx=chunk_idx,
                full_conversation_history=full_conversation_history,
                prev_generated_ids=prev_generated_ids,
                assistant_start_bias=assistant_start_bias,
                assistant_end_bias=assistant_end_bias,
                recent_video_window_clips=recent_video_window_clips,
                recent_pixel_values_videos=recent_pixel_values_videos,
                state=state,
            )

            try:
                ele = {
                    "type": "video",
                    "video": video_path,
                    "video_start": start_time,
                    "video_end": end_time,
                }
                current_video_chunk, _, _ = baseline_inf._read_video_decord_plus(
                    ele,
                    return_pts=True,
                    strict_fps=True,
                    only_get_last_frame=int(self.chunk_duration * baseline_inf.FPS),
                    vr=video_reader,
                )
                current_video_chunk = baseline_inf._spatial_resize_video(
                    current_video_chunk
                )
            except Exception as e:
                raise RuntimeError(
                    f"[{self.__class__.__name__}] Failed to read the chunk at "
                    f"t={start_time:.1f}s in {video_path}. Refusing to write a "
                    "partial trace; re-run with --resume to continue from the "
                    "last completed video."
                ) from e

            recent_video_window_clips.append(current_video_chunk)
            generation_timer = GenerationTimer(torch).start()

            chunk_questions = q_by_chunk.get(chunk_idx, [])
            chunk_question_texts = self._on_chunk_questions(
                chunk_questions=chunk_questions,
                events=events,
                state=state,
            )

            prompt = f"Time={start_time:.1f}-{end_time:.1f}s"
            user_content = [{"type": "text", "text": prompt}]
            if chunk_idx == 0:
                user_content.append({"type": "video", "video": video_path})
            else:
                user_content.append(
                    {
                        "type": "video",
                        "video": video_path,
                        "start": start_time,
                        "duration": self.chunk_duration,
                    }
                )

            extra_text = self._extra_prompt_text(chunk_idx=chunk_idx, state=state)

            if chunk_idx == 0:
                init_query = self._initial_query_text(
                    chunk_question_texts=chunk_question_texts,
                    state=state,
                )

                if system_prompt and self._append_system_prompt_to_user():
                    user_content.append({"type": "text", "text": system_prompt})
                if extra_text:
                    user_content.append({"type": "text", "text": extra_text})
                if init_query:
                    user_content.append({"type": "text", "text": init_query})

                previous_text = self._previous_text_seed(
                    system_prompt=system_prompt, state=state
                )
                full_conversation_history = [
                    {"role": "previous text", "content": previous_text},
                    {"role": "user", "content": user_content},
                ]
                text = self.processor.apply_chat_template(
                    full_conversation_history,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                if chunk_question_texts:
                    user_content.append(
                        {"type": "text", "text": "\n".join(chunk_question_texts)}
                    )
                if extra_text:
                    user_content.append({"type": "text", "text": extra_text})

                full_conversation_history.append(
                    {"role": "user", "content": user_content}
                )
                text = self.processor.apply_chat_template(
                    [{"role": "user", "content": user_content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                text = "\n" + text[baseline_inf.SYSTEM_PROMPT_OFFSET :]

            inputs = self.processor(
                text=[text],
                videos=recent_video_window_clips[-1],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            if prev_generated_ids is not None:
                if prev_generated_ids[:, -1].item() != baseline_inf.TOKEN_IDS["\n"]:
                    inputs["input_ids"] = torch.cat(
                        [prev_generated_ids, inputs["input_ids"]], dim=1
                    )
                else:
                    inputs["input_ids"] = torch.cat(
                        [prev_generated_ids, inputs["input_ids"][:, 1:]], dim=1
                    )
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

            recent_pixel_values_videos.append(inputs["pixel_values_videos"])
            if streaming_args.pos_mode == "shrink":
                streaming_args.input_ids = inputs["input_ids"]
                if chunk_idx == 0:
                    streaming_args.video_grid_thw = inputs["video_grid_thw"]
                    streaming_args.second_per_grid_ts = (
                        inputs["second_per_grid_ts"]
                        if "second_per_grid_ts" in inputs
                        else None
                    )
                else:
                    streaming_args.video_grid_thw = torch.cat(
                        [streaming_args.video_grid_thw, inputs["video_grid_thw"]], dim=0
                    )
                    if "second_per_grid_ts" in inputs:
                        if streaming_args.second_per_grid_ts is None:
                            streaming_args.second_per_grid_ts = inputs[
                                "second_per_grid_ts"
                            ]
                        else:
                            streaming_args.second_per_grid_ts = torch.cat(
                                [
                                    streaming_args.second_per_grid_ts,
                                    inputs["second_per_grid_ts"],
                                ],
                                dim=0,
                            )

            current_input_len = inputs["input_ids"].shape[1]
            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "use_cache": True,
                "return_dict_in_generate": True,
                "do_sample": False,
                "repetition_penalty": self.repetition_penalty,
                "streaming_args": streaming_args,
                "pad_token_id": 151645,
            }
            gen_kwargs.update(
                self._generation_overrides(chunk_idx=chunk_idx, state=state)
            )
            if gen_kwargs.get("do_sample") is False:
                gen_kwargs.pop("temperature", None)
            else:
                gen_kwargs["temperature"] = self.temperature
            outputs = self.model.generate(
                **inputs,
                past_key_values=past_key_values,
                **gen_kwargs,
            )

            generated_ids = outputs.sequences
            if generated_ids[0, -1].item() != 151645:
                generated_ids = torch.cat(
                    [generated_ids, torch.tensor([[151645]], device=self.device)], dim=1
                )
            newly_generated_ids = generated_ids[:, current_input_len:]

            decoded = self.processor.batch_decode(
                newly_generated_ids, skip_special_tokens=True
            )[0]
            response_text = self._sanitize_response(decoded)
            chunk_latency = generation_timer.finish()

            self._handle_chunk_response(
                response_text=response_text,
                raw_text=decoded,
                end_time=end_time,
                events=events,
                state=state,
                latency=chunk_latency,
            )

            (
                past_key_values,
                prev_generated_ids,
                assistant_history,
            ) = self._post_generation_context(
                generated_ids=generated_ids,
                current_input_len=current_input_len,
                past_key_values=outputs.past_key_values,
                response_text=response_text,
                raw_text=decoded,
                state=state,
            )
            full_conversation_history.append(
                {"role": "assistant", "content": assistant_history}
            )

        events.sort(
            key=lambda x: (x.get("time", 0.0), 0 if x.get("type") == "question" else 1)
        )

        return events


class StreamingVLM(_StreamingVLMBase):
    """
    Multi-turn adaptation:
    - System/task policy pinned in the never-evicted "previous text" sink
    - Queries injected once at ask-time
    - Query lines re-pinned in the sink once their user turn is evicted
    """

    TURN_LINE_RE = re.compile(r"^\s*(Q\d+)\s*(?::|-)?\s*(.+?)\s*$", re.IGNORECASE)
    TURN_QUERY_RE = re.compile(r"^\s*(Q\d+)\s*:?\s*(.+)\s*$", re.IGNORECASE)

    def __init__(self, args, stream_fps: int | None = None):
        super().__init__(args=args, stream_fps=stream_fps)

        self.window_size = 16
        self.text_round = 4
        self.text_sink = 512
        self.text_sliding_window = 512

        # Native StreamingVLM (Qwen2.5-VL default) system prompt.
        self._base_system_prompt = "You are a helpful assistant."

        self._pnr_previous_text = (
            "Output format: Q1 answer\n"
            "If the evidence is insufficient, output no.\n"
            "Do not narrate, do not repeat the question."
        )
        self._abd_previous_text = (
            "Output format: Q1 answer\n"
            "If the evidence is insufficient, output no.\n"
            "Do not narrate, do not repeat the question."
        )
        self._spg_previous_text = (
            "You are guiding the person through the task below, whose steps are listed in "
            "order. As they progress, output one short imperative instruction telling them "
            "the next action to perform."
        )
        self._si_previous_text = (
            "You are helping the person carry out the task below, whose steps are listed "
            "in order. When they hesitate or begin to make a mistake, output one short "
            "imperative instruction that corrects them or tells them the next step to take."
        )
        self._ui_user_prompt = (
            "Describe the video as concise live guidance for a blind person. "
            "Mention hazards, obstacles, traffic, steps, edges, or needed actions "
            "when visible."
        )
        self._default_previous_text = (
            "Output format: Q1 response\n"
            "If the evidence is insufficient, output no.\n"
            "Do not narrate, do not repeat the question."
        )

    def _new_state(self) -> dict[str, Any]:
        return {
            "next_turn_id": 0,
            "turn_questions": {},
        }

    def _new_turn_id(self, state: dict[str, Any]) -> str:
        turn_id = f"Q{state['next_turn_id'] + 1}"
        state["next_turn_id"] += 1
        return turn_id

    _RESPOND_WORDS_RE = re.compile(
        r"([.!?])\s*Respond\b[^:]*:\s*(?P<words>.+?)\s*$", re.IGNORECASE
    )

    @classmethod
    def _answer_cue_question(cls, question: str) -> str:
        match = cls._RESPOND_WORDS_RE.search(question)
        if not match:
            return question

        head = (question[: match.start()] + match.group(1)).strip()
        words: list[str] = []
        for word in re.split(r"\s*(?:,|\band\b|\bor\b|/)\s*", match.group("words")):
            word = word.strip().strip(".,;:").lower()
            if word and word not in words:
                words.append(word)

        if not words:
            return head
        return f"{head} Answer with exactly one token: {' or '.join(words)}"

    def _question_for_prompt(self, question: str) -> str:
        task = (self._active_task or "").lower()
        if task == "abd":
            return self._answer_cue_question(question)
        return question

    def _compose_system_prompt(self) -> str:
        return self._base_system_prompt

    def _get_system_prompt(self, state: dict[str, Any]) -> str:
        _ = state
        return self._compose_system_prompt()

    def _on_chunk_questions(
        self,
        chunk_questions: list[dict[str, Any]],
        events: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> list[str]:
        ds = (self._active_task or "").lower()
        query_lines: list[str] = []
        for item in chunk_questions:
            question = str(item.get("question", "")).strip()
            if not question:
                continue
            if ds == "ui":
                # Pinned in the sink instead: user content is evicted, and StreamingVLM
                # trims generated outputs, so an evicted instruction never returns.
                continue
            if ds in {"si", "spg"}:
                state["manual"] = question
                continue
            ask_time = float(item.get("ask_time", 0.0))

            turn_id = self._new_turn_id(state)
            state["turn_questions"][turn_id] = question
            self._append_question_event(
                events,
                ask_time=ask_time,
                question=question,
                turn_id=turn_id,
            )
            query_lines.append(f"{turn_id} {self._question_for_prompt(question)}")

        return query_lines

    def _append_system_prompt_to_user(self) -> bool:
        return False

    def _previous_text_seed(self, system_prompt: str, state: dict[str, Any]) -> str:
        _ = system_prompt
        ds = (self._active_task or "").lower()
        if ds in {"si", "spg"}:
            instruction = self._si_previous_text if ds == "si" else self._spg_previous_text
            manual = state.get("manual", "")
            return f"{instruction}\n{manual}" if manual else instruction
        return {
            "ui": self._ui_user_prompt,
            "pnr": self._pnr_previous_text,
            "abd": self._abd_previous_text,
        }.get(ds, self._default_previous_text)

    def _normalize_label(self, text: str) -> str | None:
        token = str(text).strip().lower().split()
        if not token:
            return None
        head = token[0]
        if head in {"now", "no", "start", "end", "done", "continuous"}:
            return head
        return None

    @staticmethod
    def _normalize_silence_output(text: str) -> str:
        s = str(text).strip()
        if not s:
            return s

        if not any(ch.isalnum() for ch in s):
            return "no"
        return s

    def _parse_turn_lines(self, response_text: str) -> dict[str, str]:
        turn_hits: dict[str, str] = {}
        for line in response_text.splitlines():
            match = self.TURN_LINE_RE.match(line)
            if not match:
                continue
            turn_id = match.group(1).upper()
            answer = match.group(2).strip()
            if not answer:
                continue
            turn_hits[turn_id] = answer
        return turn_hits

    def _handle_chunk_response(
        self,
        response_text: str,
        raw_text: str,
        end_time: float,
        events: list[dict[str, Any]],
        state: dict[str, Any],
        latency: float | None,
    ):
        _ = state
        response_text = self._normalize_silence_output(response_text)

        turn_hits = self._parse_turn_lines(response_text)
        if turn_hits:
            for turn_id, answer in turn_hits.items():
                answer = self._normalize_silence_output(answer)
                self._append_response_event(
                    events,
                    t=end_time,
                    value=answer,
                    turn_id=turn_id,
                    raw_text=raw_text,
                    parsed_label=self._normalize_label(answer),
                    latency=latency,
                )
            return

        if response_text:
            self._append_response_event(
                events,
                t=end_time,
                value=response_text,
                raw_text=raw_text,
                latency=latency,
            )

    def _generation_overrides(
        self, chunk_idx: int, state: dict[str, Any]
    ) -> dict[str, Any]:

        _ = chunk_idx
        _ = state
        ds = (self._active_task or "").lower()
        if ds in {"pnr", "abd"}:
            return {
                "do_sample": False,
                "max_new_tokens": 8,
            }
        return {
            "do_sample": True,
            "max_new_tokens": 20,
        }

    def _encode_text_tokens(self, text: str) -> list[int]:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer(text, add_special_tokens=False)["input_ids"]
        encoded = self.processor(text=[text])["input_ids"][0]
        return [int(x) for x in encoded]

    @staticmethod
    def _find_subsequence_in_span(
        seq: list[int], subseq: list[int], start_idx: int, end_idx: int
    ) -> tuple[int, int] | None:
        if not subseq:
            return None
        max_start = end_idx - len(subseq) + 1
        if max_start < start_idx:
            return None
        for s in range(start_idx, max_start + 1):
            if seq[s : s + len(subseq)] == subseq:
                return s, s + len(subseq) - 1
        return None

    @staticmethod
    def _resort_id_and_kv(
        input_ids: torch.Tensor,
        past_key_values: Any,
        src_start_idx: int,
        src_end_idx: int,
        dst_idx: int,
    ) -> tuple[torch.Tensor, Any]:

        assert dst_idx < src_start_idx <= src_end_idx
        input_ids = torch.cat(
            [
                input_ids[:, : dst_idx + 1],
                input_ids[:, src_start_idx : src_end_idx + 1],
                input_ids[:, dst_idx + 1 : src_start_idx],
                input_ids[:, src_end_idx + 1 :],
            ],
            dim=1,
        )
        for layer_i, (k_layer, v_layer) in enumerate(past_key_values):
            past_key_values.key_cache[layer_i] = torch.cat(
                [
                    k_layer[:, :, : dst_idx + 1],
                    k_layer[:, :, src_start_idx : src_end_idx + 1],
                    k_layer[:, :, dst_idx + 1 : src_start_idx],
                    k_layer[:, :, src_end_idx + 1 :],
                ],
                dim=2,
            )
            past_key_values.value_cache[layer_i] = torch.cat(
                [
                    v_layer[:, :, : dst_idx + 1],
                    v_layer[:, :, src_start_idx : src_end_idx + 1],
                    v_layer[:, :, dst_idx + 1 : src_start_idx],
                    v_layer[:, :, src_end_idx + 1 :],
                ],
                dim=2,
            )
        return input_ids, past_key_values

    def _extract_query_lines_from_user(self, user_entry: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        for item in user_entry.get("content", []):
            if item.get("type") != "text":
                continue
            text = str(item.get("text", ""))
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if self.TURN_QUERY_RE.match(line):
                    lines.append(line)
        return lines

    @staticmethod
    def _append_to_previous_text(
        full_conversation_history: list[dict[str, Any]], line: str
    ) -> None:
        prev = str(full_conversation_history[0].get("content", ""))
        existing = {x.strip() for x in prev.splitlines() if x.strip()}
        if line in existing:
            return
        full_conversation_history[0]["content"] = (
            f"{prev}\n{line}" if prev and not prev.endswith("\n") else f"{prev}{line}"
        )

    def _find_query_span_in_user(
        self,
        prev_generated_ids: torch.Tensor,
        user_start_idx: int,
        user_end_idx: int,
        query_line: str,
    ) -> tuple[int, int] | None:
        seq = prev_generated_ids.flatten().tolist()
        candidates = [
            query_line,
            "\n" + query_line,
            query_line + "\n",
            "\n" + query_line + "\n",
            " " + query_line,
            "\n " + query_line,
        ]
        for cand in candidates:
            cand_ids = self._encode_text_tokens(cand)
            match = self._find_subsequence_in_span(
                seq, cand_ids, user_start_idx, user_end_idx
            )
            if match is not None:
                return match
        return None

    def _move_retired_queries_to_previous_text(
        self,
        *,
        prev_generated_ids: torch.Tensor,
        past_key_values: Any,
        full_conversation_history: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Any]:
        retired_user = full_conversation_history[-2 * self.text_round]
        query_lines = self._extract_query_lines_from_user(retired_user)
        if not query_lines:
            return prev_generated_ids, past_key_values

        for query_line in query_lines:
            try:
                previous_text_start_idx, previous_text_end_idx = (
                    baseline_inf.get_qwen_range(
                        prev_generated_ids, "previous text", 0, contain_lf=False
                    )
                )
                _ = previous_text_start_idx
                user_start_idx, user_end_idx = baseline_inf.get_qwen_range(
                    prev_generated_ids, "user", -self.text_round
                )
            except Exception:
                continue

            match = self._find_query_span_in_user(
                prev_generated_ids, user_start_idx, user_end_idx, query_line
            )
            if match is None:
                self._log(
                    f"[{self.__class__.__name__}] Could not migrate query span: {query_line!r}"
                )
                continue
            src_start_idx, src_end_idx = match
            if src_start_idx <= previous_text_end_idx - 1:
                continue

            prev_generated_ids, past_key_values = self._resort_id_and_kv(
                prev_generated_ids,
                past_key_values,
                src_start_idx,
                src_end_idx,
                previous_text_end_idx - 1,
            )
            self._append_to_previous_text(full_conversation_history, query_line)

        return prev_generated_ids, past_key_values

    def _process_past_kv(
        self,
        past_key_values: Any,
        chunk_idx: int,
        full_conversation_history: list[dict[str, Any]],
        prev_generated_ids: torch.Tensor | None,
        assistant_start_bias: int,
        assistant_end_bias: int,
        recent_video_window_clips: list[Any],
        recent_pixel_values_videos: list[Any],
        state: dict[str, Any],
    ) -> tuple[Any, torch.Tensor | None, list[Any], list[Any]]:
        _ = state
        if prev_generated_ids is None:
            return (
                past_key_values,
                prev_generated_ids,
                recent_video_window_clips,
                recent_pixel_values_videos,
            )

        if chunk_idx >= self.text_round:
            assert full_conversation_history[0]["role"] == "previous text"
            assert full_conversation_history[-2 * self.text_round]["role"] == "user"
            assert (
                full_conversation_history[-(2 * self.text_round - 1)]["role"]
                == "assistant"
            )
            full_conversation_history[0]["content"] += full_conversation_history[
                -(2 * self.text_round - 1)
            ]["content"][:-4]

            assistant_text_start_idx, assistant_text_end_idx = (
                baseline_inf.get_qwen_range(prev_generated_ids, "assistant", 0)
            )
            _, previous_text_end_idx = baseline_inf.get_qwen_range(
                prev_generated_ids, "previous text", 0, contain_lf=False
            )

            if (
                prev_generated_ids[0, assistant_text_end_idx]
                == baseline_inf.TOKEN_IDS["\n"]
            ):
                src_start_idx = assistant_text_start_idx + assistant_start_bias
                src_end_idx = assistant_text_end_idx - assistant_end_bias - 1
            else:
                src_start_idx = assistant_text_start_idx + assistant_start_bias
                src_end_idx = assistant_text_end_idx - assistant_end_bias
            if src_start_idx <= src_end_idx:
                prev_generated_ids, past_key_values = self._resort_id_and_kv(
                    prev_generated_ids,
                    past_key_values,
                    src_start_idx,
                    src_end_idx,
                    previous_text_end_idx - 1,
                )

            prev_generated_ids, past_key_values = (
                self._move_retired_queries_to_previous_text(
                    prev_generated_ids=prev_generated_ids,
                    past_key_values=past_key_values,
                    full_conversation_history=full_conversation_history,
                )
            )

            for k, item in enumerate(
                full_conversation_history[-2 * self.text_round]["content"]
            ):
                if item["type"] == "text":
                    del full_conversation_history[-2 * self.text_round]["content"][k]
                    break

            del full_conversation_history[-(2 * self.text_round - 1)]
            if self.window_size > self.text_round:
                user_text_start_idx, user_text_end_idx = baseline_inf.get_qwen_range(
                    prev_generated_ids, "user_text", -self.text_round, contain_lf=False
                )
                prev_generated_ids, past_key_values = (
                    baseline_inf.prune_id_and_kv_cache(
                        prev_generated_ids,
                        past_key_values,
                        user_text_start_idx,
                        user_text_end_idx,
                    )
                )
            assistant_text_start_idx, assistant_text_end_idx = (
                baseline_inf.get_qwen_range(
                    prev_generated_ids, "assistant", -self.text_round
                )
            )
            prev_generated_ids, past_key_values = baseline_inf.prune_id_and_kv_cache(
                prev_generated_ids,
                past_key_values,
                assistant_text_start_idx,
                assistant_text_end_idx,
            )

        if chunk_idx >= self.window_size:
            recent_video_window_clips.pop(0)
            recent_pixel_values_videos.pop(0)
            if self.window_size < self.text_round:
                if self.window_size >= self.text_round:
                    full_conversation_history[1]["content"] = [
                        item
                        for item in full_conversation_history[1]["content"]
                        if item["type"] != "video"
                    ]
                else:
                    full_conversation_history[-2 * self.window_size]["content"] = [
                        item
                        for item in full_conversation_history[-2 * self.window_size][
                            "content"
                        ]
                        if item["type"] != "video"
                    ]

                video_token_start_idx, video_token_end_idx = (
                    baseline_inf.get_qwen_range(prev_generated_ids, "vision", 0)
                )
                prev_generated_ids, past_key_values = (
                    baseline_inf.prune_id_and_kv_cache(
                        prev_generated_ids,
                        past_key_values,
                        video_token_start_idx,
                        video_token_end_idx,
                    )
                )

        if chunk_idx >= max(self.window_size, self.text_round):
            del full_conversation_history[1]
            user_start_idx, user_end_idx = baseline_inf.get_qwen_range(
                prev_generated_ids, "user", 0
            )
            prev_generated_ids, past_key_values = baseline_inf.prune_id_and_kv_cache(
                prev_generated_ids,
                past_key_values,
                user_start_idx,
                user_end_idx,
            )

        if chunk_idx > 0:
            if self.text_sink is not None or self.text_sliding_window is not None:
                previous_text_start_idx, previous_text_end_idx = (
                    baseline_inf.get_qwen_range(prev_generated_ids, "previous text", 0)
                )
                cut_start_idx = (
                    previous_text_start_idx + self.text_sink + 4
                    if self.text_sink is not None
                    else previous_text_start_idx
                )
                cut_end_idx = (
                    previous_text_end_idx - self.text_sliding_window - 1
                    if self.text_sliding_window is not None
                    else previous_text_end_idx
                )
                if cut_start_idx <= cut_end_idx:
                    prev_generated_ids, past_key_values = (
                        baseline_inf.prune_id_and_kv_cache(
                            prev_generated_ids,
                            past_key_values,
                            cut_start_idx,
                            cut_end_idx,
                        )
                    )
            prev_generated_ids, past_key_values = baseline_inf.contiguous_id_and_kv(
                prev_generated_ids, past_key_values
            )

        return (
            past_key_values,
            prev_generated_ids,
            recent_video_window_clips,
            recent_pixel_values_videos,
        )

    def _post_generation_context(
        self,
        *,
        generated_ids: torch.Tensor,
        current_input_len: int,
        past_key_values: Any,
        response_text: str,
        raw_text: str,
        state: dict[str, Any],
    ) -> tuple[Any, torch.Tensor, str]:
        """
        Do not retain generated narration text in rolling context.
        Keep only assistant structural close token (<|im_end|>) so turn formatting remains valid.
        """
        if past_key_values is None:
            return past_key_values, generated_ids.clone(), ""

        seq_end = generated_ids.shape[1] - 1

        if seq_end - 1 >= current_input_len:
            trimmed_ids, trimmed_kv = baseline_inf.prune_id_and_kv_cache(
                generated_ids,
                past_key_values,
                current_input_len,
                seq_end - 1,
            )
            return trimmed_kv, trimmed_ids.clone(), ""

        return past_key_values, generated_ids.clone(), ""
