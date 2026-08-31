import importlib.util
import math
import os
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch
import decord
from PIL import Image

from .base_model import ModelStreaming
from .utils import GenerationTimer, verbose_log


DEFAULT_MODEL_PATH = "wangyueqian/MMDuet2"

BASELINE_ROOT = os.environ.get("MMDUET2_ROOT", "")
BASELINE_DEMO = os.path.join(BASELINE_ROOT, "demo") if BASELINE_ROOT else ""
BASELINE_PROACTIVE_EVAL = (
    os.path.join(BASELINE_ROOT, "proactive_eval") if BASELINE_ROOT else ""
)

for path in (BASELINE_DEMO, BASELINE_PROACTIVE_EVAL):
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _load_baseline_demo_module():
    if not BASELINE_ROOT:
        raise RuntimeError(
            "MMDUET2_ROOT is unset. Clone "
            "https://github.com/yellow-binary-tree/MMDuet2 and set MMDUET2_ROOT to it."
        )

    module_name = "mmduet2_demo_inference"
    module_path = os.path.join(BASELINE_DEMO, "inference.py")
    if not os.path.exists(module_path):
        raise FileNotFoundError(
            f"Missing MMDuet2 demo module: {module_path}. MMDUET2_ROOT must point at a "
            "clone of https://github.com/yellow-binary-tree/MMDuet2."
        )

    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_baseline_demo = _load_baseline_demo_module()
ProactiveInferenceClient = _baseline_demo.ProactiveInferenceClient
process_vision_info = _baseline_demo.process_vision_info

try:
    import qwen2_5_vl.modeling_qwen2_5_vl_DTD as qwen2_5_modeling
except ImportError as e:
    raise ImportError(
        "Could not import MMDuet2's vendored Qwen2.5-VL modeling module from "
        f"{BASELINE_PROACTIVE_EVAL or '<MMDUET2_ROOT unset>'}. MMDUET2_ROOT must point "
        "at a full clone of https://github.com/yellow-binary-tree/MMDuet2."
    ) from e

try:
    from transformers.cache_utils import DynamicCache
except Exception:
    DynamicCache = None


@dataclass
class _PreRopeSegment:

    tag: str
    kind: str
    role: str
    content: Any
    token_ids: torch.Tensor
    pre_rope_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    vision_meta: dict[str, Any] = field(default_factory=dict)
    image_grid_thw: torch.Tensor | None = None
    video_grid_thw: torch.Tensor | None = None
    second_per_grid_ts: torch.Tensor | None = None


class MMDuet2(ModelStreaming):
    RESPONSE_LINE_RE = re.compile(r"^\s*(Q\d+)\s*(?::|-)?\s*(.+?)\s*$", re.IGNORECASE)

    def __init__(self, args, stream_fps: int = 2):
        super().__init__(stream_fps=stream_fps)
        self.args = args
        self.decision_fps = int(getattr(args, "decision_fps", 1) or 1)
        if self.stream_fps != 2 or self.decision_fps != 1:
            raise ValueError("MMDuet2 latency protocol requires input=2 FPS, output=1 Hz")
        self._configure_inductor_for_stability()

        self.model_path = getattr(args, "model_path", None) or DEFAULT_MODEL_PATH
        self.attn_implementation = "flash_attention_2"
        self.do_sample = False
        self.temperature = 1.0
        self.top_k = 40
        self.max_new_tokens = 8

        self.kv_mode = str(getattr(args, "kv_mode", "kvflush")).strip()
        print(f"[MMDuet2] init kv_mode={self.kv_mode}")

        self.visual_round = 16
        self.text_sink = 512
        self.text_sliding_window = 512
        self.text_round = 4

        # Native MMDuet2 (ProactiveVideoQA) system prompt
        self._base_system_prompt = (
            "You are a helpful assistant. Your task is to answer questions based on "
            "continuously incoming video frames. Your responses should include "
            "information from the video since your last reply (if any). If the "
            "information in this segment of the video cannot answer the question, "
            'output "NO REPLY".'
        )

        self._spg_user_prompt = (
            "You are guiding the person through the task below, whose steps are listed in "
            "order. As they progress, output one short imperative instruction telling them "
            "the next action to perform."
        )
        self._si_user_prompt = (
            "You are helping the person carry out the task below, whose steps are listed "
            "in order. When they hesitate or begin to make a mistake, output one short "
            "imperative instruction that corrects them or tells them the next step to take."
        )
        self._ui_user_prompt = (
            "Describe the video as concise live guidance for a blind person. "
            "Mention hazards, obstacles, traffic, steps, edges, or needed actions "
            "when visible."
        )

        self._model_init()
        self._init_runtime_state()

    _OPEN_ENDED_SAMPLING = True
    _OPEN_ENDED_TASKS = {"sqa", "spg", "si", "ui"}
    _OPEN_ENDED_TEMPERATURE = 1.0  # MMDuet2 default
    _OPEN_ENDED_MAX_NEW_TOKENS = 512  # MMDuet2 default

    def _is_open_ended_sampling_task(self) -> bool:
        if not self._OPEN_ENDED_SAMPLING:
            return False
        return (self._active_task or "").lower() in self._OPEN_ENDED_TASKS

    def _do_sample_for_active_task(self) -> bool:
        if self._is_open_ended_sampling_task():
            return True
        return bool(self.do_sample)

    def _temperature_for_active_task(self) -> float:
        if self._is_open_ended_sampling_task():
            return float(self._OPEN_ENDED_TEMPERATURE)
        return float(self.temperature)

    def _max_new_tokens_for_active_task(self) -> int:
        ds = (self._active_task or "").lower()
        if ds in {"pnr", "abd"}:
            return int(self.max_new_tokens)
        if self._is_open_ended_sampling_task():
            return int(self._OPEN_ENDED_MAX_NEW_TOKENS)
        return 128

    @staticmethod
    def _configure_inductor_for_stability():
        os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
        os.environ.setdefault("TORCHINDUCTOR_WORKER_START", "subprocess")
        try:
            import torch._inductor.config as inductor_config

            inductor_config.compile_threads = 1
        except Exception:
            pass

    def _init_runtime_state(self):
        self._prev_generated_ids: torch.Tensor | None = None
        self._system_sink_len: int | None = None

        self._memory_text = ""
        self._memory_turn_tag: str | None = None

        self._stream_query_ledger: list[str] = []
        self._stream_recent_turns: deque[list[dict[str, Any]]] = deque(
            maxlen=max(1, int(self.visual_round))
        )

        self._frame_turn_tags: deque[str] = deque()
        self._query_turn_tags: deque[str] = deque()

        self._next_frame_tag = 0
        self._next_query_tag = 0
        self._next_memory_tag = 0

        self._svlm_system_segment: _PreRopeSegment | None = None
        self._svlm_memory_segment: _PreRopeSegment | None = None
        self._svlm_query_segments: deque[_PreRopeSegment] = deque()
        self._svlm_frame_segments: deque[_PreRopeSegment] = deque()
        self._svlm_last_evicted_tags: list[str] = []
        self._svlm_last_guard: dict[str, Any] = {}
        self._svlm_last_prefix_position_max: float = -1.0
        self._svlm_capture_enabled = False
        self._svlm_captured_pre_rope_keys: list[torch.Tensor] | None = None

    def _new_frame_tag(self) -> str:
        self._next_frame_tag += 1
        return f"F:{self._next_frame_tag:07d}"

    def _new_query_tag(self) -> str:
        self._next_query_tag += 1
        return f"Q:{self._next_query_tag:07d}"

    def _new_memory_tag(self) -> str:
        self._next_memory_tag += 1
        return f"M:{self._next_memory_tag:07d}"

    def _log(self, msg: str):
        verbose_log(msg)

    def _build_client_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            llm_pretrained=self.model_path,
            attn_implementation=self.attn_implementation,
            system_prompt=self._base_system_prompt,
            input_assistant_turns=False,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k=self.top_k,
        )

    @staticmethod
    def _is_hf_repo_id(path: str) -> bool:
        return (
            not os.path.isabs(path)
            and not path.startswith((".", "~"))
            and path.count("/") == 1
        )

    def _model_init(self):
        if not self._is_hf_repo_id(self.model_path) and not os.path.exists(
            self.model_path
        ):
            self._log(
                f"[MMDuet2] Checkpoint not found at {self.model_path}. "
                "Model init may fail."
            )

        self.client = ProactiveInferenceClient(args=self._build_client_args())
        self.client._last_input_len = 0
        self.client._last_generated_len = 0
        self.client._last_sequences = None
        self._wrap_generate_for_token_stats()
        self._init_token_ids()
        self._init_prerope_runtime_hooks()
        self._sanitize_generation_config_for_decode()

    def _sanitize_generation_config_for_decode(self):
        if bool(self.do_sample):
            return
        gen_cfg = getattr(self.client.model, "generation_config", None)
        if gen_cfg is None:
            return

        if hasattr(gen_cfg, "do_sample"):
            gen_cfg.do_sample = False
        if hasattr(gen_cfg, "temperature"):
            gen_cfg.temperature = None

    def _wrap_generate_for_token_stats(self):
        model = self.client.model
        if getattr(model, "_mmduet2_generate_wrapped", False):
            return

        original_generate = model.generate

        def wrapped_generate(*args, **kwargs):
            kwargs["max_new_tokens"] = int(self._max_new_tokens_for_active_task())
            kwargs["do_sample"] = bool(self._do_sample_for_active_task())
            if not kwargs["do_sample"]:
                kwargs.pop("temperature", None)
            else:
                kwargs["temperature"] = float(self._temperature_for_active_task())
            input_ids = kwargs.get("input_ids", None)
            if input_ids is None and args:
                first = args[0]
                if torch.is_tensor(first):
                    input_ids = first

            input_len = int(input_ids.shape[1]) if torch.is_tensor(input_ids) else 0
            output = original_generate(*args, **kwargs)

            generated_len = 0
            sequences = getattr(output, "sequences", None)
            if torch.is_tensor(sequences) and sequences.ndim >= 2:
                generated_len = max(0, int(sequences.shape[1]) - input_len)

            self.client._last_input_len = input_len
            self.client._last_generated_len = generated_len
            self.client._last_sequences = sequences
            return output

        model.generate = wrapped_generate
        model._mmduet2_generate_wrapped = True

    def _init_token_ids(self):
        tok = getattr(self.client.processor, "tokenizer", None)
        self._tok_ids: dict[str, int] = {
            "im_start": 151644,
            "im_end": 151645,
            "user": 872,
            "assistant": 77091,
            "vision_start": 151652,
            "vision_end": 151653,
            "newline": 198,
        }
        if tok is None:
            return

        def single_id(s: str, key: str):
            try:
                ids = tok(s, add_special_tokens=False)["input_ids"]
                if ids:
                    self._tok_ids[key] = int(ids[0])
            except Exception:
                pass

        single_id("<|im_start|>", "im_start")
        single_id("<|im_end|>", "im_end")
        single_id("user", "user")
        single_id("assistant", "assistant")
        single_id("<|vision_start|>", "vision_start")
        single_id("<|vision_end|>", "vision_end")
        single_id("\n", "newline")

    def _init_prerope_runtime_hooks(self):
        self._svlm_apply_rotary_orig = getattr(
            qwen2_5_modeling, "apply_multimodal_rotary_pos_emb", None
        )
        if self._svlm_apply_rotary_orig is None:
            raise RuntimeError(
                "Missing apply_multimodal_rotary_pos_emb in qwen2_5_vl modeling module."
            )
        self._svlm_dynamic_cache_cls = DynamicCache
        self._install_prerope_capture_hook()
        self._install_prerope_position_ids_bridge()
        self._install_generation_cache_position_compat_hook()
        tok = getattr(self.client.processor, "tokenizer", None)
        if tok is None:
            raise RuntimeError("Tokenizer is required for streamingvlm pre-rope mode.")
        prompt_ids = tok("<|im_start|>assistant\n", add_special_tokens=False)[
            "input_ids"
        ]
        if not prompt_ids:
            raise RuntimeError("Failed to tokenize assistant prompt.")
        self._svlm_assistant_prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)

    def _install_prerope_capture_hook(self):
        if getattr(self.client.model, "_mmduet2_prerope_hooked", False):
            return

        original_apply = self._svlm_apply_rotary_orig

        def wrapped_apply_multimodal_rotary_pos_emb(
            q: torch.Tensor,
            k: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            mrope_section: list[int],
            unsqueeze_dim: int = 1,
        ):
            if self._svlm_capture_enabled:
                if self._svlm_captured_pre_rope_keys is None:
                    self._svlm_captured_pre_rope_keys = []
                self._svlm_captured_pre_rope_keys.append(k.detach().clone())
            return original_apply(
                q,
                k,
                cos,
                sin,
                mrope_section,
                unsqueeze_dim=unsqueeze_dim,
            )

        qwen2_5_modeling.apply_multimodal_rotary_pos_emb = (
            wrapped_apply_multimodal_rotary_pos_emb
        )
        self.client.model._mmduet2_prerope_hooked = True

    def _install_prerope_position_ids_bridge(self):
        model = self.client.model
        model_core = getattr(model, "model", None)
        if model_core is None:
            raise RuntimeError(
                "[MMDuet2][streamingvlm-prerope] missing model core for position_ids bridge."
            )
        if getattr(model_core, "_mmduet2_prerope_pos_bridge", False):
            return

        original_forward = model_core.forward

        def wrapped_forward(*args, **kwargs):
            pos = kwargs.get("position_ids", None)
            if (
                torch.is_tensor(pos)
                and pos.ndim == 3
                and int(pos.shape[0]) == 1
                and int(pos.shape[1]) == 3
            ):

                kwargs["position_ids"] = pos.transpose(0, 1).contiguous()
            past = kwargs.get("past_key_values", None)
            if past is not None:
                self._normalize_dynamic_cache_for_flash_attn(past)
            return original_forward(*args, **kwargs)

        model_core.forward = wrapped_forward
        model_core._mmduet2_prerope_pos_bridge = True

    def _normalize_dynamic_cache_for_flash_attn(self, cache: Any):
        key_cache = getattr(cache, "key_cache", None)
        value_cache = getattr(cache, "value_cache", None)
        if not isinstance(key_cache, list) or not isinstance(value_cache, list):
            return
        model_core = getattr(self.client.model, "model", None)
        layers = getattr(model_core, "layers", None)
        expected_heads = 0
        if isinstance(layers, (list, tuple)) and layers:
            first_attn = getattr(layers[0], "self_attn", None)
            expected_heads = int(getattr(first_attn, "num_key_value_heads", 0) or 0)

        max_layers = min(len(key_cache), len(value_cache))
        for layer_idx in range(max_layers):
            k = key_cache[layer_idx]
            v = value_cache[layer_idx]
            if not torch.is_tensor(k) or not torch.is_tensor(v):
                continue
            if int(k.numel()) <= 0 or int(v.numel()) <= 0:
                continue
            if k.ndim != 4 or v.ndim != 4:
                raise RuntimeError(
                    f"[MMDuet2][streamingvlm-prerope] invalid dynamic cache rank at layer={layer_idx}: "
                    f"k.shape={tuple(k.shape)} v.shape={tuple(v.shape)}"
                )

            if expected_heads > 0:
                if (
                    int(k.shape[1]) == expected_heads
                    and int(v.shape[1]) == expected_heads
                ):
                    k_norm, v_norm = k, v
                elif (
                    int(k.shape[2]) == expected_heads
                    and int(v.shape[2]) == expected_heads
                ):
                    k_norm = k.transpose(1, 2).contiguous()
                    v_norm = v.transpose(1, 2).contiguous()
                else:
                    raise RuntimeError(
                        f"[MMDuet2][streamingvlm-prerope] dynamic cache head-axis mismatch at layer={layer_idx}: "
                        f"expected_heads={expected_heads} k.shape={tuple(k.shape)} v.shape={tuple(v.shape)}"
                    )
            else:

                if int(k.shape[1]) <= int(k.shape[2]) and int(v.shape[1]) <= int(
                    v.shape[2]
                ):
                    k_norm, v_norm = k, v
                else:
                    k_norm = k.transpose(1, 2).contiguous()
                    v_norm = v.transpose(1, 2).contiguous()

            if tuple(k_norm.shape[:3]) != tuple(v_norm.shape[:3]):
                raise RuntimeError(
                    f"[MMDuet2][streamingvlm-prerope] dynamic cache K/V mismatch at layer={layer_idx}: "
                    f"k.shape={tuple(k_norm.shape)} v.shape={tuple(v_norm.shape)}"
                )

            key_cache[layer_idx] = k_norm
            value_cache[layer_idx] = v_norm

    def _install_generation_cache_position_compat_hook(self):
        model = self.client.model
        if getattr(model, "_mmduet2_cache_position_compat_hooked", False):
            return

        original_get_initial_cache_position = getattr(
            model, "_get_initial_cache_position", None
        )
        if not callable(original_get_initial_cache_position):
            raise RuntimeError(
                "[MMDuet2][streamingvlm-prerope] missing model._get_initial_cache_position for generation compatibility patch."
            )

        def _normalize_cache_position(
            cache_position: Any, *, device: torch.device
        ) -> torch.Tensor | None:
            if cache_position is None:
                return None
            if isinstance(cache_position, bool):
                return None
            if torch.is_tensor(cache_position):
                if int(cache_position.numel()) <= 0:
                    return None
                return (
                    cache_position.to(device=device, dtype=torch.long)
                    .reshape(-1)
                    .contiguous()
                )
            if isinstance(cache_position, (list, tuple)):
                if len(cache_position) <= 0:
                    return None
                try:
                    return torch.as_tensor(
                        cache_position, device=device, dtype=torch.long
                    ).reshape(-1)
                except Exception:
                    return None
            if isinstance(cache_position, int):
                return torch.tensor([cache_position], device=device, dtype=torch.long)
            return None

        def wrapped_get_initial_cache_position(
            seq_length: int, device: torch.device, model_kwargs: dict[str, Any]
        ):
            if isinstance(model_kwargs, dict) and "cache_position" in model_kwargs:
                normalized = _normalize_cache_position(
                    model_kwargs.get("cache_position"), device=device
                )
                if normalized is not None and int(normalized.numel()) > 0:
                    model_kwargs["cache_position"] = normalized
                    return model_kwargs
                model_kwargs.pop("cache_position", None)
            return original_get_initial_cache_position(seq_length, device, model_kwargs)

        model._get_initial_cache_position = wrapped_get_initial_cache_position
        model._mmduet2_cache_position_compat_hooked = True

    def _begin_prerope_capture(self):
        self._svlm_captured_pre_rope_keys = []
        self._svlm_capture_enabled = True

    def _end_prerope_capture(self) -> list[torch.Tensor]:
        self._svlm_capture_enabled = False
        captured = self._svlm_captured_pre_rope_keys or []
        self._svlm_captured_pre_rope_keys = None
        return captured

    def _iter_kv_layers(self, past_key_values: Any):
        if past_key_values is None:
            return

        if hasattr(past_key_values, "to_legacy_cache"):
            try:
                legacy = past_key_values.to_legacy_cache()
                if isinstance(legacy, (tuple, list)):
                    for layer in legacy:
                        if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                            yield layer[0], layer[1]
                    return
            except Exception:
                pass

        layers = getattr(past_key_values, "layers", None)
        if isinstance(layers, (list, tuple)) and layers:
            for layer in layers:
                k = getattr(layer, "keys", None)
                v = getattr(layer, "values", None)
                if torch.is_tensor(k) and torch.is_tensor(v):
                    yield k, v
            return

        key_cache = getattr(past_key_values, "key_cache", None)
        value_cache = getattr(past_key_values, "value_cache", None)
        if isinstance(key_cache, (list, tuple)) and isinstance(
            value_cache, (list, tuple)
        ):
            for i in range(min(len(key_cache), len(value_cache))):
                k = key_cache[i]
                v = value_cache[i]
                if torch.is_tensor(k) and torch.is_tensor(v):
                    yield k, v
            return

        if isinstance(past_key_values, (tuple, list)):
            for layer in past_key_values:
                if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                    k, v = layer[0], layer[1]
                    if torch.is_tensor(k) and torch.is_tensor(v):
                        yield k, v

    def _to_legacy_kv(
        self, past_key_values: Any
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        legacy_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        for k, v in self._iter_kv_layers(past_key_values):
            legacy_layers.append((k.detach().contiguous(), v.detach().contiguous()))
        return tuple(legacy_layers)

    @staticmethod
    def _slice_legacy_kv(
        legacy_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        start_index: int,
        end_index: int,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        if not legacy_kv:
            return tuple()
        start_index = int(max(0, start_index))
        end_index = int(max(start_index, end_index))
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for k, v in legacy_kv:
            out.append(
                (
                    k[:, :, start_index:end_index, :].contiguous(),
                    v[:, :, start_index:end_index, :].contiguous(),
                )
            )
        return tuple(out)

    @staticmethod
    def _cat_kv_list(
        kv_list: list[tuple[tuple[torch.Tensor, torch.Tensor], ...]],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        if not kv_list:
            return tuple()
        num_layers = len(kv_list[0])
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            k_parts = [kv[layer_idx][0] for kv in kv_list]
            v_parts = [kv[layer_idx][1] for kv in kv_list]
            if len(k_parts) == 1:
                out.append((k_parts[0].contiguous(), v_parts[0].contiguous()))
            else:
                out.append(
                    (
                        torch.cat(k_parts, dim=2).contiguous(),
                        torch.cat(v_parts, dim=2).contiguous(),
                    )
                )
        return tuple(out)

    def _prepare_past_for_model(
        self, legacy_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    ):
        if not legacy_kv:
            return None
        legacy_kv = self._normalize_legacy_kv_for_flash_attn(legacy_kv)
        if self._svlm_dynamic_cache_cls is None:
            raise RuntimeError(
                "DynamicCache is unavailable; cannot materialize past_key_values from assembled legacy KV."
            )
        ctor = getattr(self._svlm_dynamic_cache_cls, "from_legacy_cache", None)
        if ctor is None:
            raise RuntimeError("DynamicCache.from_legacy_cache is unavailable.")
        try:
            return ctor(legacy_kv)
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert assembled legacy KV to DynamicCache: {e}"
            )

    def _normalize_legacy_kv_for_flash_attn(
        self, legacy_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        model_core = getattr(self.client.model, "model", None)
        model_cfg = getattr(model_core, "config", None)
        num_kv_heads = int(getattr(model_cfg, "num_key_value_heads", 0) or 0)

        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        expected_seq_len: int | None = None
        for layer_idx, (k, v) in enumerate(legacy_kv):
            if not torch.is_tensor(k) or not torch.is_tensor(v):
                raise RuntimeError(
                    f"[MMDuet2][streamingvlm-prerope] non-tensor KV at layer={layer_idx}"
                )
            if k.ndim != 4 or v.ndim != 4:
                raise RuntimeError(
                    f"[MMDuet2][streamingvlm-prerope] KV rank mismatch at layer={layer_idx}: "
                    f"k.shape={tuple(k.shape)} v.shape={tuple(v.shape)}"
                )

            if num_kv_heads > 0:
                if int(k.shape[1]) == num_kv_heads and int(v.shape[1]) == num_kv_heads:
                    k_norm, v_norm = k, v
                elif (
                    int(k.shape[2]) == num_kv_heads and int(v.shape[2]) == num_kv_heads
                ):
                    k_norm = k.transpose(1, 2).contiguous()
                    v_norm = v.transpose(1, 2).contiguous()
                else:
                    raise RuntimeError(
                        f"[MMDuet2][streamingvlm-prerope] KV head-axis mismatch at layer={layer_idx}: "
                        f"num_kv_heads={num_kv_heads} k.shape={tuple(k.shape)} v.shape={tuple(v.shape)}"
                    )
            else:

                if int(k.shape[1]) <= int(k.shape[2]) and int(v.shape[1]) <= int(
                    v.shape[2]
                ):
                    k_norm, v_norm = k, v
                else:
                    k_norm = k.transpose(1, 2).contiguous()
                    v_norm = v.transpose(1, 2).contiguous()

            if tuple(k_norm.shape[:3]) != tuple(v_norm.shape[:3]):
                raise RuntimeError(
                    f"[MMDuet2][streamingvlm-prerope] KV shape mismatch at layer={layer_idx}: "
                    f"k.shape={tuple(k_norm.shape)} v.shape={tuple(v_norm.shape)}"
                )

            seq_len = int(k_norm.shape[2])
            if expected_seq_len is None:
                expected_seq_len = seq_len
            elif seq_len != expected_seq_len:
                raise RuntimeError(
                    "[MMDuet2][streamingvlm-prerope] inconsistent KV seq length across layers: "
                    f"layer={layer_idx} seq_len={seq_len} expected={expected_seq_len}"
                )

            out.append((k_norm.contiguous(), v_norm.contiguous()))

        return tuple(out)

    @staticmethod
    def _tensorize_to_device(
        inputs: dict[str, Any], device: torch.device
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                out[key] = value.to(device)
            else:
                out[key] = value
        return out

    @staticmethod
    def _build_position_ids(
        *, start: int, length: int, device: torch.device
    ) -> torch.Tensor:
        pos = torch.arange(start, start + length, device=device, dtype=torch.long)
        pos = pos.view(1, -1)
        return pos.unsqueeze(0).expand(3, -1, -1).contiguous()

    @staticmethod
    def _svlm_to_lm_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(position_ids) or position_ids.ndim != 3:
            raise RuntimeError(
                "[MMDuet2][streamingvlm-prerope] invalid position_ids for lm forward."
            )
        if int(position_ids.shape[0]) != 3:
            raise RuntimeError(
                f"[MMDuet2][streamingvlm-prerope] expected position_ids[0]=3, got {tuple(position_ids.shape)}"
            )
        if int(position_ids.shape[1]) != 1:
            raise RuntimeError(
                f"[MMDuet2][streamingvlm-prerope] expected batch=1 position_ids, got {tuple(position_ids.shape)}"
            )

        return position_ids.transpose(0, 1).contiguous()

    @staticmethod
    def _build_attention_mask(
        *, prefix_len: int, input_len: int, device: torch.device
    ) -> torch.Tensor:
        total = int(max(0, prefix_len) + max(0, input_len))
        return torch.ones((1, total), dtype=torch.long, device=device)

    @staticmethod
    def _svlm_build_decode_cache_position(
        *, prefix_len: int, prompt_len: int, device: torch.device
    ) -> torch.Tensor:
        if int(prompt_len) <= 0:
            raise RuntimeError(
                f"[MMDuet2][streamingvlm-prerope] invalid prompt length for decode cache_position: prompt_len={prompt_len}"
            )
        start = int(max(0, prefix_len))
        cache_position = torch.arange(
            start, start + int(prompt_len), device=device, dtype=torch.long
        )
        if cache_position.ndim != 1 or int(cache_position.numel()) != int(prompt_len):
            raise RuntimeError(
                "[MMDuet2][streamingvlm-prerope] decode cache_position shape mismatch."
            )
        return cache_position.contiguous()

    @staticmethod
    def _svlm_optional_cpu_tensor(
        value: Any, *, dtype: torch.dtype | None = None
    ) -> torch.Tensor | None:
        if value is None:
            return None
        if not torch.is_tensor(value):
            raise RuntimeError(
                "[MMDuet2][streamingvlm-prerope] expected tensor metadata for multimodal segment."
            )
        if value.numel() <= 0:
            return None
        out = value.detach().to("cpu")
        if dtype is not None:
            out = out.to(dtype=dtype)
        return out.contiguous()

    def _svlm_build_multimodal_ledger(
        self,
        *,
        segments: list[_PreRopeSegment],
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if not segments:
            raise RuntimeError(
                "[MMDuet2][streamingvlm-prerope] cannot build multimodal ledger from empty segments."
            )

        token_parts: list[torch.Tensor] = []
        image_parts: list[torch.Tensor] = []
        video_parts: list[torch.Tensor] = []
        second_parts: list[torch.Tensor] = []

        for seg in segments:
            tok = seg.token_ids
            if not torch.is_tensor(tok) or tok.ndim != 1 or int(tok.numel()) <= 0:
                raise RuntimeError(
                    f"[MMDuet2][streamingvlm-prerope] invalid token_ids for segment tag={seg.tag}"
                )
            token_parts.append(tok.to(device=device, dtype=torch.long).view(1, -1))

            if seg.image_grid_thw is not None:
                img = seg.image_grid_thw
                if not torch.is_tensor(img) or img.ndim != 2 or int(img.shape[-1]) != 3:
                    raise RuntimeError(
                        f"[MMDuet2][streamingvlm-prerope] invalid image_grid_thw for segment tag={seg.tag}"
                    )
                image_parts.append(img.to(device=device, dtype=torch.long))

            if seg.video_grid_thw is not None:
                vid = seg.video_grid_thw
                if not torch.is_tensor(vid) or vid.ndim != 2 or int(vid.shape[-1]) != 3:
                    raise RuntimeError(
                        f"[MMDuet2][streamingvlm-prerope] invalid video_grid_thw for segment tag={seg.tag}"
                    )
                video_parts.append(vid.to(device=device, dtype=torch.long))

            if seg.second_per_grid_ts is not None:
                sec = seg.second_per_grid_ts
                if not torch.is_tensor(sec):
                    raise RuntimeError(
                        f"[MMDuet2][streamingvlm-prerope] invalid second_per_grid_ts for segment tag={seg.tag}"
                    )
                second_parts.append(sec.to(device=device, dtype=torch.float32).view(-1))

        input_ids = torch.cat(token_parts, dim=1).contiguous()
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        image_grid_thw = (
            torch.cat(image_parts, dim=0).contiguous() if image_parts else None
        )
        video_grid_thw = (
            torch.cat(video_parts, dim=0).contiguous() if video_parts else None
        )
        second_per_grid_ts = (
            torch.cat(second_parts, dim=0).contiguous() if second_parts else None
        )

        if second_per_grid_ts is not None and video_grid_thw is None:
            raise RuntimeError(
                "[MMDuet2][streamingvlm-prerope] second_per_grid_ts exists without video_grid_thw."
            )
        if video_grid_thw is not None and second_per_grid_ts is not None:
            if int(video_grid_thw.shape[0]) != int(second_per_grid_ts.shape[0]):
                raise RuntimeError(
                    "[MMDuet2][streamingvlm-prerope] video_grid_thw and second_per_grid_ts length mismatch."
                )

        model_cfg = getattr(self.client.model, "config", None)
        vision_start_token_id = getattr(model_cfg, "vision_start_token_id", None)
        image_token_id = getattr(model_cfg, "image_token_id", None)
        if (
            image_grid_thw is not None
            and vision_start_token_id is not None
            and image_token_id is not None
        ):
            row = input_ids[0]
            starts = torch.argwhere(row == int(vision_start_token_id)).squeeze(1)
            if starts.numel() > 0:
                starts = starts[starts + 1 < row.shape[0]]
            vision_tokens = (
                row.index_select(0, starts + 1)
                if starts.numel() > 0
                else torch.empty((0,), device=row.device, dtype=row.dtype)
            )
            image_blocks = int((vision_tokens == int(image_token_id)).sum().item())
            if image_blocks != int(image_grid_thw.shape[0]):
                raise RuntimeError(
                    "[MMDuet2][streamingvlm-prerope] image block/grid mismatch in assembled ledger."
                )

        video_token_id = getattr(model_cfg, "video_token_id", None)
        if (
            video_grid_thw is not None
            and vision_start_token_id is not None
            and video_token_id is not None
        ):
            row = input_ids[0]
            starts = torch.argwhere(row == int(vision_start_token_id)).squeeze(1)
            if starts.numel() > 0:
                starts = starts[starts + 1 < row.shape[0]]
            vision_tokens = (
                row.index_select(0, starts + 1)
                if starts.numel() > 0
                else torch.empty((0,), device=row.device, dtype=row.dtype)
            )
            video_blocks = int((vision_tokens == int(video_token_id)).sum().item())
            if video_blocks != int(video_grid_thw.shape[0]):
                raise RuntimeError(
                    "[MMDuet2][streamingvlm-prerope] video block/grid mismatch in assembled ledger."
                )

        return (
            input_ids,
            attention_mask,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts,
        )

    def _svlm_compute_position_ids_for_segments(
        self, *, segments: list[_PreRopeSegment], device: torch.device
    ) -> torch.Tensor:
        if not segments:
            return torch.zeros((3, 1, 0), dtype=torch.long, device=device)

        get_rope_index = getattr(self.client.model, "get_rope_index", None)
        if not callable(get_rope_index):
            raise RuntimeError(
                "[MMDuet2][streamingvlm-prerope] model.get_rope_index is unavailable."
            )

        (
            input_ids,
            attention_mask,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts,
        ) = self._svlm_build_multimodal_ledger(segments=segments, device=device)

        try:
            rope_out = get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
        except TypeError:
            rope_out = get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )

        position_ids = rope_out[0] if isinstance(rope_out, (tuple, list)) else rope_out
        if not torch.is_tensor(position_ids):
            raise RuntimeError(
                "[MMDuet2][streamingvlm-prerope] get_rope_index did not return tensor position_ids."
            )

        position_ids = position_ids.to(device=device)
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).contiguous()
        elif position_ids.ndim == 3:
            if int(position_ids.shape[0]) == 3:
                position_ids = position_ids.contiguous()
            elif int(position_ids.shape[1]) == 3 and int(position_ids.shape[0]) == 1:
                position_ids = position_ids.transpose(0, 1).contiguous()
            else:
                raise RuntimeError(
                    f"[MMDuet2][streamingvlm-prerope] unsupported position_ids shape from get_rope_index: {tuple(position_ids.shape)}"
                )
        else:
            raise RuntimeError(
                f"[MMDuet2][streamingvlm-prerope] invalid position_ids rank from get_rope_index: {position_ids.ndim}"
            )

        expected_len = int(input_ids.shape[1])
        if (
            int(position_ids.shape[0]) != 3
            or int(position_ids.shape[2]) != expected_len
        ):
            raise RuntimeError(
                "[MMDuet2][streamingvlm-prerope] position_ids length mismatch with assembled multimodal ledger."
            )
        if int(position_ids.shape[1]) != int(input_ids.shape[0]):
            raise RuntimeError(
                "[MMDuet2][streamingvlm-prerope] position_ids batch mismatch with assembled multimodal ledger."
            )

        return position_ids

    def _svlm_active_segments(self) -> list[_PreRopeSegment]:
        segs: list[_PreRopeSegment] = []
        if self._svlm_system_segment is not None:
            segs.append(self._svlm_system_segment)
        if self._svlm_memory_segment is not None:
            segs.append(self._svlm_memory_segment)
        segs.extend(list(self._svlm_query_segments))
        segs.extend(list(self._svlm_frame_segments))
        return segs

    def _svlm_expected_prefix_len(self) -> int:
        total = 0
        for seg in self._svlm_active_segments():
            total += int(seg.token_ids.numel())
        return total

    def _svlm_assemble_prerope_legacy(
        self,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        segments = self._svlm_active_segments()
        if not segments:
            return tuple()
        return self._cat_kv_list([seg.pre_rope_kv for seg in segments])

    def _svlm_apply_rope_to_legacy(
        self,
        legacy_pre_rope: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        *,
        position_ids: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        if not legacy_pre_rope:
            return tuple()

        total_len = int(legacy_pre_rope[0][0].shape[2])
        if total_len <= 0:
            return tuple()

        if not torch.is_tensor(position_ids):
            raise RuntimeError(
                "[MMDuet2][streamingvlm-prerope] position_ids must be a tensor for RoPE application."
            )

        model_core = getattr(self.client.model, "model", None)
        if model_core is None:
            raise RuntimeError(
                "Model core is missing; cannot compute rotary embeddings."
            )
        hidden_size = int(getattr(model_core.config, "hidden_size", 0) or 0)
        if hidden_size <= 0:
            raise RuntimeError("Invalid model hidden size for rotary embedding.")

        key0 = legacy_pre_rope[0][0]
        device = key0.device
        dtype = key0.dtype

        pos_ids = position_ids.to(device=device)
        if pos_ids.ndim != 3 or int(pos_ids.shape[0]) != 3:
            raise RuntimeError(
                f"[MMDuet2][streamingvlm-prerope] invalid RoPE position_ids shape: {tuple(pos_ids.shape)}"
            )
        if int(pos_ids.shape[1]) != 1:
            raise RuntimeError(
                f"[MMDuet2][streamingvlm-prerope] RoPE position_ids batch must be 1, got {tuple(pos_ids.shape)}"
            )
        if int(pos_ids.shape[2]) != total_len:
            raise RuntimeError(
                "[MMDuet2][streamingvlm-prerope] RoPE position_ids length mismatch with assembled pre-rope KV."
            )

        dummy_hidden = torch.zeros(
            (1, total_len, hidden_size), device=device, dtype=dtype
        )
        cos, sin = model_core.rotary_emb(dummy_hidden, pos_ids)

        rope_scaling = getattr(self.client.model.config, "rope_scaling", None) or {}
        mrope_section = rope_scaling.get("mrope_section", [16, 24, 24])

        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for pre_k, v in legacy_pre_rope:
            dummy_q = torch.zeros_like(pre_k)
            _, rotated_k = self._svlm_apply_rotary_orig(
                dummy_q, pre_k, cos, sin, mrope_section
            )
            out.append((rotated_k.contiguous(), v.contiguous()))
        return tuple(out)

    def _svlm_assemble_postrope_prefix(
        self,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        segments = self._svlm_active_segments()
        pre = self._svlm_assemble_prerope_legacy()
        if not pre:
            self._svlm_last_prefix_position_max = -1.0
            return tuple()

        device = pre[0][0].device
        position_ids = self._svlm_compute_position_ids_for_segments(
            segments=segments, device=device
        )
        if int(position_ids.shape[2]) != int(pre[0][0].shape[2]):
            raise RuntimeError(
                "[MMDuet2][streamingvlm-prerope] assembled prefix position_ids length mismatch."
            )
        self._svlm_last_prefix_position_max = (
            float(position_ids.max().item()) if int(position_ids.numel()) > 0 else -1.0
        )
        return self._svlm_apply_rope_to_legacy(pre, position_ids=position_ids)

    def _svlm_guard_prefix_budget(
        self,
        assembled_post_rope: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ):
        expected = self._svlm_expected_prefix_len()
        actual = self._kv_seq_len(assembled_post_rope)
        status = "ok" if expected == actual else "drift"
        self._svlm_last_guard = {
            "expected": expected,
            "actual": actual,
            "status": status,
        }
        if expected != actual:
            raise RuntimeError(
                f"[MMDuet2][streamingvlm-prerope] token budget drift: expected={expected} actual={actual}"
            )

    def _compose_system_prompt(self) -> str:
        return self._base_system_prompt

    @staticmethod
    def _strip_trailing_respond_sentence(question: str) -> str:
        text = " ".join(str(question or "").split()).strip()
        if not text:
            return ""
        return re.sub(
            r"([.!?])\s*Respond\b.*$", r"\1", text, flags=re.IGNORECASE
        ).strip()

    _ABD_RESPOND_WORDS_RE = re.compile(
        r"\bRespond\b.*?\bwords?\s*:\s*(?P<words>.+?)\s*$", re.IGNORECASE
    )

    @classmethod
    def _abd_answer_instruction(cls, raw_question: str) -> str:
        match = cls._ABD_RESPOND_WORDS_RE.search(raw_question)
        if not match:
            return (
                "Answer with exactly one token: start if the event is "
                'occurring, otherwise output "NO REPLY".'
            )

        words = [
            w.strip().strip(".,;:").lower()
            for w in re.split(r"\s*(?:,|\band\b|\bor\b|/)\s*", match.group("words"))
            if w.strip().strip(".,;:")
        ]
        if not words:
            return (
                "Answer with exactly one token: start if the event is "
                'occurring, otherwise output "NO REPLY".'
            )

        seen: list[str] = []
        for w in words:
            if w not in seen:
                seen.append(w)

        return (
            f"Answer with exactly one token: {' or '.join(seen)} if the event is "
            'occurring, otherwise output "NO REPLY".'
        )

    def _normalize_question_for_active_task(self, question: str) -> str:
        text = " ".join(str(question or "").split()).strip()
        ds = (self._active_task or "").lower()
        if ds in {"pnr", "abd"}:
            stripped = self._strip_trailing_respond_sentence(text)
            if stripped:
                if ds == "abd":
                    # Replace only the trailing instruction, keeping the query verbatim.
                    return f"{stripped} {self._abd_answer_instruction(text)}"
                return stripped
        return text

    def _question_for_prompt(self, question: str) -> str:
        if (self._active_task or "").lower() == "pnr":
            return (
                f"{question} Answer with exactly one token: now if the event is "
                'happening now, otherwise output "NO REPLY".'
            )
        return question

    def _build_query_lines(
        self,
        tick_questions: list[dict[str, Any]],
        events: list[dict[str, Any]],
    ) -> list[str]:
        ds = (self._active_task or "").lower()
        lines: list[str] = []
        for q in tick_questions:
            question = str(q["question"]).strip()
            if not question:
                continue
            if ds == "ui":
                lines.append(self._ui_user_prompt)
                continue
            if ds in {"si", "spg"}:
                instr = self._si_user_prompt if ds == "si" else self._spg_user_prompt
                lines.append(f"{instr}\n{question}")
                continue
            ask_time = float(q["ask_time"])
            turn_id = str(q["turn_id"])
            self._append_question_event(
                events, ask_time=ask_time, question=question, turn_id=turn_id
            )
            if self._normalize_kv_mode() == "streamingvlm":
                lines.append(f"{turn_id} {self._question_for_prompt(question)}")
            else:
                lines.append(self._question_for_prompt(question))
        return lines

    def _group_questions_by_tick(
        self, turns: list[dict[str, Any]]
    ) -> dict[int, list[dict[str, Any]]]:
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for turn_idx, turn in enumerate(turns):
            question = self._normalize_question_for_active_task(
                str(turn.get("question", ""))
            )
            ask_time = float(turn.get("ask_time", 0.0) or 0.0)
            tick = max(0, int(round(ask_time * self.stream_fps)))
            grouped[tick].append(
                {
                    "turn_idx": turn_idx,
                    "turn_id": f"Q{turn_idx + 1}",
                    "ask_time": ask_time,
                    "question": question,
                }
            )

        for tick in grouped:
            grouped[tick].sort(key=lambda x: (x["ask_time"], x["turn_idx"]))

        return grouped

    @staticmethod
    def _sanitize_response(text: str) -> str:
        cleaned = str(text or "")
        cleaned = cleaned.replace("<|im_end|>", "")
        cleaned = cleaned.replace("</s>", "")
        cleaned = cleaned.strip()
        if cleaned.lower().startswith("i must reply."):
            cleaned = cleaned[len("i must reply.") :].strip()
        return cleaned

    @staticmethod
    def _is_silent(text: str) -> bool:
        s = str(text or "").strip().lower()
        if not s:
            return True
        if s in {
            "no",
            "none",
            "nothing",
            "no answer",
            "no response",
            "no reply",
            "noreply",
        }:
            return True
        if s.startswith("no reply"):
            return True
        if not any(ch.isalnum() for ch in s):
            return True
        return False

    def _normalize_silence_output(self, text: str) -> str:
        if self._is_silent(text):
            return "no"
        return str(text).strip()

    @staticmethod
    def _append_question_event(
        events: list[dict[str, Any]], ask_time: float, question: str, turn_id: str
    ):
        events.append(
            {
                "time": float(round(ask_time, 3)),
                "type": "question",
                "value": question,
                "turn_id": turn_id,
            }
        )

    def _parse_response_lines(self, response_text: str) -> dict[str, str]:
        hits: dict[str, str] = {}
        for line in response_text.splitlines():
            match = self.RESPONSE_LINE_RE.match(line)
            if not match:
                continue
            turn_id = match.group(1).upper()
            answer = match.group(2).strip()
            if not answer:
                continue
            hits[turn_id] = answer
        return hits

    @classmethod
    def _append_response_event(
        cls,
        events: list[dict[str, Any]],
        *,
        t: float,
        value: str,
        raw_text: str,
        latency: float | None,
        turn_id: str | None = None,
    ):
        event = {
            "time": float(round(t, 3)),
            "type": "response",
            "value": value,
            "raw_text": raw_text,
        }
        if turn_id is not None:
            event["turn_id"] = turn_id
        if latency is not None and not cls._is_silent(value):
            event["latency"] = float(round(latency, 4))
        events.append(event)

    FRAME_SHORT_SIDE = 560

    @staticmethod
    def _frame_to_image(frame_tchw: torch.Tensor) -> Image.Image:
        if isinstance(frame_tchw, torch.Tensor):
            frame_hwc = frame_tchw.permute(1, 2, 0).contiguous().cpu().numpy()
        else:
            frame_hwc = (
                frame_tchw.asnumpy() if hasattr(frame_tchw, "asnumpy") else frame_tchw
            )
        if frame_hwc.dtype != "uint8":
            frame_hwc = frame_hwc.clip(0, 255).astype("uint8")
        image = Image.fromarray(frame_hwc)

        short_side = min(image.size)
        if short_side > MMDuet2.FRAME_SHORT_SIDE:
            scale = MMDuet2.FRAME_SHORT_SIDE / short_side
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.BICUBIC,
            )
        return image

    @staticmethod
    def _clone_content(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [dict(item) for item in content]

    @staticmethod
    def _extract_query_lines_from_content(content: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for item in content:
            if item.get("type") != "text":
                continue
            text = str(item.get("text", ""))
            for raw in text.splitlines():
                line = raw.strip()
                if line:
                    lines.append(line)
        return lines

    def _kv_seq_len(self, past_key_values: Any) -> int:
        if past_key_values is None:
            return 0

        if hasattr(past_key_values, "get_seq_length"):
            try:
                return int(past_key_values.get_seq_length())
            except Exception:
                pass

        layers = getattr(past_key_values, "layers", None)
        if layers:
            first_layer = layers[0]
            keys = getattr(first_layer, "keys", None)
            if torch.is_tensor(keys) and keys.ndim >= 3:
                return int(keys.shape[2])

        key_cache = getattr(past_key_values, "key_cache", None)
        if isinstance(key_cache, (list, tuple)) and key_cache:
            first_key = key_cache[0]
            if torch.is_tensor(first_key) and first_key.ndim >= 3:
                return int(first_key.shape[2])

        if isinstance(past_key_values, (tuple, list)) and past_key_values:
            first_layer = past_key_values[0]
            if isinstance(first_layer, (tuple, list)) and first_layer:
                first_key = first_layer[0]
                if torch.is_tensor(first_key) and first_key.ndim >= 3:
                    return int(first_key.shape[2])

        return 0

    def _trim_kv_tail(self, past_key_values: Any, n_tokens: int) -> Any:
        if past_key_values is None or n_tokens <= 0:
            return past_key_values

        seq_len = self._kv_seq_len(past_key_values)
        if seq_len <= 0:
            return past_key_values

        keep_len = seq_len - int(n_tokens)
        if keep_len <= 0:
            return None

        layers = getattr(past_key_values, "layers", None)
        if layers:
            for layer in layers:
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                if torch.is_tensor(keys) and keys.ndim >= 3:
                    layer.keys = keys[:, :, :keep_len, :]
                if torch.is_tensor(values) and values.ndim >= 3:
                    layer.values = values[:, :, :keep_len, :]
            if hasattr(past_key_values, "_seen_tokens"):
                try:
                    past_key_values._seen_tokens = keep_len
                except Exception:
                    pass
            return past_key_values

        key_cache = getattr(past_key_values, "key_cache", None)
        value_cache = getattr(past_key_values, "value_cache", None)
        if isinstance(key_cache, list) and isinstance(value_cache, list):
            for layer_i in range(min(len(key_cache), len(value_cache))):
                k = key_cache[layer_i]
                v = value_cache[layer_i]
                if torch.is_tensor(k) and k.ndim >= 3:
                    key_cache[layer_i] = k[:, :, :keep_len, :]
                if torch.is_tensor(v) and v.ndim >= 3:
                    value_cache[layer_i] = v[:, :, :keep_len, :]
            if hasattr(past_key_values, "_seen_tokens"):
                try:
                    past_key_values._seen_tokens = keep_len
                except Exception:
                    pass
            return past_key_values

        if isinstance(past_key_values, tuple):
            trimmed = []
            for layer in past_key_values:
                if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                    k, v = layer[0], layer[1]
                    if torch.is_tensor(k) and torch.is_tensor(v):
                        trimmed.append((k[:, :, :keep_len, :], v[:, :, :keep_len, :]))
                    else:
                        trimmed.append(layer)
                else:
                    trimmed.append(layer)
            return tuple(trimmed)

        if isinstance(past_key_values, list):
            trimmed = []
            for layer in past_key_values:
                if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                    k, v = layer[0], layer[1]
                    if torch.is_tensor(k) and torch.is_tensor(v):
                        trimmed.append((k[:, :, :keep_len, :], v[:, :, :keep_len, :]))
                    else:
                        trimmed.append(layer)
                else:
                    trimmed.append(layer)
            return trimmed

        return past_key_values

    def _drop_generated_from_kv(self):
        past_key_values = getattr(self.client, "past_key_values", None)
        generated_len = int(getattr(self.client, "_last_generated_len", 0) or 0)
        if past_key_values is None or generated_len <= 0:
            self.client._last_generated_len = 0
            return

        pre_len = self._kv_seq_len(past_key_values)
        self.client.past_key_values = self._trim_kv_tail(past_key_values, generated_len)
        post_len = self._kv_seq_len(self.client.past_key_values)
        self.client._last_generated_len = 0

    def _clear_last_assistant_text(self):
        history = getattr(self.client, "history", None)
        if not history:
            return
        if history[-1].get("role") == "assistant":
            history[-1]["content"] = ""

    @staticmethod
    def _seq_list(ids: torch.Tensor | None) -> list[int]:
        if ids is None:
            return []
        return [int(x) for x in ids.flatten().tolist()]

    def _collect_role_spans(
        self, ids: torch.Tensor | None, role_key: str
    ) -> list[tuple[int, int]]:
        seq = self._seq_list(ids)
        if not seq:
            return []

        im_start = self._tok_ids["im_start"]
        im_end = self._tok_ids["im_end"]
        role_id = self._tok_ids[role_key]
        newline = self._tok_ids["newline"]

        spans: list[tuple[int, int]] = []
        n = len(seq)
        i = 0
        while i < n - 1:
            if seq[i] == im_start and seq[i + 1] == role_id:
                s = i
                j = i + 2
                while j < n and seq[j] != im_end:
                    j += 1
                if j >= n:
                    break
                e = j
                if j + 1 < n and seq[j + 1] == newline:
                    e = j + 1
                spans.append((s, e))
                i = e + 1
            else:
                i += 1
        return spans

    def _collect_user_spans(self, ids: torch.Tensor | None) -> list[tuple[int, int]]:
        return self._collect_role_spans(ids, "user")

    def _refresh_turn_tag_state_from_history(self):
        frame_tags: deque[str] = deque()
        query_tags: deque[str] = deque()
        memory_tag: str | None = None

        for turn in self.client.history:
            if turn.get("role") != "user":
                continue
            tag = turn.get("_kv_tag")
            if not tag:
                continue
            if str(tag).startswith("F:"):
                frame_tags.append(str(tag))
            elif str(tag).startswith("Q:"):
                query_tags.append(str(tag))
            elif str(tag).startswith("M:"):
                memory_tag = str(tag)

        self._frame_turn_tags = frame_tags
        self._query_turn_tags = query_tags
        self._memory_turn_tag = memory_tag

    def _sync_prev_generated_ids_after_decode(self):
        seq = getattr(self.client, "_last_sequences", None)
        if torch.is_tensor(seq) and seq.ndim >= 2:
            self._prev_generated_ids = seq.detach().clone()
        else:
            return

        generated_len = int(getattr(self.client, "_last_generated_len", 0) or 0)
        if generated_len <= 0:
            return

        keep = int(self._prev_generated_ids.shape[1]) - generated_len
        if keep <= 0:
            self._prev_generated_ids = None
        else:
            self._prev_generated_ids = self._prev_generated_ids[:, :keep].contiguous()

    def _update_system_sink_len_from_ids(self):
        if self._prev_generated_ids is None:
            return
        user_spans = self._collect_user_spans(self._prev_generated_ids)
        if user_spans:
            self._system_sink_len = int(user_spans[0][0])
            return
        if self._system_sink_len is None:
            self._system_sink_len = int(self._prev_generated_ids.shape[1])

    def _invalidate_model_keep_masks(self):

        model_core = getattr(getattr(self.client, "model", None), "model", None)
        if model_core is None:
            return
        if hasattr(model_core, "all_keep_masks"):
            model_core.all_keep_masks = []

    def _normalize_kv_mode(self) -> str:
        """
        original     -- the unmodified MMDuet2 cache; grows without bound and
                        OOMs on long videos.
        kvflush      -- flush back to the system sink whenever a new turn opens.
        streamingvlm -- StreamingVLM-style pre-rope sink + sliding window.
        """
        mode = str(getattr(self, "kv_mode", "kvflush") or "kvflush").strip().lower()
        if mode in {"original", "kvflush", "streamingvlm"}:
            return mode
        raise ValueError(
            f"Unknown kv_mode={mode!r}. Expected one of: original, kvflush, streamingvlm."
        )

    def _reset_runtime(
        self,
        *,
        step: float,
        system_prompt: str,
        video_time: float,
    ):
        self.client.reset()
        self.client.set_fps(frame_interval=step)
        self.client.system_prompt = system_prompt
        self.client.history = [{"role": "system", "content": self.client.system_prompt}]
        self.client.query_queue.clear()
        self.client._last_input_len = 0
        self.client._last_generated_len = 0
        self.client._last_sequences = None
        self.client.video_time = max(0.0, float(video_time))

        self._prev_generated_ids = None
        self._system_sink_len = None
        self._refresh_turn_tag_state_from_history()

    def _mode_original_pre(
        self, *, current_turn: dict[str, Any], **_: Any
    ) -> dict[str, Any]:
        self.client.query_queue.append(current_turn)
        return {"mode": "original"}

    def _mode_original_post(self, **_: Any):
        return

    def _encode_text_tokens(self, text: str) -> list[int]:
        tokenizer = getattr(self.client.processor, "tokenizer", None)
        if tokenizer is None:
            return []
        try:
            return tokenizer(text, add_special_tokens=False)["input_ids"]
        except Exception:
            return []

    def _decode_text_tokens(self, token_ids: list[int]) -> str:
        tokenizer = getattr(self.client.processor, "tokenizer", None)
        if tokenizer is None:
            return ""
        try:
            return tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            return ""

    def _roll_text_by_budget(self, text: str) -> str:
        text = str(text or "").strip()
        if not text:
            return ""

        token_ids = self._encode_text_tokens(text)
        if not token_ids:
            return text

        sink = self.text_sink
        window = self.text_sliding_window

        if sink is None and window is None:
            return text
        if sink is None:
            keep_ids = token_ids[-int(window) :] if int(window) > 0 else []
            decoded = self._decode_text_tokens(keep_ids)
            return decoded if decoded else text
        if window is None:
            keep_ids = token_ids[: int(sink)] if int(sink) > 0 else []
            decoded = self._decode_text_tokens(keep_ids)
            return decoded if decoded else text

        sink = int(max(0, sink))
        window = int(max(0, window))
        if len(token_ids) <= sink + window:
            return text

        head = token_ids[:sink] if sink > 0 else []
        tail = token_ids[-window:] if window > 0 else []
        head_text = self._decode_text_tokens(head)
        tail_text = self._decode_text_tokens(tail)
        rolled = "\n...\n".join(x for x in [head_text, tail_text] if x)
        return rolled if rolled else text

    def _build_stream_memory_text(self) -> str:
        if not self._stream_query_ledger:
            return ""
        joined = "\n".join(self._stream_query_ledger)
        rolled = self._roll_text_by_budget(joined)
        if not rolled:
            return ""
        return "Long-term query memory (older user queries):\n" f"{rolled}"

    def _svlm_query_line_from_segment(self, seg: _PreRopeSegment) -> str:
        if str(seg.kind) != "query":
            return ""
        content = seg.content
        if not isinstance(content, list):
            return ""
        lines = self._extract_query_lines_from_content(content)
        if not lines:
            return ""
        return str(lines[-1]).strip()

    def _svlm_clear_post_system_segments(self, *, clear_query_ledger: bool):
        if self._svlm_memory_segment is not None:
            self._svlm_last_evicted_tags.append(self._svlm_memory_segment.tag)
            self._svlm_memory_segment = None

        while self._svlm_query_segments:
            evicted = self._svlm_query_segments.popleft()
            self._svlm_last_evicted_tags.append(evicted.tag)
        while self._svlm_frame_segments:
            evicted = self._svlm_frame_segments.popleft()
            self._svlm_last_evicted_tags.append(evicted.tag)

        self._query_turn_tags.clear()
        self._frame_turn_tags.clear()
        self._memory_turn_tag = None
        self._memory_text = ""
        if clear_query_ledger:
            self._stream_query_ledger = []

    def _svlm_encode_segment(
        self,
        *,
        tag: str,
        kind: str,
        role: str,
        content: Any,
        vision_meta: dict[str, Any] | None = None,
    ) -> _PreRopeSegment:
        if role not in {"system", "user"}:
            raise RuntimeError(
                f"[MMDuet2][streamingvlm-prerope] unsupported role={role!r} for segment encoding"
            )

        if role == "system":
            message = {"role": "system", "content": str(content)}
        else:
            message = {"role": "user", "content": self._clone_content(content)}
        messages = [message]

        text = self.client.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.client.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = next(self.client.model.parameters()).device
        inputs = self._tensorize_to_device(inputs, device=device)
        input_ids = inputs["input_ids"]
        seg_len = int(input_ids.shape[1])
        if seg_len <= 0:
            raise RuntimeError(
                f"[MMDuet2][streamingvlm-prerope] empty tokenized segment tag={tag}"
            )

        token_ids = input_ids[0].detach().to("cpu").long().contiguous()
        image_grid_thw = self._svlm_optional_cpu_tensor(
            inputs.get("image_grid_thw", None), dtype=torch.long
        )
        video_grid_thw = self._svlm_optional_cpu_tensor(
            inputs.get("video_grid_thw", None), dtype=torch.long
        )
        second_per_grid_ts = self._svlm_optional_cpu_tensor(
            inputs.get("second_per_grid_ts", None), dtype=torch.float32
        )
        incoming_segment = _PreRopeSegment(
            tag=tag,
            kind=kind,
            role=role,
            content=message["content"],
            token_ids=token_ids,
            pre_rope_kv=tuple(),
            vision_meta=dict(vision_meta or {}),
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
        )

        prefix_post_rope = self._svlm_assemble_postrope_prefix()
        prefix_len = self._kv_seq_len(prefix_post_rope)
        full_segments = self._svlm_active_segments() + [incoming_segment]
        full_position_ids = self._svlm_compute_position_ids_for_segments(
            segments=full_segments, device=device
        )
        expected_full_len = sum(int(seg.token_ids.numel()) for seg in full_segments)
        if int(full_position_ids.shape[2]) != expected_full_len:
            raise RuntimeError(
                "[MMDuet2][streamingvlm-prerope] segment ledger / position_ids length mismatch."
            )
        if prefix_len != expected_full_len - seg_len:
            raise RuntimeError(
                f"[MMDuet2][streamingvlm-prerope] prefix length mismatch for tag={tag}: "
                f"prefix_len={prefix_len} expected={expected_full_len - seg_len}"
            )

        position_ids = full_position_ids[:, :, -seg_len:].contiguous()
        attention_mask = self._build_attention_mask(
            prefix_len=prefix_len, input_len=seg_len, device=device
        )
        past_key_values = self._prepare_past_for_model(prefix_post_rope)

        if hasattr(self.client.model, "rope_deltas"):
            self.client.model.rope_deltas = None

        self._begin_prerope_capture()
        try:
            forward_inputs = dict(inputs)
            forward_inputs["attention_mask"] = attention_mask
            forward_inputs["position_ids"] = self._svlm_to_lm_position_ids(position_ids)
            with torch.no_grad():
                outputs = self.client.model(
                    **forward_inputs,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    drop_method="none",
                    drop_threshold=1.0,
                    drop_absolute=True,
                )
        finally:
            captured_pre_rope_keys = self._end_prerope_capture()

        full_legacy = self._to_legacy_kv(outputs.past_key_values)
        full_len = self._kv_seq_len(full_legacy)
        if full_len < prefix_len + seg_len:
            raise RuntimeError(
                f"[MMDuet2][streamingvlm-prerope] invalid cache growth for tag={tag}: "
                f"prefix_len={prefix_len} seg_len={seg_len} full_len={full_len}"
            )

        seg_slice = self._slice_legacy_kv(full_legacy, prefix_len, prefix_len + seg_len)
        if len(captured_pre_rope_keys) != len(seg_slice):
            raise RuntimeError(
                f"[MMDuet2][streamingvlm-prerope] pre-rope capture layer mismatch for tag={tag}: "
                f"captured={len(captured_pre_rope_keys)} sliced={len(seg_slice)}"
            )

        pre_rope_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, ((_, v_slice), k_pre) in enumerate(
            zip(seg_slice, captured_pre_rope_keys)
        ):
            if not torch.is_tensor(k_pre) or int(k_pre.shape[2]) != seg_len:
                raise RuntimeError(
                    f"[MMDuet2][streamingvlm-prerope] invalid pre-rope key shape at layer={layer_idx} tag={tag}"
                )
            pre_rope_layers.append(
                (k_pre.detach().contiguous(), v_slice.detach().contiguous())
            )

        return _PreRopeSegment(
            tag=tag,
            kind=kind,
            role=role,
            content=message["content"],
            token_ids=token_ids,
            pre_rope_kv=tuple(pre_rope_layers),
            vision_meta=dict(vision_meta or {}),
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
        )

    def _svlm_add_segment(
        self,
        *,
        tag: str,
        kind: str,
        role: str,
        content: Any,
        vision_meta: dict[str, Any] | None = None,
    ):
        seg = self._svlm_encode_segment(
            tag=tag,
            kind=kind,
            role=role,
            content=content,
            vision_meta=vision_meta,
        )

        if kind == "system":
            self._svlm_system_segment = seg
            return

        if kind == "memory":
            if self._svlm_memory_segment is not None:
                self._svlm_last_evicted_tags.append(self._svlm_memory_segment.tag)
            self._svlm_memory_segment = seg
            self._memory_turn_tag = seg.tag
            return

        if kind == "query":
            self._svlm_query_segments.append(seg)
            self._query_turn_tags.append(seg.tag)
            while len(self._svlm_query_segments) > int(self.text_round):
                evicted = self._svlm_query_segments.popleft()
                self._svlm_last_evicted_tags.append(evicted.tag)
                evicted_line = self._svlm_query_line_from_segment(evicted)
                if evicted_line:
                    self._stream_query_ledger.append(evicted_line)
            while len(self._query_turn_tags) > int(self.text_round):
                self._query_turn_tags.popleft()
            return

        if kind == "frame":
            self._svlm_frame_segments.append(seg)
            self._frame_turn_tags.append(seg.tag)
            while len(self._svlm_frame_segments) > int(self.visual_round):
                evicted = self._svlm_frame_segments.popleft()
                self._svlm_last_evicted_tags.append(evicted.tag)
            while len(self._frame_turn_tags) > int(self.visual_round):
                self._frame_turn_tags.popleft()
            return

        raise RuntimeError(
            f"[MMDuet2][streamingvlm-prerope] unsupported segment kind={kind!r}"
        )

    def _svlm_remove_memory_segment(self):
        if self._svlm_memory_segment is None:
            return
        self._svlm_last_evicted_tags.append(self._svlm_memory_segment.tag)
        self._svlm_memory_segment = None
        self._memory_turn_tag = None

    def _svlm_rebuild_history(self, *, tick_time: float):
        history: list[dict[str, Any]] = [
            {"role": "system", "content": self.client.system_prompt}
        ]
        if self._svlm_memory_segment is not None:
            history.append(
                {
                    "role": "user",
                    "content": self._clone_content(self._svlm_memory_segment.content),
                    "_kv_tag": self._svlm_memory_segment.tag,
                }
            )
        for seg in self._svlm_query_segments:
            history.append(
                {
                    "role": "user",
                    "content": self._clone_content(seg.content),
                    "_kv_tag": seg.tag,
                }
            )
        for seg in self._svlm_frame_segments:
            history.append(
                {
                    "role": "user",
                    "content": self._clone_content(seg.content),
                    "_kv_tag": seg.tag,
                }
            )
        history.append({"role": "assistant", "content": "", "time": float(tick_time)})
        self.client.history = history
        self.client.video_time = float(tick_time)
        self.client.query_queue.clear()
        self._refresh_turn_tag_state_from_history()

    def _svlm_decode_response(self) -> str:
        prefix_post_rope = self._svlm_assemble_postrope_prefix()
        self._svlm_guard_prefix_budget(prefix_post_rope)
        prefix_len = self._kv_seq_len(prefix_post_rope)

        device = next(self.client.model.parameters()).device
        prompt_ids = self._svlm_assistant_prompt_ids.to(device).unsqueeze(0)
        prompt_len = int(prompt_ids.shape[1])
        if prompt_len <= 0:
            raise RuntimeError(
                f"[MMDuet2][streamingvlm-prerope] invalid decode prompt length: prefix_len={prefix_len} prompt_len={prompt_len}"
            )
        prompt_start = 0
        if prefix_len > 0:
            if float(self._svlm_last_prefix_position_max) < 0.0:
                raise RuntimeError(
                    "[MMDuet2][streamingvlm-prerope] missing prefix position max for decode anchoring."
                )
            prompt_start = (
                int(math.floor(float(self._svlm_last_prefix_position_max))) + 1
            )
        position_ids = self._build_position_ids(
            start=prompt_start, length=prompt_len, device=device
        )
        if tuple(position_ids.shape) != (3, 1, prompt_len):
            raise RuntimeError(
                "[MMDuet2][streamingvlm-prerope] decode position_ids shape mismatch: "
                f"prefix_len={prefix_len} prompt_len={prompt_len} "
                f"position_ids.shape={tuple(position_ids.shape)}"
            )
        cache_position = self._svlm_build_decode_cache_position(
            prefix_len=prefix_len, prompt_len=prompt_len, device=device
        )
        if int(cache_position.numel()) != prompt_len:
            raise RuntimeError(
                "[MMDuet2][streamingvlm-prerope] decode cache_position length mismatch: "
                f"prefix_len={prefix_len} prompt_len={prompt_len} "
                f"cache_position.shape={tuple(cache_position.shape)}"
            )
        past_key_values = self._prepare_past_for_model(prefix_post_rope)

        if hasattr(self.client.model, "rope_deltas"):
            self.client.model.rope_deltas = None

        eos_token_ids: set[int] = {int(self._tok_ids.get("im_end", 151645))}
        gen_cfg = getattr(self.client.model, "generation_config", None)
        if gen_cfg is not None:
            eos_cfg = getattr(gen_cfg, "eos_token_id", None)
            if isinstance(eos_cfg, int):
                eos_token_ids.add(int(eos_cfg))
            elif isinstance(eos_cfg, (list, tuple)):
                for token_id in eos_cfg:
                    if isinstance(token_id, int):
                        eos_token_ids.add(int(token_id))

        with torch.no_grad():

            last_logits: torch.Tensor | None = None
            for i in range(prompt_len):
                step_input_ids = prompt_ids[:, i : i + 1]
                step_position_ids = position_ids[:, :, i : i + 1]
                step_cache_position = cache_position[i : i + 1]
                outputs = self.client.model(
                    input_ids=step_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    position_ids=self._svlm_to_lm_position_ids(step_position_ids),
                    cache_position=step_cache_position,
                )
                past_key_values = outputs.past_key_values
                logits = getattr(outputs, "logits", None)
                if not torch.is_tensor(logits) or logits.ndim < 3:
                    raise RuntimeError(
                        "[MMDuet2][streamingvlm-prerope] invalid logits during prompt prefill."
                    )
                last_logits = logits[:, -1, :]

            if last_logits is None:
                return ""

            decode_budget = int(self._max_new_tokens_for_active_task())
            generated_token_ids: list[int] = []
            for step_idx in range(decode_budget):
                if self._do_sample_for_active_task():
                    logits_for_sampling = last_logits
                    temp = max(1e-5, float(self._temperature_for_active_task()))
                    logits_for_sampling = logits_for_sampling / temp
                    top_k = int(self.top_k)
                    if top_k > 0 and top_k < int(logits_for_sampling.shape[-1]):
                        topk_vals, _ = torch.topk(logits_for_sampling, k=top_k, dim=-1)
                        kth = topk_vals[:, -1].unsqueeze(-1)
                        logits_for_sampling = logits_for_sampling.masked_fill(
                            logits_for_sampling < kth, float("-inf")
                        )
                    probs = torch.softmax(logits_for_sampling, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

                token_id = int(next_token.item())
                generated_token_ids.append(token_id)
                if token_id in eos_token_ids:
                    break

                next_position = self._build_position_ids(
                    start=prompt_start + prompt_len + step_idx,
                    length=1,
                    device=device,
                )
                next_cache_position = torch.tensor(
                    [prefix_len + prompt_len + step_idx],
                    device=device,
                    dtype=torch.long,
                )
                outputs = self.client.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    position_ids=self._svlm_to_lm_position_ids(next_position),
                    cache_position=next_cache_position,
                )
                past_key_values = outputs.past_key_values
                logits = getattr(outputs, "logits", None)
                if not torch.is_tensor(logits) or logits.ndim < 3:
                    raise RuntimeError(
                        "[MMDuet2][streamingvlm-prerope] invalid logits during autoregressive decode."
                    )
                last_logits = logits[:, -1, :]

        if not generated_token_ids:
            return ""
        generated = torch.tensor([generated_token_ids], device=device, dtype=torch.long)
        decoded = self.client.processor.batch_decode(
            generated, skip_special_tokens=True
        )
        if not decoded:
            return ""
        return str(decoded[0] or "")

    def _svlm_reset_prerope_runtime(self, *, step: float, system_prompt: str):
        self._reset_runtime(step=step, system_prompt=system_prompt, video_time=0.0)
        self._svlm_system_segment = None
        self._svlm_memory_segment = None
        self._svlm_query_segments = deque()
        self._svlm_frame_segments = deque()
        self._svlm_last_evicted_tags = []
        self._svlm_last_guard = {}
        self._svlm_last_prefix_position_max = -1.0
        self._stream_query_ledger = []
        self._memory_text = ""
        self._memory_turn_tag = None
        self._frame_turn_tags = deque()
        self._query_turn_tags = deque()
        self._stream_recent_turns = deque(maxlen=max(1, int(self.visual_round)))

        self._svlm_add_segment(
            tag="S:0000000",
            kind="system",
            role="system",
            content=system_prompt,
            vision_meta={"source": "system_prompt"},
        )
        self.client.history = [{"role": "system", "content": self.client.system_prompt}]

    def _inference_streamingvlm_prerope(
        self,
        *,
        video_path: str,
        frames: torch.Tensor,
        src_fps: float,
        duration: float,
        num_ticks: int,
        step: float,
        questions_by_tick: dict[int, list[dict[str, Any]]],
        system_prompt: str,
        profile: str,
    ) -> list[dict[str, Any]]:
        if profile not in {"streamingvlm", "kvflush"}:
            raise RuntimeError(f"[MMDuet2] unsupported prerope profile={profile!r}")
        log_prefix = f"[MMDuet2][{profile}-prerope]"

        self._svlm_reset_prerope_runtime(step=step, system_prompt=system_prompt)
        events: list[dict[str, Any]] = []

        total_frames = int(len(frames))
        stream_iter = range(num_ticks)
        decision_every = max(
            1, int(round(self.stream_fps / max(self.decision_fps, 1)))
        )
        for tick in stream_iter:
            tick_time = tick * step
            frame_idx = int(round(tick_time * src_fps))
            frame_idx = max(0, min(frame_idx, total_frames - 1))
            image = self._frame_to_image(frames[frame_idx])

            tick_questions = questions_by_tick.get(tick, [])
            question_lines = self._build_query_lines(tick_questions, events)

            self._svlm_last_evicted_tags = []
            is_decision = (tick + 1) % decision_every == 0
            response_time = min(duration, (tick + 1) * step)
            if is_decision:
                self._generation_timer = GenerationTimer(torch).start()

            if profile == "streamingvlm":
                for line in question_lines:
                    self._svlm_add_segment(
                        tag=self._new_query_tag(),
                        kind="query",
                        role="user",
                        content=[{"type": "text", "text": line}],
                        vision_meta={"source": "query"},
                    )

                new_memory_text = self._build_stream_memory_text()
                if new_memory_text != self._memory_text:
                    self._svlm_remove_memory_segment()
                    self._memory_text = new_memory_text
                    if self._memory_text:
                        self._svlm_add_segment(
                            tag=self._new_memory_tag(),
                            kind="memory",
                            role="user",
                            content=[{"type": "text", "text": self._memory_text}],
                            vision_meta={"source": "memory"},
                        )
            else:
                latest_query = question_lines[-1] if question_lines else ""
                if latest_query:

                    self._svlm_clear_post_system_segments(clear_query_ledger=True)
                    self._svlm_add_segment(
                        tag=self._new_query_tag(),
                        kind="query",
                        role="user",
                        content=[{"type": "text", "text": latest_query}],
                        vision_meta={"source": "query", "profile": "kvflush"},
                    )
                elif self._svlm_memory_segment is not None:
                    self._svlm_remove_memory_segment()
                    self._memory_text = ""

            frame_content = [{"type": "image", "image": image}]
            self._svlm_add_segment(
                tag=self._new_frame_tag(),
                kind="frame",
                role="user",
                content=frame_content,
                vision_meta={"source": "frame", "frame_idx": frame_idx},
            )
            self._stream_recent_turns.append(self._clone_content(frame_content))

            if not is_decision:
                self._svlm_rebuild_history(tick_time=response_time)
                continue

            try:
                raw_text = self._svlm_decode_response()
            except Exception as e:
                raise RuntimeError(
                    f"{log_prefix} tick={tick} decode failed: {e}"
                ) from e

            latency = self._generation_timer.finish()
            self._svlm_rebuild_history(tick_time=response_time)
            sanitized = self._sanitize_response(raw_text)

            response_hits = self._parse_response_lines(sanitized)
            if response_hits:
                for tid, answer in response_hits.items():
                    self._append_response_event(
                        events,
                        t=response_time,
                        value=self._normalize_silence_output(answer),
                        raw_text=raw_text,
                        latency=latency,
                        turn_id=tid,
                    )
            else:
                self._append_response_event(
                    events,
                    t=response_time,
                    value=self._normalize_silence_output(sanitized),
                    raw_text=raw_text,
                    latency=latency,
                )

            self._log(
                f"{log_prefix} t={response_time:.3f}s tick={tick} frame={frame_idx} "
                f"segments(system={1 if self._svlm_system_segment is not None else 0}, "
                f"memory={1 if self._svlm_memory_segment is not None else 0}, "
                f"query={len(self._svlm_query_segments)}, frame={len(self._svlm_frame_segments)}) "
                f"guard={self._svlm_last_guard} evicted={self._svlm_last_evicted_tags} "
                f"response={sanitized!r} latency={latency:.3f}s"
            )

        events.sort(
            key=lambda x: (x.get("time", 0.0), 0 if x.get("type") == "question" else 1)
        )
        return events

    def _run_one_decode(self, *, response_time_fallback: float) -> tuple[str, float]:
        raw_text = ""
        response_time = response_time_fallback

        self._invalidate_model_keep_masks()
        self.client._last_generated_len = 0
        self.client._last_sequences = None
        self.client._encode_query(debug_print=False)

        if self.client.history and self.client.history[-1].get("role") == "assistant":
            reply_turn = self.client.history[-1]
            raw_text = str(reply_turn.get("content", "") or "")
            response_time = float(
                reply_turn.get("time", response_time) or response_time
            )

        return raw_text, response_time

    def inference(self, video_path: str, turns: list) -> list:
        if not os.path.exists(video_path):
            print(f"[MMDuet2] Missing video: {video_path}")
            return []

        try:
            frames = decord.VideoReader(video_path, num_threads=2)
        except Exception as e:
            print(f"[MMDuet2] Failed to read video {video_path}: {e}")
            return []

        if len(frames) == 0:
            return []

        src_fps = float(frames.get_avg_fps() or 0.0)
        if src_fps <= 0:
            src_fps = float(self.stream_fps)

        total_frames = int(len(frames))
        duration = float(total_frames / src_fps) if src_fps > 0 else 0.0
        if duration <= 0:
            return []

        mode = self._normalize_kv_mode()
        step = 1.0 / float(self.stream_fps)
        num_ticks = max(1, int(math.ceil(duration * self.stream_fps)))
        questions_by_tick = self._group_questions_by_tick(turns)

        self._init_runtime_state()
        system_prompt = self._compose_system_prompt()
        if mode in {"streamingvlm", "kvflush"}:
            return self._inference_streamingvlm_prerope(
                video_path=video_path,
                frames=frames,
                src_fps=src_fps,
                duration=duration,
                num_ticks=num_ticks,
                step=step,
                questions_by_tick=questions_by_tick,
                system_prompt=system_prompt,
                profile=mode,
            )

        self._reset_runtime(step=step, system_prompt=system_prompt, video_time=0.0)

        events: list[dict[str, Any]] = []
        decision_every = max(
            1, int(round(self.stream_fps / max(self.decision_fps, 1)))
        )

        stream_iter = range(num_ticks)
        for tick in stream_iter:
            tick_time = tick * step
            is_decision = (tick + 1) % decision_every == 0
            frame_idx = int(round(tick_time * src_fps))
            frame_idx = max(0, min(frame_idx, total_frames - 1))
            image = self._frame_to_image(frames[frame_idx])

            tick_questions = questions_by_tick.get(tick, [])
            question_lines = self._build_query_lines(tick_questions, events)

            current_content: list[dict[str, Any]] = [{"type": "image", "image": image}]
            if question_lines:
                current_content.append(
                    {"type": "text", "text": "\n".join(question_lines)}
                )

            current_turn = {"role": "user", "content": current_content}
            has_new_turn = bool(question_lines)

            self._mode_original_pre(
                current_turn=current_turn,
                current_content=current_content,
                has_new_turn=has_new_turn,
                tick_questions=tick_questions,
                step=step,
                tick_time=tick_time,
                system_prompt=system_prompt,
            )

            self._generation_timer = GenerationTimer(torch).start()
            raw_text = ""
            response_time = min(duration, (tick + 1) * step)

            try:
                raw_text, response_time = self._run_one_decode(
                    response_time_fallback=response_time
                )
            except Exception as e:
                raise RuntimeError(
                    f"[MMDuet2] Decode failed at tick {tick} in {video_path}. "
                    "Refusing to write a partial trace; re-run with --resume to "
                    "continue from the last completed video."
                ) from e

            self._sync_prev_generated_ids_after_decode()
            self._drop_generated_from_kv()
            self._clear_last_assistant_text()

            self._mode_original_post(
                current_content=current_content,
                has_new_turn=has_new_turn,
                tick_questions=tick_questions,
                step=step,
                tick_time=tick_time,
                system_prompt=system_prompt,
            )

            self._refresh_turn_tag_state_from_history()
            self._update_system_sink_len_from_ids()

            latency = self._generation_timer.finish()
            sanitized = self._sanitize_response(raw_text)

            if not is_decision:
                continue

            response_hits = self._parse_response_lines(sanitized)
            if response_hits:
                for tid, answer in response_hits.items():
                    self._append_response_event(
                        events,
                        t=response_time,
                        value=self._normalize_silence_output(answer),
                        raw_text=raw_text,
                        latency=latency,
                        turn_id=tid,
                    )
            else:
                self._append_response_event(
                    events,
                    t=response_time,
                    value=self._normalize_silence_output(sanitized),
                    raw_text=raw_text,
                    latency=latency,
                )

            self._log(
                f"[MMDuet2] mode={mode} t={response_time:.3f}s tick={tick} frame={frame_idx} "
                f"response={sanitized!r} latency={latency:.3f}s"
            )

        events.sort(
            key=lambda x: (x.get("time", 0.0), 0 if x.get("type") == "question" else 1)
        )
        return events
