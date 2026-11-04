"""Ctrl-SPOT with a Mage-VL backbone.

The controller (prompts, short/long-term memory, retrieval, summaries, query bookkeeping,
decision loop) is `CtrlSPOT` unchanged. Only the backbone-specific parts are replaced:
Mage-VL's decoder is Qwen3 with 1D RoPE (instead of Qwen2.5-VL's 3D M-RoPE), frames go
through its image path (`pixel_values`, `image_grid_thw`, `patch_positions`), and its
vision patches are 16 px with a 2x2 merge (a 32-px grid).
"""

import sys
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3 import modeling_qwen3 as q3

from .ctrl_spot import _CAPTURE, CtrlSPOT, Segment
from .mage_vl import MAGE_VL_REPO, MAGE_VL_REVISION

_ORIG_APPLY_ROTARY_Q3 = q3.apply_rotary_pos_emb


def _capturing_apply_rotary_q3(q, k, cos, sin, *args, **kwargs):
    if _CAPTURE["on"]:
        _CAPTURE["keys"].append(k)
    return _ORIG_APPLY_ROTARY_Q3(q, k, cos, sin, *args, **kwargs)


q3.apply_rotary_pos_emb = _capturing_apply_rotary_q3


class CtrlSPOTMageVL(CtrlSPOT):
    FRAME_BLOCK = 32

    def _model_init(self):
        if self.model_path == "Qwen/Qwen2.5-VL-7B-Instruct":  # CtrlSPOT's default
            self.model_path = MAGE_VL_REPO
        revision = MAGE_VL_REVISION if self.model_path == MAGE_VL_REPO else None
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, revision=revision, dtype=torch.bfloat16, device_map={"": device},
            attn_implementation="sdpa", trust_remote_code=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path, revision=revision, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.device = next(self.model.parameters()).device

        cfg = self.model.config
        self.num_layers = int(cfg.text_config.num_hidden_layers)
        self.video_token_id = int(cfg.image_token_id)  # frames use the image path
        self.vision_start_id = int(cfg.vision_start_token_id)
        self.vision_end_id = int(cfg.vision_end_token_id)
        self.merge = int(cfg.vision_config.spatial_merge_size)
        self.rotary = self.model.model.language_model.rotary_emb
        self.build_patch_positions = sys.modules[type(self.processor.video_processor).__module__].build_patch_positions

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
            tensor = getattr(out, "last_hidden_state", None)
            if tensor is None:
                tensor = out[0] if isinstance(out, (tuple, list)) else out
            self._last_visual = tensor.detach() if torch.is_tensor(tensor) else None

        self.model.model.visual.register_forward_hook(_hook)
        print(f"[CtrlSPOT] Mage-VL from {self.model_path} on {self.device}")

    def _rotate(self, keys: List[torch.Tensor], pos: torch.Tensor) -> List[torch.Tensor]:
        cos, sin = self.rotary(keys[0], pos)
        return [_ORIG_APPLY_ROTARY_Q3(k, k, cos, sin)[1] for k in keys]

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
        pos = torch.arange(P + C, device=self.device).unsqueeze(0)  # 1D RoPE

        cache = DynamicCache()
        if P > 0:
            keys = [
                torch.cat([s.keys[l].to(self.device) for s in prefix], dim=2)
                for l in range(self.num_layers)
            ]
            rotated = self._rotate(keys, pos[:, :P])
            del keys
            for l in range(self.num_layers):
                values = torch.cat([s.values[l].to(self.device) for s in prefix], dim=2)
                cache.update(rotated[l], values, l)
            del rotated

        inputs = {
            "input_ids": chunk.unsqueeze(0).to(self.device),
            "attention_mask": torch.ones((1, P + C), dtype=torch.long, device=self.device),
            "position_ids": pos[:, P:],
            "cache_position": torch.arange(P, P + C, device=self.device),
        }
        if pixel_values is not None:
            inputs["pixel_values"] = pixel_values.to(self.device, self.model.dtype)
            inputs["image_grid_thw"] = grid.to(self.device)
            inputs["patch_positions"] = self.build_patch_positions(grid, spatial_merge_size=self.merge).to(self.device)

        _CAPTURE["on"], _CAPTURE["keys"] = True, []
        try:
            with torch.no_grad():
                out = self.model(**inputs, past_key_values=cache, use_cache=True, logits_to_keep=1)
        finally:
            captured = _CAPTURE["keys"]
            _CAPTURE["on"], _CAPTURE["keys"] = False, []
        if len(captured) != self.num_layers:
            raise RuntimeError(f"[CtrlSPOT] captured {len(captured)} key tensors, expected {self.num_layers}")
        return out, out.past_key_values, captured, pos

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
                    position_ids=torch.tensor([[next_pos + step]], device=self.device),
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

    def _encode_frame(self, image):
        enc = self.processor.image_processor(images=[image], return_tensors="pt")
        grid = enc["image_grid_thw"]
        n_tokens = int(grid[0, 0] * grid[0, 1] * grid[0, 2]) // (self.merge * self.merge)
        return enc["pixel_values"], grid, n_tokens
