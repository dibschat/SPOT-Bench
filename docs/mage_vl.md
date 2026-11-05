# Mage-VL on SPOT-Bench (ABD, PNR)

[Mage-VL](https://huggingface.co/microsoft/Mage-VL) (4B, Mage-ViT + Qwen3-4B) in the two existing harnesses,
with only the backbone swapped.

## Results

T-F1@5 (%) at 1 Hz.

| Harness | Model | ABD | PNR | Overall |
| --- | --- | --- | --- | --- |
| Ctrl-SPOT | Ctrl-SPOT baseline (current #1) | 22.2 | 19.5 | 20.9 |
| Ctrl-SPOT | Mage-VL | 33.4 | 8.5 | 21.0 |
| Ctrl-SPOT + `--no_repeat` | **Mage-VL** | **34.2** | 17.0 | **25.6** |
| QwenVL | Mage-VL | 10.9 | 10.2 | 10.6 |
| QwenVL + `--no_repeat` | Mage-VL | 27.8 | **20.5** | 24.2 |
| QwenVL + `--no_repeat` | Mage-VL codec (1/4 visual tokens) | 25.9 | 19.2 | 22.6 |

*Harness* is the existing inference wrapper in this repo: `model_wrappers/ctrl_spot.py` (Ctrl-SPOT) or
`model_wrappers/qwen_vl.py` (the offline QwenVL baseline). Only the backbone is swapped.

## What changed

* **Backbone.** `MageVL` (QwenVL harness) and `CtrlSPOTMageVL` (Ctrl-SPOT harness) subclass the existing
  wrappers and replace only model loading, frame encoding (Mage-ViT's 32-px grid) and, for Ctrl-SPOT, the key
  capture and re-rotation for Qwen3's 1D RoPE. Prompts, memory, retrieval and the decision loop are unchanged.
  The Ctrl-SPOT port reproduces a plain full forward (logit cosine 0.9999).
* **`--no_repeat`** (an `inference.py` flag, both harnesses): a reply is not written to the result when it is
  identical to the model's previous output for the same question (`no` included). The model, prompts, decisions
  and Ctrl-SPOT's memory are unchanged, and it has no parameter. Mage-VL keeps answering `now` while an event
  is in view; the flag removes these repeats, which cuts false positives but reports a recurring action only
  once, so T-Score drops.
* **Codec.** `MageVLCodec` replaces the 8 frames of the QwenVL harness (4,464 visual tokens for 1080p) with
  Mage-VL's codec input over the same 4 s window: up to 64 frames in 1,152 tokens. Windows are cut from causal
  H.264 copies (`scripts/transcode_causal.sh`: no B-frames or lookahead, a keyframe per second), so no input
  frame is later than the decision time.

## Details

T-Score@5 / T-F1@5 (%), full sets (ABD 322 videos, PNR 286 videos).

| Setting | ABD | PNR | FP (ABD / PNR) |
| --- | --- | --- | --- |
| Ctrl-SPOT, Mage-VL | 33.2 / 33.4 | 80.1 / 8.5 | 1,332 / 20,898 |
| Ctrl-SPOT `--no_repeat`, Mage-VL | 32.8 / 34.2 | 39.2 / 17.0 | 1,164 / 3,694 |
| QwenVL, Mage-VL | 41.0 / 10.9 | 70.1 / 10.2 | 14,103 / 14,666 |
| QwenVL `--no_repeat`, Mage-VL | 30.7 / 27.8 | 32.8 / 20.5 | 1,921 / 2,032 |
| QwenVL `--no_repeat`, Mage-VL codec | 31.0 / 25.9 | 34.9 / 19.2 | 2,348 / 2,577 |

## Reproducing

Mage-VL needs `transformers>=5.7` (`qwen_vl.py` falls back to `AutoModelForImageTextToText` when
`AutoModelForVision2Seq` is missing); the codec input needs `ffmpeg` and `cv-preinfer` from
[`codec-video-prep`](https://pypi.org/project/codec-video-prep/). Model revision
`d88b153285f1633a61b2f693c59c8576693af185`.

```bash
python inference.py --task PNR --model CtrlSPOTMageVL [--no_repeat] --result_dir results/CtrlSPOTMageVL
python inference.py --task PNR --model MageVL [--no_repeat] --result_dir results/MageVL

bash scripts/transcode_causal.sh ABD PNR
MAGE_CODEC_WINDOW=4 MAGE_CODEC_CANVAS=8 \
  python inference.py --task PNR --model MageVLCodec --no_repeat --result_dir results/MageVLCodec

# 8 GPUs: MODEL=MageVL ARGS=--no_repeat bash scripts/run_sharded.sh ABD PNR
```

## Caveats

* Mage-VL's processor stamps pre-decoded frames 0, 1, 2, … s, so in the QwenVL harness the 8 frames at 2 FPS
  are labelled 0–7 s instead of 0–3.5 s (Ctrl-SPOT and the codec input use true times).
* The codec input adds 0.8 s of CPU per decision for cutting and codec preprocessing (prefetched; `score.py`'s
  latency is generation only).
* SQA, SPG, SI and UI were not run.
