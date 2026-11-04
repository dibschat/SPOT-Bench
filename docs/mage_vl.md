# Mage-VL on SPOT-Bench (detection: ABD, PNR)

[Mage-VL](https://huggingface.co/microsoft/Mage-VL) is a 4B codec-native video LLM (Mage-ViT +
Qwen3-4B decoder). This page reports it on the two fully released detection tasks, in three
settings that build on each other:

1. **Mage-VL**: the offline sliding-window protocol of the QwenVL baseline, unchanged.
2. **Mage-VL (codec)**: the same decisions, but each one sees its window through Mage-VL's
   codec-native video input: many more frames for fewer visual tokens.
3. **+ edge-trigger controller**: a causal rule on the model's outputs that stops it from
   repeating the same reply every second.

All numbers are T-Score@5 / T-F1@5 (%) at 1 Hz on the full ABD (322 videos, 2,340 slots) and
PNR (286 videos, 1,234 slots) sets, produced with `score.py` of this repository.

## Summary

| Model | Visual input per decision | ABD | PNR | T-F1@5 Overall |
| --- | --- | --- | --- | --- |
| Mage-VL | 8 frames, last 4 s @ 2 FPS (4,464 tokens¹) | 41.0 / 10.9 | 70.1 / 10.2 | 10.6 |
| Mage-VL (codec) | ≤ 64 frames, last 4 s, 8 codec canvases (1,152 tokens¹) | 42.1 / 10.2 | 72.2 / 10.1 | 10.2 |
| Mage-VL + edge-trigger | as Mage-VL | 30.7 / **28.6** | 32.8 / **20.5** | **24.6** |
| Mage-VL (codec) + edge-trigger | as Mage-VL (codec) | 31.0 / 26.8 | 34.9 / 19.2 | 23.0 |

¹ For a 1920×1080 video; the count scales with the aspect ratio.

For reference, the leaderboard's other entries (T-F1@5 ABD / PNR / Overall): Ctrl-SPOT 22.2 / 19.5 /
20.9, JoyAI-VL-Interaction 17.0 / 15.6 / 16.3, MiniCPM-o 4.5 15.4 / 14.3 / 14.9, MMDuet2 14.2 / 5.1 /
9.7, Qwen2.5-VL (offline) 10.0 / 8.1 / 9.1, Qwen3-VL (offline) 10.7 / 7.2 / 9.0, StreamingVLM 3.9 /
2.5 / 3.2.

**What these numbers support.** Under the identical offline protocol, Mage-VL (4B) is slightly above
Qwen2.5-VL-7B and Qwen3-VL-8B on T-F1 (10.6 vs 9.1 / 9.0) and has by far the highest PNR T-Score of
all entries (70.1; the next is Qwen3-VL at 43.8). The
edge-trigger rows are a model *plus* a response controller and belong next to Ctrl-SPOT, which is
also a training-free controller around an MLLM. They do **not** show that Mage-VL alone beats the
offline Qwen rows: those were not re-run with the same controller, and would likely gain from it
as well (see [Fairness](#3-edge-trigger-controller)).

## 1. Mage-VL with the QwenVL protocol

`model_wrappers/mage_vl.py:MageVL` subclasses `QwenVL` and changes only model loading and
generation: same prompts, the same 4 s window at 2 FPS (8 frames, short side 560), one decision per
second, the same active-query bookkeeping (4 rounds, 120 s TTL), greedy decoding with 8 new tokens.
Frames are resized to the 32-px grid of Mage-ViT (16-px patches, 2×2 merge) instead of Qwen's 28.

| Task | T-Score@5 | T-F1@5 | FP | FN | T-F1@1 | Latency |
| --- | --- | --- | --- | --- | --- | --- |
| ABD | 41.0 | 10.9 | 14,103 | 1,274 | 5.5 | 0.259 s |
| PNR | 70.1 | 10.2 | 14,666 | 215 | 4.0 | 0.289 s |

The timing of the replies is good (T-Score; few misses), but T-F1 is low because the model keeps
answering `now` / `start` / `end` at every tick while the event is inside its 4 s window. On the
first PNR video, for instance, it answers `now` from 3 s to 14 s for five annotated moments.
Each repeat beyond the occupancy budget is a false positive (≈14k per task).

## 2. Codec-native input

`MageVLCodec` keeps the prompts, the 1 Hz decision times and the query bookkeeping, and changes the
visual input. At decision time *t* it cuts the window [*t* − *W*, *t*] and runs it through
Mage-VL's HEVC codec processor (`video_backend="codec"`), which keeps anchor-frame patches and only
the motion-salient patches of the other frames, packing up to `target_canvas / 4 × 32` frames into
`target_canvas` canvases of ≤150k pixels.

Causality. Windows are cut with `-c copy` from H.264 copies made by
`scripts/transcode_causal.sh`: x264 `-tune zerolatency` (no B-frames, no lookahead, so no frame is
encoded from a later one), a keyframe every second, short side 540. A cut may start up to one
keyframe before *t* − *W* but never contains a frame later than *t* (checked by counting the frames
of cuts against the source timestamps).

| Config | Window | Frames sampled | Visual tokens¹ | ABD | PNR | Overall T-F1 | Gen. latency | Codec prep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frames (§1) | 4 s | 8 | 4,464 | 41.0 / 10.9 | 70.1 / 10.2 | 10.6 | 0.27 s | – |
| codec, `target_canvas=8` | 4 s | ≤ 64 | 1,152 | 42.1 / 10.2 | 72.2 / 10.1 | 10.2 | 0.12 s | 0.82 s |
| codec, `target_canvas=8` | 8 s | ≤ 64 | 1,152 | 45.5 / 8.9 | 78.3 / 9.3 | 9.1 | 0.13 s | 0.99 s |

With a quarter of the visual tokens, the 4 s codec input matches the frame input (T-F1 −0.4 overall,
T-Score +1.1 / +2.1). The 8 s window has the best T-Score of all (PNR 78.3) but the lowest T-F1: an
event stays in the window twice as long, so the model repeats itself for longer.

Latency. `Latency` in `score.py` is the generation time only. The codec path additionally spends
0.82 s (4 s window) / 0.99 s (8 s window) of CPU per decision on cutting and codec preprocessing
(median over 15 windows, single process, idle node). The wrapper prefetches it in a thread pool, so
it does not show in the reported latency, but end to end a codec decision currently takes ≈0.9 s,
longer than a frame decision. This is a property of re-running `cv-preinfer` on every overlapping
window, not of the model; an incremental codec front end would remove most of it.

## 3. Edge-trigger controller

`scripts/controller.py` keeps a reply to a turn only when it differs from the model's previous
output for that turn (`no → now`, `start → end`, ...), or when at least *C* seconds have passed since
the last kept reply with the same value. It reads only the model's own earlier outputs, so it
runs online.

Fairness protocol.

* **Causal**: it never sees later frames, later outputs or annotations.
* **No tuning on the reported set**: the only parameter, *C*, is chosen by T-F1@5 on a fixed dev
  split (videos with `md5(id) % 5 == 0`, 20%) from {∞, 30, 15, 8, 5, 3, 2} s. The detection tasks
  have no separate validation set, so the dev split comes from the benchmark itself; results are
  reported on the held-out 80% and on the full set.
* **Chosen setting**: *C* = ∞ (a pure edge trigger, no parameter) for all frame and 4 s codec runs.

| Input | Task | Raw (full) | Dev, chosen *C* | Held-out 80% | Full |
| --- | --- | --- | --- | --- | --- |
| frames | ABD | 41.0 / 10.9 | ∞: 30.9 / 29.5 | 30.6 / 28.4 | 30.7 / 28.6 |
| frames | PNR | 70.1 / 10.2 | ∞: 34.9 / 21.8 | 32.3 / 20.2 | 32.8 / 20.5 |
| codec 4 s | ABD | 42.1 / 10.2 | ∞: 31.0 / 27.5 | 31.1 / 26.6 | 31.0 / 26.8 |
| codec 4 s | PNR | 72.2 / 10.1 | ∞: 36.8 / 19.8 | 34.5 / 19.1 | 34.9 / 19.2 |
| codec 8 s | ABD | 45.5 / 8.9 | ∞: 29.9 / 25.7 | 30.9 / 26.0 | 30.7 / 25.9 |
| codec 8 s | PNR | 78.3 / 9.3 | 3 s: 61.4 / 16.5² | 58.5 / 15.0 | 59.0 / 15.2 |

² For codec 8 s on PNR the dev split selects *C* = 3 s; with *C* = ∞ it would score 29.2 / 16.6 on
the full set. We report the dev choice.

The held-out numbers are within 0.3 points of the full set, so the rule is not fitted to the split.

The cost of the controller. T-F1 rises because false positives drop by ~86% (ABD 14,103 → 1,773;
PNR 14,666 → 2,032), but T-Score falls (ABD 41.0 → 30.7, PNR 70.1 → 32.8) and misses rise (PNR 215
→ 549): when an action recurs while the model's output stays `now`, only the first occurrence is
reported. Both metrics should be read together.

Relation to Ctrl-SPOT. Both are training-free controllers around an MLLM and target the same
failure. Ctrl-SPOT tells the model about its earlier replies in the prompt ("You replied to Q1 at
…", "never repeat a boundary you already replied to") and adds a KV-cache stream memory with
retrieval and summaries around Qwen2.5-VL-7B. The edge trigger is a ~15-line rule on the outputs,
with no memory, and works with any model's trace:

```bash
python scripts/controller.py apply --task ABD --model QwenVL --result_dir results/QwenVL_3
```

A like-for-like base-model comparison would apply it to the Qwen-VL traces too; we have not run
that.

## Reproducing

Environment: Mage-VL needs `transformers>=5.7` (the Qwen-VL baselines pin `<5`, so use a separate
environment); `qwen_vl.py` imports `AutoModelForImageTextToText` as `AutoModelForVision2Seq` when the
latter is missing. The codec path additionally needs `ffmpeg` and the `cv-preinfer` binary of
[`codec-video-prep`](https://pypi.org/project/codec-video-prep/) (`CV_PREINFER_BIN` if it is not on
`PATH`); see the [Mage-VL README](https://github.com/microsoft/Mage/tree/main/mage_vl).

```bash
conda create -n magevl python=3.12 -y && conda activate magevl
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==5.7.0" accelerate decord qwen-vl-utils tqdm pillow opencv-python-headless
pip install codec-video-prep==0.2.5   # codec path only; Mage-VL's docs keep it in its own env (NumPy<2)

# 1. frames, the QwenVL protocol
python inference.py --task ABD --model MageVL --result_dir results/MageVL
# or on 8 GPUs: MODEL=MageVL bash scripts/run_sharded.sh ABD PNR

# 2. codec input (4 s window, 8 canvases)
bash scripts/transcode_causal.sh ABD PNR                       # data/videos_causal/
MAGE_CODEC_WINDOW=4 MAGE_CODEC_CANVAS=8 \
  python inference.py --task ABD --model MageVLCodec --result_dir results/MageVLCodec_w4

# 3. edge-trigger controller: select C on dev, then apply and score
python scripts/controller.py tune  --task ABD --model MageVL --result_dir results/MageVL
python scripts/controller.py apply --task ABD --model MageVL --result_dir results/MageVL \
  --out_dir results/MageVL_ctrl
python score.py --tasks ABD --model MageVL --result_dir results/MageVL_ctrl
```

The model revision is pinned to `d88b153285f1633a61b2f693c59c8576693af185`. Runs used 8×H800 with
one shard per GPU (ABD and PNR concurrently, two processes per GPU).

## Caveats

* **Frame timestamps.** Mage-VL's processor stamps pre-decoded frames `<0.0 seconds>`,
  `<1.0 seconds>`, …, so the 8 window frames at 2 FPS are labelled 0–7 s instead of 0–3.5 s. The
  QwenVL baseline stamps the true times. The prompts ask about the last frame, not absolute times,
  but this is a difference from the baseline. The codec path uses the true times of the cut.
* **No streaming gate.** Mage-VL ships a query-agnostic cognition gate trained for event-gated
  commentary; ABD and PNR are query-conditioned, so it is not used here.
* **Detection only.** SQA, SPG, SI and UI were not run.
