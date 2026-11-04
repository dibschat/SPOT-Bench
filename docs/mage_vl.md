# Mage-VL on SPOT-Bench (ABD, PNR)

[Mage-VL](https://huggingface.co/microsoft/Mage-VL) is a 4B codec-native video LLM (Mage-ViT + Qwen3-4B).
We evaluate it on the two fully released detection tasks, first with the harness of the existing offline
baselines, then with a simple controller on top.

## Results

T-F1@5 (%) at 1 Hz, the leaderboard metric. Rank is among the current leaderboard entries.

| | Setting | ABD | PNR | Overall | Rank |
| --- | --- | --- | --- | --- | --- |
| **Before** | Mage-VL, QwenVL offline harness | 10.9 | 10.2 | 10.6 | 4 / 8 |
| **After** | Mage-VL + edge-trigger controller | **28.6** | **20.5** | **24.6** | **1 / 8** |
| | Mage-VL (codec input, 1/4 visual tokens) + edge-trigger controller | 26.8 | 19.2 | 23.0 | 1 / 8 |

Current leaderboard for reference: Ctrl-SPOT 22.2 / 19.5 / 20.9, JoyAI-VL-Interaction 17.0 / 15.6 / 16.3,
MiniCPM-o 4.5 15.4 / 14.3 / 14.9, MMDuet2 14.2 / 5.1 / 9.7, Qwen2.5-VL 10.0 / 8.1 / 9.1, Qwen3-VL 10.7 / 7.2 / 9.0,
StreamingVLM 3.9 / 2.5 / 3.2.

**Before.** With the exact harness of the Qwen-VL baselines (`model_wrappers/qwen_vl.py`: same prompts,
4 s window at 2 FPS, one decision per second), Mage-VL is slightly above Qwen2.5-VL-7B and Qwen3-VL-8B
(10.6 vs 9.1 / 9.0). Its timing is good: PNR T-Score 70.1, the highest of all entries (next: 43.8).
T-F1 is low because it repeats its reply at every tick while an event is in its window: ~14k false
positives per task.

**After.** The edge-trigger controller keeps a reply only when it changes the model's previous output for
that question (`no → now`); repeats are dropped. This lifts T-F1 from 10.6 to 24.6.

**How to read the "after" rows.** They are a model plus a controller, the same kind of entry as
Ctrl-SPOT (a training-free controller around Qwen2.5-VL-7B). They do not show that Mage-VL alone
beats the offline Qwen rows, which were not run with the controller and would likely gain from it too.

## The controller

`scripts/controller.py`, ~15 lines of logic, works on any model's trace:

* **Rule.** For each question, keep a reply if it differs from the model's previous output for that
  question, or if at least *C* seconds passed since the last kept reply with the same value.
* **Causal.** It reads only the model's own earlier outputs, never later frames or annotations, so it
  runs online.
* **Not tuned on the reported set.** *C* is chosen on a fixed dev split (videos with `md5(id) % 5 == 0`,
  20%) from {∞, 30, 15, 8, 5, 3, 2} s. The dev split selects *C* = ∞, the parameter-free edge trigger.
  The held-out 80% scores within 0.3 points of the full set.
* **Cost.** Recurring actions are reported once while the output stays `now`, so T-Score drops
  (ABD 41.0 → 30.7, PNR 70.1 → 32.8) and PNR misses rise (215 → 549), while false positives fall by
  ~86%. Report both metrics.
* **vs Ctrl-SPOT.** Ctrl-SPOT feeds the model its earlier replies in the prompt and asks it not to repeat
  them, alongside a KV-cache memory with retrieval and summaries. The edge trigger enforces
  no-repeat on the outputs, with no memory.

## Codec input

`MageVLCodec` keeps the harness (prompts, decision times, query bookkeeping) and replaces the 8 frames
with Mage-VL's codec-native input: the last *W* seconds go through its HEVC codec processor, which
keeps anchor-frame patches and only the motion-salient patches of other frames.

| Input (4 s window) | Frames | Visual tokens¹ | T-F1 ABD / PNR / Overall | T-F1 + controller |
| --- | --- | --- | --- | --- |
| frames (harness default) | 8 | 4,464 | 10.9 / 10.2 / 10.6 | 28.6 / 20.5 / 24.6 |
| codec, 8 canvases | ≤ 64 | 1,152 | 10.2 / 10.1 / 10.2 | 26.8 / 19.2 / 23.0 |

¹ For a 1920×1080 video.

The codec input reaches about the same score with a quarter of the visual tokens and 8× the frames,
and raises T-Score (ABD 41.0 → 42.1, PNR 70.1 → 72.2). It does not raise T-F1. An 8 s codec window
raises T-Score further (PNR 78.3) but lowers T-F1 (9.1 raw, 20.6 with the dev-selected controller): the event
stays in the window longer, so the model repeats itself longer.

* **Causal.** Windows are cut with `-c copy` from H.264 copies made by
  `scripts/transcode_causal.sh`. It uses x264 `zerolatency` (no B-frames, no lookahead), a keyframe
  every second and a short side of 540. A cut never contains a frame later than the decision time.
* **Latency.** GPU generation drops from 0.27 s to 0.12 s per decision. Cutting and codec
  preprocessing add 0.82 s of CPU per decision (prefetched, so `score.py`'s latency does not include
  it). End to end, a codec decision currently takes longer than a frame decision because the codec
  is re-run on every overlapping window.

## Detailed numbers

T-Score@5 / T-F1@5 (%), full sets (ABD: 322 videos, 2,340 slots; PNR: 286 videos, 1,234 slots).

| Setting | ABD | PNR | FP (ABD / PNR) | FN (ABD / PNR) | Gen. latency |
| --- | --- | --- | --- | --- | --- |
| frames | 41.0 / 10.9 | 70.1 / 10.2 | 14,103 / 14,666 | 1,274 / 215 | 0.26 / 0.29 s |
| frames + controller | 30.7 / 28.6 | 32.8 / 20.5 | 1,773 / 2,032 | 1,435 / 549 | |
| codec 4 s | 42.1 / 10.2 | 72.2 / 10.1 | 15,934 / 15,409 | 1,248 / 219 | 0.13 / 0.12 s |
| codec 4 s + controller | 31.0 / 26.8 | 34.9 / 19.2 | 2,170 / 2,577 | 1,419 / 553 | |
| codec 8 s | 45.5 / 8.9 | 78.3 / 9.3 | 20,384 / 18,535 | 1,177 / 184 | 0.14 / 0.12 s |
| codec 8 s + controller | 30.7 / 25.9 | 59.0 / 15.2² | 2,279 / 7,280 | 1,418 / 197 | |

Controller on the dev split / held-out 80% (cooldown chosen on dev):

| Setting | Task | Chosen *C* | Dev | Held-out 80% | Full |
| --- | --- | --- | --- | --- | --- |
| frames | ABD | ∞ | 30.9 / 29.5 | 30.6 / 28.4 | 30.7 / 28.6 |
| frames | PNR | ∞ | 34.9 / 21.8 | 32.3 / 20.2 | 32.8 / 20.5 |
| codec 4 s | ABD | ∞ | 31.0 / 27.5 | 31.1 / 26.6 | 31.0 / 26.8 |
| codec 4 s | PNR | ∞ | 36.8 / 19.8 | 34.5 / 19.1 | 34.9 / 19.2 |
| codec 8 s | ABD | ∞ | 29.9 / 25.7 | 30.9 / 26.0 | 30.7 / 25.9 |
| codec 8 s | PNR | 3 s | 61.4 / 16.5 | 58.5 / 15.0 | 59.0 / 15.2 |

² The dev split selects *C* = 3 s here; with *C* = ∞ it would be 29.2 / 16.6. We report the dev choice.

## Reproducing

Mage-VL needs `transformers>=5.7` (the Qwen-VL baselines pin `<5`; `qwen_vl.py` falls back to
`AutoModelForImageTextToText` when `AutoModelForVision2Seq` is missing). The codec input needs `ffmpeg`
and `cv-preinfer` from [`codec-video-prep`](https://pypi.org/project/codec-video-prep/)
(`CV_PREINFER_BIN` if not on `PATH`). The model revision is pinned to
`d88b153285f1633a61b2f693c59c8576693af185`.

```bash
# before: the QwenVL harness with Mage-VL
python inference.py --task ABD --model MageVL --result_dir results/MageVL
# (8 GPUs: MODEL=MageVL bash scripts/run_sharded.sh ABD PNR)

# after: choose the cooldown on dev, apply, score
python scripts/controller.py tune  --task ABD --model MageVL --result_dir results/MageVL
python scripts/controller.py apply --task ABD --model MageVL --result_dir results/MageVL --out_dir results/MageVL_ctrl
python score.py --tasks ABD --model MageVL --result_dir results/MageVL_ctrl

# codec input (4 s window, 8 canvases)
bash scripts/transcode_causal.sh ABD PNR
MAGE_CODEC_WINDOW=4 MAGE_CODEC_CANVAS=8 \
  python inference.py --task ABD --model MageVLCodec --result_dir results/MageVLCodec_w4
```

## Caveats

* **Frame timestamps.** Mage-VL's processor stamps pre-decoded frames `<0.0 seconds>`, `<1.0 seconds>`, …,
  so the 8 window frames at 2 FPS are labelled 0–7 s instead of 0–3.5 s. The Qwen baseline uses the true
  times; the prompts ask about the last frame, not absolute times. The codec input uses true times.
* **No streaming gate.** Mage-VL's built-in cognition gate is query-agnostic (trained for commentary);
  ABD and PNR are query-conditioned, so it is not used.
* **Dev split.** The detection tasks have no validation set, so the dev split is taken from the
  benchmark itself.
* **Detection only.** SQA, SPG, SI and UI were not run.
