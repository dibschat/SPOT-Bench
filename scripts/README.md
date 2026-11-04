# Scripts

Keep annotations and videos under `data/`, following the layout described in [`data/README.md`](../data/README.md) .

## Inference

### StreamingVLM

```bash
conda activate streamingvlm
export STREAMING_VLM_ROOT=/path/to/streaming-vlm

python inference.py --task PNR --model StreamingVLM --result_dir results/StreamingVLM
```

### MMDuet2

```bash
conda activate streamingvlm
export MMDUET2_ROOT=/path/to/MMDuet2

python inference.py --task PNR --model MMDuet2 --kv_mode kvflush --result_dir results/MMDuet2
# --kv_mode also accepts `original` and `streamingvlm`
```

### Qwen-VL

```bash
conda activate spotbench

python inference.py --task PNR --model QwenVL --qwen_version 3 --result_dir results/QwenVL_3
```

### Ctrl-SPOT

```bash
conda activate spotbench

python inference.py --task PNR --model CtrlSPOT --result_dir results/CtrlSPOT
```

### Mage-VL

```bash
conda activate magevl

# QwenVL protocol (4 s window, 8 frames)
python inference.py --task PNR --model MageVL --result_dir results/MageVL

# codec-native input: causal H.264 copies first, then the window W (s) and canvas count
bash scripts/transcode_causal.sh ABD PNR
MAGE_CODEC_WINDOW=4 MAGE_CODEC_CANVAS=8 \
  python inference.py --task PNR --model MageVLCodec --result_dir results/MageVLCodec_w4
```

`scripts/run_sharded.sh` runs any model with one shard per GPU and merges the shards:
`MODEL=MageVL bash scripts/run_sharded.sh ABD PNR`.

### MiniCPM-o

```bash
conda activate minicpmo

python inference.py --task PNR --model MiniCPM_o --result_dir results/MiniCPM_o
```

### JoyAI-VL

Start the service once using the `joyai` environment from the JoyAI-VL-Interaction repository:

```bash
conda activate joyai
KEEP_QA_HISTORY=false MAIN_GPU=0,1 SUMMARY_GPU=2 ./serve_joyaivl.sh
```

Then run inference as usual:

```bash
conda activate spotbench
export JOYAI_VL_API_BASE=http://127.0.0.1:8070/v1

python inference.py --task PNR --model JoyAI_VL --result_dir results/JoyAI_VL
```

To run inference on all six tasks at once:

```bash
for task in ABD PNR SQA SPG SI UI; do
  python inference.py --task $task --model StreamingVLM \
    --result_dir results/StreamingVLM --resume
done
```

`--resume` picks up from the last completed video

## Scoring

Scoring reads the generated traces and does not require a GPU.

```bash
conda activate spotbench

# Detection only, for the leaderboard
python score.py --tasks ABD PNR --model StreamingVLM \
  --result_dir results/StreamingVLM

# Everything
export OPENAI_API_KEY=...   # SQA, SPG, SI, UI are judged by an LLM
python score.py --tasks ABD PNR SQA SPG SI UI --model StreamingVLM \
  --result_dir results/StreamingVLM

```

## Edge-trigger controller

`scripts/controller.py` drops replies that repeat the model's previous output for the same turn. It is causal
and works on any model's trace. `tune` selects the cooldown on a fixed 20% dev split and reports the held-out
80% and the full set; `apply` writes a controlled trace to score as usual. See [docs/mage_vl.md](../docs/mage_vl.md).

```bash
python scripts/controller.py tune  --task PNR --model MageVL --result_dir results/MageVL
python scripts/controller.py apply --task PNR --model MageVL --result_dir results/MageVL --out_dir results/MageVL_ctrl
python score.py --tasks PNR --model MageVL --result_dir results/MageVL_ctrl
```
