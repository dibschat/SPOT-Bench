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
