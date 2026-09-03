# SPOT-Bench

[![Webpage](https://img.shields.io/badge/Webpage-SPOT--Bench-1f883d?logo=googlechrome&style=flat-square)](https://dibschat.github.io/SPOT-Bench)
[![arXiv](https://img.shields.io/badge/arXiv-2604.24317-b31b1b?logo=arxiv&style=flat-square)](https://arxiv.org/abs/2604.24317)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-SPOT--Bench-ffd21e?style=flat-square&logo=huggingface)](https://huggingface.co/datasets/cvml-nus/spot-bench)

A fully **proactive** benchmark for **streaming video models** and **interaction models**.

This repository provides official implementation of:

> **Don't Pause! Every prediction matters in a streaming video** \
> Dibyadip Chatterjee, Zhanzhong Pang, Fadime Sener, Yale Song and Angela Yao

### ⚡ TL;DR
SPOT-Bench requires a streaming model to monitor a live video stream and proactively decide when to respond. Every prediction across the full video is evaluated using the Timeliness-F1 metric.*

> [!NOTE]
> **Detection** tasks (ABD, PNR) are released in full. For **Interaction** (SQA, SPG) and
> **Intervention** (SI, UI) we release a representative validation set while holding out the test set for an upcoming challenge.
> Stay tuned for updates on our [webpage](https://dibschat.github.io/SPOT-Bench/).
> **The Detection Leaderboard is [live](https://dibschat.github.io/SPOT-Bench/#Leaderboard)! Submit a pull request or contact us at dibyadip@u.nus.edu with a link to your paper to be featured on the leaderboard.**


### Release Plan

- [ ] Release challenge for Interaction and Intervention held-out test splits
- [ ] Update paper on arXiv *(Coming Soon)*
- [ ] Add Ctrl-SPOT (our updated baseline) and MiniCPM-o 4.5 *(Coming Soon)*
- [X] Add JoyAI-VL-Interaction to the baselines
- [X] Release streaming inference code
- [X] Release SPOT-Bench on HuggingFace
- [X] Project page live
- [X] Release paper on arXiv

## Tasks

| Family       | Task          | Description                                                                    |
| ------------ | ------------- | ------------------------------------------------------------------------------ |
| Detection    | **ABD** | Action Boundary Detection: detect when a queried action **start**s or **end**        |
| Detection    | **PNR** | Point-of-no-return Detection: detect the critical moment when a queried action takes effect    |
| Interaction  | **SQA** | Streaming Question Answering: answer once sufficient visual evidence becomes available |
| Interaction  | **SPG** | Streaming Procedural Guidance — provide the next step as the current step nears completion |
| Intervention | **SI**  | Solicited Intervention — intervene when the user hesitates or makes an error while performing a known task |
| Intervention | **UI**  | Unsolicited Intervention — detect imminent risks, intervene and warn the user |

## Benchmark Results

Each cell reports **T-Score@5 / T-F1@5** (%).

| Model | #Frames | PT? | Model Size | ABD | PNR | SQA | SPG | SI | UI | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL | 8 | X | 7B | 41.8/10.0 | 31.7/8.1 | 5.7/3.9 | 2.9/3.4 | 0.0/0.0 | 0.0/0.0 | 13.7/4.2 |
| Qwen3-VL | 8 | X | 8B | 40.5/10.7 | 43.8/7.2 | 5.7/4.0 | 2.2/2.7 | 0.0/0.0 | 0.4/0.6 | 15.4/4.2 |
| StreamingVLM | 2fps | ✅ | 7B | 4.8/3.9 | 4.4/2.5 | 22.8/1.2 | 12.9/1.3 | 13.2/0.7 | 14.8/4.4 | 12.2/2.3 |
| MMDuet2 + KVflush | 2fps | ✅ | 3B | 13.9/14.2 | 4.1/5.1 | 19.5/5.9 | 1.6/2.0 | 0.0/0.0 | 20.0/8.4 | 9.9/5.9 |
| MMDuet2 + StreamingVLM | 2fps | ✅ | 3B | 11.5/13.8 | 0.9/1.6 | 13.7/3.9 | 2.9/4.0 | 2.5/4.5 | 22.4/9.4 | 9.0/6.2 |
| MiniCPM-o 4.5* | 2fps | ✅ | 9B | | | | | | | |
| JoyAI-VL-Interaction | 2fps | ✅ | 8B+4B | 15.3/17.0 | 32.2/15.6 | 21.2/5.9 | 8.1/6.9 | 5.0/2.3 | 22.8/10.2 | 17.4/9.7 |

\*MiniCPM-o 4.5 results coming soon.

## Data

SPOT-Bench videos and annotations are available on [HuggingFace](https://huggingface.co/datasets/cvml-nus/spot-bench). 
Place the annotations and the unzipped `videos` folder under `data/`. The directory structure should look like:

```text
data/
  abd.json
  pnr.json
  sqa.json
  spg.json
  si.json
  ui.json
  videos/
    0000125.mp4
    0000172.mp4
    ...
```

## Installation

Clone the two baseline repos anywhere:

```bash
git clone https://github.com/mit-han-lab/streaming-vlm.git
git clone https://github.com/yellow-binary-tree/MMDuet2.git
```

#### StreamingVLM \& MMDuet2

Follow [StreamingVLM](https://github.com/mit-han-lab/streaming-vlm)'s own install instructions.

```bash
conda activate streamingvlm
export STREAMING_VLM_ROOT=/path/to/streaming-vlm
export MMDUET2_ROOT=/path/to/MMDuet2
```

#### JoyAI-VL-Interaction

```bash
conda create -n joyai python=3.12 -y
conda activate joyai

pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu129
pip install https://github.com/vllm-project/vllm/releases/download/v0.22.0/vllm-0.22.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl
pip install pillow requests
```

#### Scoring + Qwen-VL

```bash
conda create -n spotbench python=3.11 -y
conda activate spotbench

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -U setuptools wheel packaging ninja
pip install flash-attn==2.8.3 --no-build-isolation
pip install "transformers>=4.57,<5" accelerate qwen-vl-utils decord tqdm pillow requests openai
```

See `scripts/` for instructions on running inference and scoring.

## Adding a model

This repo is designed to be model-agnostic: annotations, evaluation metrics, and model wrappers are fully decoupled. 
Adding a new streaming model requires implementing only a single inference method.

1. Subclass `ModelStreaming` in `model_wrappers/` and implement `inference()`, returning a
   flat list of timestamped responses.
2. Register your model in `MODEL_REGISTRY` inside
   [`model_wrappers/__init__.py`](model_wrappers/__init__.py).

All inference and scoring are handled by the base class. Set `SPOTBENCH_VERBOSE=1` to enable per-tick diagnostics.

## Reproducing Table 1: Online VideoQA Baselines

Table 1 of the paper reports Qwen2.5-VL and Qwen3-VL results on OVO-Bench and StreamingBench under three settings: **blind** (no visual input), **single frame** (query frame only), and **four recent frames** (4 frames sampled at 1 fps from the most recent 4 seconds). These are simple offline baselines that demonstrate performing well on retrospective benchmarks does not require streaming the entire video.

Download [StreamingBench](https://github.com/thunlp-mt/streamingbench) and [OVO-Bench](https://github.com/joeleelyf/ovo-bench) from their respective repositories. We use the default prompts provided by each benchmark without modification. Both models share the same inference setup:

```python
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info

# model_name = "Qwen2.5-VL-7B-Instruct"
model_name = "Qwen3-VL-8B-Instruct"

model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_name)

# frames passed as individual images, not as a video - uses Qwen's multi-image setting
# frames: [] (blind), [f1] (single frame), or [f1, f2, f3, f4] (4-frame setting)
content = [{"type": "image", "image": Image.fromarray(frame, "RGB")} for frame in frames]

messages = [{"role": "user", "content": content + [{"type": "text", "text": prompt}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
output = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0].strip()
```

## Citation

If you find our work useful, please cite:

```bibtex
@article{chatterjee2026don,
  title={Don't Pause! Every prediction matters in a streaming video},
  author={Chatterjee, Dibyadip and Pang, Zhanzhong and Sener, Fadime and Song, Yale and Yao, Angela},
  journal={arXiv preprint arXiv:2604.24317},
  year={2026}
}
```

## Acknowledgements

Our evaluation code builds upon the well-organized repositories of [StreamingBench](https://github.com/thunlp-mt/streamingbench) and [OVO-Bench](https://github.com/joeleelyf/ovo-bench). We also thank the authors of [StreamingVLM](https://github.com/mit-han-lab/streaming-vlm), [MMDuet2](https://github.com/yellow-binary-tree/MMDuet2) and [JoyAI-VL-Interaction](https://github.com/jd-opensource/JoyAI-VL-Interaction) for their excellent open-source releases.
