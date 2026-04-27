# SPOT-Bench \& AsynKV

This repository provides official implementation of:
> **Don't Pause! Every prediction matters in a streaming video**  
>Dibyadip Chatterjee, Zhanzhong Pang, Fadime Sener, Yale Song and Angela Yao.  

[![Webpage](https://img.shields.io/badge/Webpage-SPOT--Bench-1f883d?logo=googlechrome&style=flat-square)](https://dibschat.github.io/SPOT-Bench)
[![arXiv](https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg?style=flat-square&logo=arxiv)](https://arxiv.org/abs/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Coming%20Soon-ffd21e?style=flat-square&logo=huggingface)](#)

Release Plan

- [ ] Release AsynKV inference code (Estimated: 14 days)
- [ ] Release SPOT-Bench on HuggingFace (Estimated: 7 days)
- [ ] Release StreamingVLM and MMDuet2 baselines (Estimated: 3 days)
- [ ] Release evaluation code (Estimated: 3 days)
- [x] Project page live
- [x] Release paper on arXiv

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
@article{chatterjee2025spotbench,
  title={Don't Pause! Every prediction matters in a streaming video},
  author={Chatterjee, Dibyadip and Pang, Zhanzhong and Sener, Fadime and Song, Yale and Yao, Angela},
  journal={arXiv preprint},
  year={2025}
}
```

## Acknowledgements
Our evaluation code builds upon the well-organized repositories of [StreamingBench](https://github.com/thunlp-mt/streamingbench) and [OVO-Bench](https://github.com/joeleelyf/ovo-bench). We also thank the authors of [StreamingVLM](https://github.com/mit-han-lab/streaming-vlm) and [MMDuet2](https://github.com/yellow-binary-tree/MMDuet2) for their excellent open-source releases.
