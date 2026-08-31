# Data

SPOT-Bench annotations and videos are distributed on Hugging Face: **[cvml-nus/spot-bench](https://huggingface.co/datasets/cvml-nus/spot-bench)**

## Download

```bash
pip install -U "huggingface_hub[cli]"
hf auth login

hf download cvml-nus/spot-bench --repo-type dataset --local-dir data
unzip data/videos.zip -d data && rm data/videos.zip
```

## Expected layout

Everything in this repository defaults to `data/` for annotations and `data/videos/` for videos, so if the tree looks like this no path flags are needed:

```text
data/
  abd.json      # Action Boundary Detection       322 videos
  pnr.json      # Point-of-No-Return Detection    286 videos
  sqa.json      # Streaming Question Answering     15 videos
  spg.json      # Streaming Procedural Guidance    13 videos
  si.json       # Solicited Intervention           13 videos
  ui.json       # Unsolicited Intervention         13 videos
  videos/       # 662 MP4 files, ~40 GB
```
