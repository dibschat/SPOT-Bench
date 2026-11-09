#!/bin/bash
# Causal H.264 copies of the SPOT-Bench videos for the Mage-VL codec path: zerolatency
# (no B-frames / lookahead, so no frame references the future), a keyframe every
# second (so 1 s-aligned windows cut with -c copy), short side 540.
# usage: bash scripts/transcode_causal.sh TASK...   (from the repo root; JOBS=40 parallel encodes)
set -e
OUT=data/videos_causal
mkdir -p $OUT
python - "$@" <<'PY' | xargs -P ${JOBS:-40} -I{} bash -c '
  in=data/videos/{}.mp4; out='$OUT'/{}.mp4
  [ -s $out ] && exit 0
  ffmpeg -nostdin -loglevel error -y -i $in -an \
    -vf "scale=w='"'"'if(gt(iw,ih),-2,540)'"'"':h='"'"'if(gt(iw,ih),540,-2)'"'"'" \
    -c:v libx264 -preset veryfast -tune zerolatency -g 300 -sc_threshold 0 \
    -force_key_frames "expr:gte(t,n_forced*1)" -threads 4 $out.tmp.mp4 && mv $out.tmp.mp4 $out \
    || echo "FAILED {}"'
import json, sys
from tasks import annotation_path
vids = set()
for t in sys.argv[1:]:
    vids |= {v.replace(".mp4", "") for v in json.load(open(annotation_path("data", t.upper())))}
print("\n".join(sorted(vids)))
PY
echo "transcoded $(ls $OUT | grep -vc tmp) videos"
