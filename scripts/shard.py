"""Shard / merge SPOT-Bench annotations and results for multi-GPU runs (see run_sharded.sh).

Videos are spread over the shards longest-first by file size, so shards finish together.

  python scripts/shard.py shard ABD 8 results/MageVL
  python scripts/shard.py merge ABD 8 results/MageVL MageVL
"""
import json
import os
import sys

from tasks import annotation_path, task_key

cmd, task, n, run_dir = sys.argv[1], sys.argv[2].upper(), int(sys.argv[3]), sys.argv[4]
key = task_key(task)

if cmd == "shard":
    ann = json.load(open(annotation_path("data", task)))
    size = lambda vid: os.path.getsize(f"data/videos/{vid.replace('.mp4', '')}.mp4")
    shards, loads = [{} for _ in range(n)], [0] * n
    for vid in sorted(ann, key=size, reverse=True):  # greedy longest-first
        i = loads.index(min(loads))
        shards[i][vid] = ann[vid]
        loads[i] += size(vid)
    for i, shard in enumerate(shards):
        os.makedirs(f"{run_dir}/shard{i}", exist_ok=True)
        json.dump(shard, open(f"{run_dir}/shard{i}/{key}.json", "w"))
        print(f"shard{i}: {len(shard)} videos, {loads[i] / 1e9:.1f} GB")
elif cmd == "merge":
    model = sys.argv[5]
    merged = {}
    for i in range(n):
        merged.update(json.load(open(f"{run_dir}/shard{i}/{key}_{model}.json")))
    expected = len(json.load(open(annotation_path("data", task))))
    assert len(merged) == expected, f"{len(merged)} merged vs {expected} annotated"
    json.dump(merged, open(f"{run_dir}/{key}_{model}.json", "w"), indent=4)
    print(f"merged {len(merged)} videos -> {run_dir}/{key}_{model}.json")
