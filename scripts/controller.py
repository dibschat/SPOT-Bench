"""Causal edge-trigger response controller for SPOT-Bench traces.

A streaming model queried every second tends to repeat the same reply ("now", "now",
...) for as long as an event stays in its window, and every repeat is scored as a false
positive. The controller keeps a reply to a turn only when it differs from the model's
previous output for that turn (e.g. no -> now), or when at least `--cooldown` seconds
have passed since the last kept reply with the same value. It reads nothing but the
model's own earlier outputs, so it can run online, inside the stream.

Apply with a fixed cooldown and score the result like any other trace:

    python scripts/controller.py apply --task ABD --model MageVL \
        --result_dir results/MageVL --out_dir results/MageVL_ctrl
    python score.py --tasks ABD --model MageVL --result_dir results/MageVL_ctrl

Select the cooldown on a fixed dev split (videos with md5(id) % 5 == 0, 20%) and report
it on the held-out rest and on the full set:

    python scripts/controller.py tune --task ABD --model MageVL --result_dir results/MageVL
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys

from tasks import annotation_path, task_key

DEFAULT_GRID = [math.inf, 30, 15, 8, 5, 3, 2]


def is_silent(text) -> bool:
    s = str(text).strip().lower()
    return not s or not any(ch.isalnum() for ch in s) or s in {
        "no", "none", "nothing", "no answer", "no response"
    }


def control(events, cooldown=math.inf):
    """Filter one video's event list; question events pass through unchanged."""
    prev, last_kept, out = {}, {}, []
    for e in sorted(events, key=lambda e: (e["time"], 0 if e["type"] == "question" else 1)):
        if e["type"] != "response":
            out.append(e)
            continue
        tid, value = e.get("turn_id", ""), str(e["value"]).strip().lower()
        changed = prev.get(tid) != value
        prev[tid] = value
        if is_silent(value):
            continue
        t_last = last_kept.get((tid, value))
        if changed or t_last is None or e["time"] - t_last >= cooldown:
            last_kept[(tid, value)] = e["time"]
            out.append(e)
    return out


def score(results, ann_fp, out_dir, model, task):
    """Write a trace and score it with score.py; returns (T-Score@5, T-F1@5, FP, FN)."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{task_key(task)}_{model}.json"), "w") as f:
        json.dump(results, f)
    log = subprocess.run(
        [sys.executable, "score.py", "--tasks", task, "--model", model,
         "--result_dir", out_dir, "--annotation_path", ann_fp],
        capture_output=True, text=True,
    ).stdout
    m = re.search(r"T-Score@5=([\d.]+)%, T-F1@5=([\d.]+)%, FP=(\d+), FN=(\d+)", log)
    return tuple(float(x) for x in m.groups()) if m else None


def dev_test_split(task, annotation_root, split_dir):
    """Fixed 20/80 split of a task's videos by md5 of the video id."""
    ann = json.load(open(annotation_path(annotation_root, task)))
    os.makedirs(split_dir, exist_ok=True)
    paths = {}
    for name, in_dev in (("dev", True), ("test", False)):
        fp = os.path.join(split_dir, f"{task_key(task)}_{name}.json")
        json.dump({v: e for v, e in ann.items()
                   if (int(hashlib.md5(v.encode()).hexdigest(), 16) % 5 == 0) == in_dev}, open(fp, "w"))
        paths[name] = fp
    paths["full"] = annotation_path(annotation_root, task)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["apply", "tune"])
    parser.add_argument("--task", required=True, metavar="{ABD,PNR,...}")
    parser.add_argument("--model", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--out_dir", default=None, help="apply: where to write the controlled trace")
    parser.add_argument("--cooldown", type=float, default=math.inf, help="apply: seconds (default: inf)")
    parser.add_argument("--grid", type=float, nargs="+", default=DEFAULT_GRID, help="tune: cooldowns to try")
    parser.add_argument("--annotation_root", default="data")
    args = parser.parse_args()

    task, key = args.task.upper(), task_key(args.task)
    raw = json.load(open(os.path.join(args.result_dir, f"{key}_{args.model}.json")))

    if args.mode == "apply":
        out_dir = args.out_dir or f"{args.result_dir.rstrip('/')}_ctrl"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{key}_{args.model}.json"), "w") as f:
            json.dump({v: control(ev, args.cooldown) for v, ev in raw.items()}, f, indent=4)
        print(f"[DONE] cooldown={args.cooldown:g}s -> {out_dir}/{key}_{args.model}.json")
        return

    splits = dev_test_split(task, args.annotation_root, os.path.join(args.result_dir, "controller", "splits"))
    rows = []
    for c in [None] + args.grid:
        tag = "raw" if c is None else f"c{c:g}"
        results = raw if c is None else {v: control(ev, c) for v, ev in raw.items()}
        rows.append((tag, {s: score(results, fp, os.path.join(args.result_dir, "controller", f"{tag}_{s}"),
                                    args.model, task)
                           for s, fp in splits.items()}))

    fmt = lambda r: f"{r[0]:5.1f} / {r[1]:4.1f} ({int(r[2])}, {int(r[3])})" if r else "n/a"
    print(f"{task} {args.model} ({args.result_dir})   cells: T-Score@5 / T-F1@5 (FP, FN)")
    for tag, r in rows:
        print(f"  {tag:5s}  dev {fmt(r['dev'])}   test {fmt(r['test'])}   full {fmt(r['full'])}")
    best = max(rows[1:], key=lambda tr: tr[1]["dev"][1])
    print(f"  -> chosen on dev: {best[0]}   test {fmt(best[1]['test'])}   full {fmt(best[1]['full'])}")


if __name__ == "__main__":
    main()
