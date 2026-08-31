import argparse
import json
import os

from model_wrappers import MODELS, load_model
from tasks import TASKS, annotation_path, task_key

# MMDuet2 KV-cache policies.
MMDUET2_KV_MODES = ["original", "kvflush", "streamingvlm"]
MMDUET2_DEFAULT_KV_MODE = "kvflush"

# QwenVL models
QWEN_VERSIONS = ["2.5", "3"]
QWEN_DEFAULT_VERSION = "2.5"


def args_parser():
    parser = argparse.ArgumentParser(description="Run SPOT-Bench streaming inference")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        metavar="{ABD,PNR,SQA,SPG,SI,UI}",
        help="Tasks to run (one or more).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=MODELS,
        help="Registered Model name.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from {task}_{model}.temp.json if available",
    )
    parser.add_argument(
        "--annotation_root",
        type=str,
        default="data",
        help="Directory containing SPOT-Bench annotations.",
    )
    parser.add_argument(
        "--annotation_path",
        type=str,
        default=None,
        help="Explicit annotation JSON path. Overrides --annotation_root.",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default="data/videos",
        help="Directory containing the video files.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results",
        help="Output directory to store the result jsons.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model checkpoint. Overrides the wrapper's default.",
    )
    parser.add_argument(
        "--kv_mode",
        type=str,
        default=None,
        choices=MMDUET2_KV_MODES,
        help=(
            "MMDuet2 KV-cache policy (default: kvflush). "
            "'original' is the unbounded baseline cache and OOMs on long videos."
        ),
    )
    parser.add_argument(
        "--qwen_version",
        type=str,
        default=None,
        choices=QWEN_VERSIONS,
        help=f"QwenVL version to run offline sliding window baseline (default: {QWEN_DEFAULT_VERSION}).",
    )

    args = parser.parse_args()

    args.task = args.task.upper()
    if args.task not in TASKS:
        parser.error(f"--task must be one of: {', '.join(TASKS)}")

    if args.kv_mode is not None and args.model != "MMDuet2":
        parser.error(f"--kv_mode only applies to MMDuet2, not {args.model}")

    if args.qwen_version is not None and args.model != "QwenVL":
        parser.error(f"--qwen_version only applies to QwenVL, not {args.model}")

    args.task_key = task_key(args.task)
    args.annotations = args.annotation_path or annotation_path(
        args.annotation_root, args.task
    )
    return args


def build_model(args):
    class ModelArgs:
        pass

    model_args = ModelArgs()
    if args.model_path is not None:
        model_args.model_path = args.model_path
    if args.model == "MMDuet2":
        model_args.kv_mode = args.kv_mode or MMDUET2_DEFAULT_KV_MODE
    if args.model == "QwenVL":
        model_args.qwen_version = args.qwen_version or QWEN_DEFAULT_VERSION

    return load_model(args.model, model_args)


def main():
    args = args_parser()
    print(f"[INFO]\nTask: {args.task}\nModel: {args.model}\nResume: {args.resume}\n")

    if not os.path.exists(args.annotations):
        raise FileNotFoundError(f"Annotation file not found: {args.annotations}")
    with open(args.annotations, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    print(f"Loaded {len(annotations)} videos from {args.annotations}\n")

    model = build_model(args)
    model.eval(
        task=args.task_key,
        annotations=annotations,
        video_root=args.video_root,
        model_name=args.model,
        result_path=args.result_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
