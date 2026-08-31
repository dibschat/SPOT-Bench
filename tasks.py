"""SPOT-Bench task registry.
Imported by `inference.py`, `score.py`, and the `evals` package.

    ABD  action boundary detection      PNR  point-of-no-return
    SQA  streaming question answering   SPG  streaming procedural guidance
    SI   solicited intervention         UI   unsolicited intervention
"""

import os

TASKS = ["ABD", "PNR", "SQA", "SPG", "SI", "UI"]

TASK_TO_FAMILY = {
    "ABD": "detection",
    "PNR": "detection",
    "SQA": "interaction",
    "SPG": "interaction",
    "SI": "intervention",
    "UI": "intervention",
}

TASK_TO_ANNOTATION = {
    "ABD": "abd.json",
    "PNR": "pnr.json",
    "SQA": "sqa.json",
    "SPG": "spg.json",
    "SI": "si.json",
    "UI": "ui.json",
}


def task_key(task: str) -> str:
    return task.lower()


def annotation_path(annotation_root: str, task: str) -> str:
    return os.path.join(annotation_root, TASK_TO_ANNOTATION[task])


def normalize_tasks(parser, raw_tasks) -> list:
    tasks = [t.upper() for t in raw_tasks]
    unknown = sorted(set(tasks) - set(TASKS))
    if unknown:
        parser.error(
            f"unknown task(s): {', '.join(unknown)}. Valid tasks: {', '.join(TASKS)}"
        )
    return tasks
