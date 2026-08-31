"""Semantic matching for SPOT-Bench.

Detection tasks (ABD, PNR) have categorical answers and are matched with keyword
rules. The open-ended tasks (SQA, SPG, SI, UI) are judged by an LLM that rates on 
a 0-5 scale; a prediction counts as correct when the score clears the task's `sim_threshold`.

The judge reads `OPENAI_API_KEY` from the environment. Export it in the shell that runs `score.py`
"""

import ast
import os
import random
import re
import time

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .metrics import norm_text

JUDGE_MODEL = "gpt-5-mini"
JUDGE_MAX_RETRIES = 3

_client = None


def _get_client():
    global _client
    if OpenAI is None:
        raise ImportError("openai is not installed. Run: pip install openai")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. The open-ended tasks (SQA, SPG, SI, UI) "
            "need the LLM judge; export the key before running score.py."
        )
    if _client is None:
        _client = OpenAI()
    return _client


ABD_START_RE = re.compile(r"\b(?:start|begin|onset|commence)\w*")
ABD_END_RE = re.compile(r"\b(?:end|finish|stop|complete|conclude|done)\w*")


def mentions_start(text: str) -> bool:
    return bool(ABD_START_RE.search(norm_text(text)))


def mentions_end(text: str) -> bool:
    return bool(ABD_END_RE.search(norm_text(text)))


def semantic_ok_detection(task: str, pred: str, gold: str) -> bool:
    """Keyword match for the detection tasks."""
    p = norm_text(pred)
    g = norm_text(gold)

    if task == "PNR":
        return bool(re.search(r"\bnow\b", p))

    if task == "ABD":
        has_start = bool(ABD_START_RE.search(p))
        has_end = bool(ABD_END_RE.search(p))
        # A boundary is a start or an end, never both at the same instant. A
        # prediction naming both -- typically narration echoing the whole
        # question -- commits to neither, so it matches neither.
        if has_start and has_end:
            return False
        if mentions_start(g):
            return has_start
        if mentions_end(g):
            return has_end
        return p == g

    return False


def semantic_ok_open_ended(
    task: str, question: str, gold: str, pred: str, score_threshold: int = 3
) -> bool:
    """Ask the LLM judge whether `pred` means the same as `gold`."""
    pred = (pred or "").strip()
    gold = (gold or "").strip()
    if not pred or not gold:
        return False

    prompt = build_prompt(task, question, gold, pred)

    for attempt in range(1, JUDGE_MAX_RETRIES + 1):
        try:
            result = _get_client().responses.create(
                model=JUDGE_MODEL,
                input=[{"role": "user", "content": prompt}],
                reasoning={"effort": "low"},
                text={"verbosity": "low"},
            )
            raw = str(getattr(result, "output_text", "") or "").strip()

            data = ast.literal_eval(raw)
            score = int(data.get("score", 0))
            if score < 0 or score > 5:
                raise ValueError(f"score out of range: {score}")
            return score >= score_threshold
        except Exception as e:
            print(f"[WARN] Judge call failed (attempt {attempt}/{JUDGE_MAX_RETRIES}): {e}")
            if attempt < JUDGE_MAX_RETRIES:
                time.sleep(0.5 * attempt + random.uniform(0, 0.25))

    print(
        f"[ERROR] Judge unreachable after {JUDGE_MAX_RETRIES} attempts. Counting this "
        "prediction as a mismatch -- the score below is a lower bound, not a verdict."
    )
    return False


SQA_EVAL_PROMPT = """
You are an automatic evaluator for video question answering.
Your job is to judge whether a model’s predicted answer meaningfully matches the
correct answer for a video-based question.

INSTRUCTIONS:
- Focus strictly on semantic meaning.
- Paraphrases are fine if they preserve meaning.
- If the predicted answer does NOT answer the question, mark "no".
- If it answers the same underlying fact, mark "yes".
- Score meaning similarity from 0–5:
  5 = same meaning
  4 = near-identical
  3 = partially correct but acceptable
  2 = weakly related
  1 = barely related
  0 = unrelated or wrong
- No explanations. Only output JSON.
- Sometimes the answers might be in Chinese characters.
- Sometimes an answer may contain up to two candidates separated by a semicolon.
  Evaluate them independently; if either candidate matches, mark "yes" and use
  the score of the better-matching candidate.

Evaluate:
Question: {question}
Correct Answer: {gold}
Predicted Answer: {pred}

Respond ONLY with:
{{"pred": "yes" or "no", "score": INTEGER}}
"""

SPG_EVAL_PROMPT = """
You are an automatic evaluator for procedural guidance.
Your job is to check whether a model’s predicted next action meaningfully matches
the correct next action for a video-based procedural task.

INSTRUCTIONS:
- Compare predicted action vs correct action purely on meaning.
- Accept paraphrases that represent the same action or intent.
- If the predicted action does NOT correspond to the same step, mark "no".
- Score similarity on a 0–5 scale:
  5 = identical step
  4 = equivalent phrasing
  3 = partial correctness
  2 = weak overlap
  1 = barely related
  0 = wrong action
- No reasoning or extra text.
- Sometimes the answers might be in Chinese characters.
- Sometimes an answer may contain up to two candidates separated by a semicolon.
  Evaluate them independently; if either candidate matches, mark "yes" and use
  the score of the better-matching candidate.

Evaluate:
Correct Action: {gold}
Predicted Action: {pred}

Respond ONLY with:
{{"pred": "yes" or "no", "score": INTEGER}}
"""

SI_EVAL_PROMPT = """
You are an automatic evaluator for an intervention task.
Here the assistant gives a short instruction to help a user with mistakes or hesitation.

Your job is to judge whether the model’s predicted instruction meaningfully
matches the correct intended instruction.

INSTRUCTIONS:
- Focus only on meaning: does the instruction guide the user toward the
  same next step or same correction?
- Paraphrasing is allowed if the guidance is equivalent.
- If the predicted instruction does not help in the same way, mark "no".
- Score similarity from 0–5:
  5 = identical guidance
  4 = equivalent instruction
  3 = partially correct but usable
  2 = weak help but still helps the user
  1 = almost irrelevant
  0 = wrong instruction
- No explanation text.
- Sometimes the answers might be in Chinese characters.
- Sometimes an answer may contain up to two candidates separated by a semicolon.
  Evaluate them independently; if either candidate matches, mark "yes" and use
  the score of the better-matching candidate.

Evaluate:
Correct Instruction: {gold}
Predicted Instruction: {pred}

Respond ONLY with:
{{"pred": "yes" or "no", "score": INTEGER}}
"""

UI_EVAL_PROMPT = """
You are an automatic evaluator for an blind assistance intervention task,
where an assistant gives short spoken warnings or guidance based on a blind
user’s video.

Your job is to judge whether the model’s predicted warning or guidance meaningfully
matches the correct warning based on the visible risk.

INSTRUCTIONS:
- Judge semantic meaning only.
- A correct prediction must warn about the same risk or obstacle.
- Paraphrases are fine; irrelevant or mismatched warnings are incorrect.
- Score similarity on a 0–5 scale:
  5 = same warning
  4 = near-identical
  3 = partially correct
  2 = weak relevance
  1 = barely related but still a warning
  0 = wrong/no warning
- No explanations.
- Sometimes multiple correct warnings are provided separated by semicolons;
- Sometimes the answers might be in Chinese characters.
- Sometimes an answer may contain up to two candidates separated by a semicolon.
  Evaluate them independently; if either candidate matches, mark "yes" and use
  the score of the better-matching candidate.

Evaluate:
Correct Warning separated by ';': {gold}
Predicted Warnings: {pred}

Respond ONLY with:
{{"pred": "yes" or "no", "score": INTEGER}}
"""


def build_prompt(task, question, gold, pred):
    if task == "SQA":
        return SQA_EVAL_PROMPT.format(question=question, gold=gold, pred=pred)
    if task == "SPG":
        return SPG_EVAL_PROMPT.format(gold=gold, pred=pred)
    if task == "SI":
        return SI_EVAL_PROMPT.format(gold=gold, pred=pred)
    if task == "UI":
        return UI_EVAL_PROMPT.format(gold=gold, pred=pred)

    raise ValueError(f"Unknown task type for semantic matching: {task}")
