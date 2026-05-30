"""Reward computations for answer correctness and structural format."""

import re
import string

from eval import cal_em, cal_f1


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return 1.0
    return 0.0


def extract_solution(solution_str):
    match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    return match.group(1).strip() if match else None


def compute_score_format(solution_str):
    """Reward 0.5 per turn satisfying the structural template."""
    if solution_str is None:
        return 0.0
    try:
        blocks = re.findall(
            r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        if not blocks:
            return 0.0

        format_reward = 0.0
        for block in blocks[:-1]:
            has_think = re.search(r'<think>.*?</think>', block, re.DOTALL)
            has_query = re.search(r'<query>.*?</query>', block, re.DOTALL)
            has_graph = re.search(r'<graph>.*?</graph>', block, re.DOTALL)
            # Any combination with <think> and at least one action tag
            if has_think and (has_query or has_graph):
                format_reward += 0.5

        if blocks:
            last = blocks[-1]
            has_think = re.search(r'<think>.*?</think>', last, re.DOTALL)
            has_answer = re.search(r'<answer>.*?</answer>', last, re.DOTALL)
            if has_think and has_answer:
                format_reward += 0.5
    except Exception as e:
        print(f"Error in compute_score_format: {e}")
        return 0.0

    return format_reward


def compute_score_answer(solution_str, ground_truth):
    if solution_str is None:
        return 0.0
    try:
        blocks = re.findall(
            r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        solution_str = blocks[-1]
        answer = extract_solution(solution_str)
        if answer is not None:
            return cal_f1([ground_truth.tolist()], [answer])
        return 0.0
    except Exception as e:
        print(f"Error in compute_score_answer: {e}")
        return 0.0


def compute_score_format_answer(solution_str, ground_truth):
    if solution_str is None or ground_truth is None:
        return 0.0
    try:
        format_reward = compute_score_format(solution_str)
        answer_reward = compute_score_answer(solution_str, ground_truth)
        format_reward = min(format_reward, 1.0)
        if format_reward == 1.0:
            return -1.0 + format_reward + answer_reward
        else:
            return -1.0 + format_reward
    except Exception as e:
        print(f"Error in compute_score_format_answer: {e}")
        return 0.0


def compute_score_em(solution_str, ground_truth):
    if solution_str is None or ground_truth is None:
        return 0.0
    try:
        blocks = re.findall(
            r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        solution_str = blocks[-1]
        answer = extract_solution(solution_str)
        if answer is None:
            return 0.0
        return float(cal_em([ground_truth.tolist()], [answer]))
    except Exception as e:
        print(f"Error in compute_score_em: {e}")
        return 0.0


def compute_score_f1(solution_str, ground_truth):
    if solution_str is None or ground_truth is None:
        return 0.0
    try:
        blocks = re.findall(
            r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        solution_str = blocks[-1]
        answer = extract_solution(solution_str)
        if answer is None:
            return 0.0
        return float(cal_f1([ground_truth.tolist()], [answer]))
    except Exception as e:
        print(f"Error in compute_score_f1: {e}")
        return 0.0
