"""utils"""
import random

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def calculate_metrics(
    all_choices: list,
    all_answers: list,
    all_response: list,
    all_index2ans: list = None,
    allow_random: bool = True,
) -> dict:
    """calculate_metrics"""
    if all_index2ans is None:
        all_index2ans = [None] * len(all_response)

    predictions = [
        parse_multi_choice_response(response, all_choices, index2ans, allow_random)
        for response, index2ans in zip(all_response, all_index2ans)
    ]

    accuracy = accuracy_score(all_answers, predictions)
    f1 = f1_score(all_answers, predictions, average="weighted", zero_division=1)
    precision = precision_score(all_answers, predictions, average="weighted", zero_division=1)
    recall = recall_score(all_answers, predictions, average="weighted", zero_division=1)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }

def parse_multi_choice_response(
    response: str,
    all_choices: list,
    index2ans: dict = None,
    allow_random: bool = True,
) -> str:
    """parse_multi_choice_response"""
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f'({choice})' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f' {choice} ' in response:
                candidates.append(choice)

    if index2ans is not None and len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans and ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        if allow_random:
            pred_index = random.choice(all_choices)
        else:
            pred_index = ""


    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f'({can})')
                    start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)

        pred_index = candidates[np.argmax(start_indexes)]
    else:
        pred_index = candidates[0]

    return pred_index
