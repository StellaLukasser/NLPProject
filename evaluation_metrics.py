import os

import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import evaluate
#!pip install evaluate
#!pip install rouge-score


path_data_task2 = os.curdir + "/data/data_stage_2"
path_results_task2 = os.curdir + "/results/task2"

rouge = evaluate.load('rouge')

def bleu_score_(ref, gen, weights):
    bleu_scores = []
    for sentence in gen:
        bleu_scores.append(sentence_bleu(ref, sentence, weights=weights))

    return np.array(bleu_scores).mean()


def rouge_score_(ref, gen):
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    rougeLsum_scores = []
    for sentence in gen:
        result_dict = rouge.compute(predictions=[sentence], references= [ref])
        rouge1_scores.append(result_dict["rouge1"])
        rouge2_scores.append(result_dict["rouge2"])
        rougeL_scores.append(result_dict["rougeL"])
        rougeLsum_scores.append(result_dict["rougeLsum"])

    result_dict = {"rouge1": np.array(rouge1_scores).mean(),
                   "rouge2": np.array(rouge2_scores).mean(),
                   "rougeL": np.array(rougeL_scores).mean(),
                   "rougeLsum": np.array(rougeLsum_scores).mean()}
    return result_dict




