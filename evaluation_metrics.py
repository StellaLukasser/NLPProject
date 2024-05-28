import os

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import evaluate
#!pip install evaluate
#!pip install rouge-score

rouge = evaluate.load('rouge')

def bleu_score_(ref, gen, weights):
    bleu_scores = []
    # can play with smoothing function see (https://www.nltk.org/api/nltk.translate.bleu_score.html)
    cc = SmoothingFunction()
    for sentence in gen:
        bleu_scores.append(sentence_bleu(ref, sentence, weights=weights, smoothing_function=cc.method4))

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




