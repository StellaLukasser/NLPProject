from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from nltk.translate.bleu_score import sentence_bleu


def bleu(ref, gen):
    ref_bleu = []
    gen_bleu = gen[0].split()
    for l in ref:
        ref_bleu.append(l.split())

    return sentence_bleu(ref_bleu, gen_bleu, weights=(0.5, 0.5, 0, 0))

