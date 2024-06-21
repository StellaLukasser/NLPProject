from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu, corpus_bleu
import evaluate
from transformers import BertTokenizer, BertModel
import numpy as np

rouge = evaluate.load('rouge')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def bleu_score_(ref, gen, weights):
    '''
    calculate bleu score. uses nltk implementation
    Args:
        ref: list of ref sentences [["this", "is", "a", "sentence"], ["this", "is", "another", "one"]]
        gen: list of gen sentences [["this", "is", "a", "sentence"], ["this", "is", "another", "one"]]
        weights: which n-grams to use for bleu score

    Returns:
        bleu score(float)
    '''

    # can play with smoothing function see (https://www.nltk.org/api/nltk.translate.bleu_score.html)
    cc = SmoothingFunction()

    # bleu_scores = []
    # this version with taking the mean is normal averaging on sentence level,
    # but original bleu accounts for the micro-average precision, so we will use the corpus_bleu function
    # for sentence in gen:
    #     bleu_scores.append(sentence_bleu(ref, sentence, weights=weights, smoothing_function=cc.method4))
    # return np.array(bleu_scores).mean()

    return corpus_bleu([ref]*len(gen), gen, weights=weights, smoothing_function=cc.method4)


def rouge_score_(ref, gen):
    '''
    calculate rouge score. uses evaluate implementation
    Args:
        ref: ref: list of ref sentences ["this is a sentence", "this is another one"]
        gen: list of ref sentences ["this is a sentence", "this is another one"]

    Returns:
    dict: rouge1, rouge2, rougeL, rougeLsum
    '''
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


def bert_score_(ref, gen):
    '''
    calculate bert score using cosine of embeddings
    Args:
        ref: list of ref sentences ["this is a sentence", "this is another one"]
        gen: list of ref sentences ["this is a sentence", "this is another one"]

    Returns:
       similarity score (cosine sim of the embeddings)
    '''
    ref_tokenized = [tokenizer(ref_sentence, return_tensors="pt", padding=True, truncation=True) for ref_sentence in ref]
    gen_tokenized = [tokenizer(gen_sentence, return_tensors="pt", padding=True, truncation=True) for gen_sentence in gen]

    outputsref = [model(**ref_tokenized_sentence) for ref_tokenized_sentence in ref_tokenized]
    outputsgen = [model(**gen_tokenized_sentence) for gen_tokenized_sentence in gen_tokenized]

    embeddingsref = [outputref.last_hidden_state.mean(dim=1).detach().numpy() for outputref in outputsref]
    embeddingsgen = [outputgen.last_hidden_state.mean(dim=1).detach().numpy() for outputgen in outputsgen]

    # Calculate cosine similarity of embeddings
    similarities = 0
    for embeddinggen in embeddingsgen:
        for embeddingref in embeddingsref:
            similarities += np.dot(embeddinggen, embeddingref.T) / (np.linalg.norm(embeddinggen) * np.linalg.norm(embeddingref))

    return similarities / (len(ref) * len(gen))
