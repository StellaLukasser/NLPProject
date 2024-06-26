import os
import re
import random
from bert_score import BERTScorer

from archive.evaluation.evaluation_metrics_Updated import bleu1, bleu_ludi, rouge_n
from archive.evaluation.task2_withEvaluation import read_file_eval


path = os.curdir + "/data"
path_models = os.curdir + "/models/task2"
path_processed_data = os.curdir + "/processed_data/task2"
path_results = os.curdir + "/results/task2"
file1_kogler = path + "/data_stage_2/data_stage2_1_kogler.txt"
file2_kickl = path + "/data_stage_2/data_stage2_2_kickl.txt"


def read_file(file):
    f = open(file, "r", encoding="utf-8")
    text = f.read()
    f.close()
    return text

def pre_processing(text):
    # remove words until first ":"
    text = text.split(":", 1)[1]
    #removing brackets i.e., (Abg. Belakowitsch: Wo genau?)
    # or (Beifall bei Grünen und ÖVP.)
    text=re.sub("\(.*?\)","",text)
    #lower
    text = text.lower()
    #removing numbers
    text = re.sub("\d+", " ", text)
    #remove -
    text = re.sub(r"-", " ", text)
    text = re.sub(r"–", " ", text)
    text = re.sub(r":", " ", text)
    #spaces before punctuation
    text = re.sub('([.,!?()])', r" \1", text)
    #removing multiple spaces
    text = re.sub("\s+", " ", text).strip()
    #convert words and punctuations to indices
    text = re.findall(r"[\w']+|[.,!?;]", text)
    #print(text)
    return text


def build_markov_chain(words, n=1):
    markov_chain = {}
    for i in range(len(words) - n):
        current_state = tuple(words[i:i + n])
        next_state = words[i + n]
        if current_state not in markov_chain:
            markov_chain[current_state] = []
        markov_chain[current_state].append(next_state)
    return markov_chain


def generate_text_markov(markov_chain, seed, num_words):
    current_state = seed
    generated_text = list(current_state)

    for _ in range(num_words):
        next_word = random.choice(markov_chain.get(current_state, ['']))
        if next_word == '':
            break
        generated_text.append(next_word)
        current_state = tuple(generated_text[-len(current_state):])

    return ' '.join(generated_text)


def build_markov_chain_adv(words, max_sequence):
    markov_chain_n = {}

    n = 1

    for i in range(max_sequence):
        markov_chain = {}

        for j in range(len(words) - n):
            current_state = tuple(words[j:j + n])
            next_state = words[j + n]
            if current_state not in markov_chain:
                markov_chain[current_state] = []
            markov_chain[current_state].append(next_state)

        markov_chain_n[n] = markov_chain
        n+=1

    return markov_chain_n

def eval_task2(generated_text_kickl, generated_text_kogler):
    reference = read_file_eval(path + "/data_stage_2/data_stage2_2_kickl.txt")
    print("kickl")
    print("bleu1:", bleu1(reference, [generated_text_kickl], (0.25, 0.25, 0.25, 0.25)))
    print("bleu1:", bleu1(reference, [generated_text_kickl], (0.5, 0.5, 0, 0)))
    print("bleu1:", bleu1(reference, [generated_text_kickl], (0.5, 0.25, 0.25, 0)))

    take = bleu_ludi(reference, [generated_text_kickl])
    print("bleu2:", take)

    take2_recall, take2_f1_score = rouge_n(reference, [generated_text_kickl])
    print("rouge_n recall ", take2_recall, ", rouge_n  f1 score ", take2_f1_score)

    # BERTScore calculation
    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score([generated_text_kickl], reference)
    print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")


    reference = read_file_eval(path + "/data_stage_2/data_stage2_1_kogler.txt")
    print("kogler")
    print("bleu1:", bleu1(reference, [generated_text_kogler], (0.25, 0.25, 0.25, 0.25)))
    print("bleu1:", bleu1(reference, [generated_text_kogler], (0.5, 0.5, 0, 0)))
    print("bleu1:", bleu1(reference, [generated_text_kogler], (0.5, 0.25, 0.25, 0)))

    take = bleu_ludi(reference, [generated_text_kogler])
    print("bleu2:", take)

    take2_recall, take2_f1_score = rouge_n(reference, [generated_text_kogler])
    print("rouge_n recall ", take2_recall, ", rouge_n  f1 score ", take2_f1_score)

    P_2, R_2, F1_2 = scorer.score([generated_text_kogler], reference)
    print(f"BERTScore Precision: {P_2.mean():.4f}, Recall: {R_2.mean():.4f}, F1: {F1_2.mean():.4f}")


def generate_text_markov_adv(markov_chain_n, seed, num_words):
    start_state = seed
    current_state = start_state
    max_sequence = len(markov_chain_n)
    generated_text = list(current_state)

    for _ in range(num_words):
        next_word_sample = []

        if max_sequence > len(generated_text):
            max_sequence = len(generated_text)
        else:
            max_sequence = len(markov_chain_n)

        for n in range(1, max_sequence + 1):
            current_state = tuple(generated_text[-n:])
            #print("Current State: ", current_state)
            markov_chain = markov_chain_n.get(n)
            next_word_sample += markov_chain.get(current_state, [])
            #print("Current Sample: ", next_word_sample)

        #print("Full Sample: ", next_word_sample)
        next_word = random.choice(next_word_sample)
        generated_text.append(next_word)
        current_state = tuple(generated_text[-len(start_state):])
        #print(generated_text)
        #print(current_state)


    return ' '.join(generated_text)


# Example usage
kogler_text = read_file(file1_kogler)
kickl_text = read_file(file2_kickl)

kogler_words = pre_processing(kogler_text)
kickl_words = pre_processing(kickl_text)

#print(kogler_words)
#print(kickl_words)

#n = 1  # Order of the Markov model

#markov_chain_kogler = build_markov_chain(kogler_words, n)
#markov_chain_kickl = build_markov_chain(kickl_words, n)

adv_kogler = build_markov_chain_adv(kogler_words, 10)
adv_kickl = build_markov_chain_adv(kickl_words, 10)

generated_text_kogler = generate_text_markov_adv(adv_kogler, tuple(kogler_words[:1]), 150)
generated_text_kickl = generate_text_markov_adv(adv_kickl, tuple(kickl_words[:1]), 150)

print("Kogler Text:")
print(generated_text_kogler)

print("Kickl Text:")
print(generated_text_kickl)

#for key in adv.keys():
#    print(adv[key])

# Generate text

num_words = 2500

eval_task2(generated_text_kickl, generated_text_kogler)

#generated_text_kogler = generate_text_markov(markov_chain_kogler, tuple(kogler_words[:n]), num_words)
#generated_text_kickl = generate_text_markov(markov_chain_kickl, tuple(kickl_words[:n]), num_words)

#print("Kogler Chain:")
#print(markov_chain_kogler)
#print("Kogler Text:")
#print(generated_text_kogler)
#print("Kickl Chain:")
#print(markov_chain_kickl)
#print("Kickl Text:")
#print(generated_text_kickl)
