import os
import re
import random
import numpy as np
import tensorflow as tf
from keras.src.saving.saving_api import load_model
from tensorflow.python import keras
from tensorflow.keras.preprocessing.text import Tokenizer


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

def save_text(generated_text, title):
    filename = path_results + "/" + title + ".txt"
    with open(filename, "w", encoding="UTF8") as f:
        f.write(generated_text)

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


def generate_text_markov_adv(markov_chain_n, seed, num_words):
    ## usually markov model goes with a probability for the starting state.
    ## here, starting state is a fixed value "seed"

    start_state = seed
    current_state = start_state
    max_sequence = len(markov_chain_n)
    generated_text = list(current_state)

    for _ in range(num_words):
        next_word_sample = []

        for n in range(1, len(generated_text) + 1):
            if n > max_sequence:
                break
            #print(n)
            current_state = tuple(generated_text[-n:])
            #print("Current State: ", current_state)
            markov_chain = markov_chain_n.get(n)
            next_word_sample += markov_chain.get(current_state, [])
            #print("Current Sample: ", next_word_sample)
        #print("Full Sample: ", next_word_sample)
        next_word = random.choice(next_word_sample)
        #print("Next Word: ", next_word)
        generated_text.append(next_word)
        current_state = tuple(generated_text[-len(start_state):])
        #print("New Current State: ", current_state)
        #print("Generated text: ", generated_text)


    return ' '.join(generated_text)

def main():


    # set to True to generate new text, see generate text for model parameters
    generate = False

    if generate:
        # Example usage
        kogler_text = read_file(file1_kogler)
        kickl_text = read_file(file2_kickl)

        kogler_words = pre_processing(kogler_text)
        kickl_words = pre_processing(kickl_text)

        #n = 1  # Order of the Markov model
        num_words = 150

        adv_kogler = build_markov_chain_adv(kogler_words, 20)
        adv_kickl = build_markov_chain_adv(kickl_words, 20)

        generated_text_kogler = generate_text_markov_adv(adv_kogler, tuple(kogler_words[:1]), 150)
        generated_text_kickl = generate_text_markov_adv(adv_kickl, tuple(kickl_words[:1]), 150)

        print("Kogler Text:")
        print(generated_text_kogler)

        print("Kickl Text:")
        print(generated_text_kickl)

        save_text(generated_text_kogler, "markov_adv_kogler_maxseq20")
        save_text(generated_text_kickl, "markov_adv_kickl_maxseq20")

if __name__ == '__main__':
    main()