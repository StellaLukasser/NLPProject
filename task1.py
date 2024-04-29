import os
from progress.bar import Bar
from keras.src.preprocessing.text import Tokenizer
from keras.src.saving.saving_api import load_model
from keras.src.utils import to_categorical

import numpy as np
import tensorflow as tf

replacement_map = {"-": " ",
                   "—": " ",
                   "!": "",
                   "(": "",
                   ")": "",
                   "?": "",
                   "]": "",
                   "[": "",
                   ";": "",
                   ",": "",
                   ":": "",
                   ".": "",
                   "\n": "",
                   '"': "",
                   "'": ""}
path_models = os.curdir + "/models"
path_processed_data = os.curdir + "/processed_data"
path_results = os.curdir + "/results"

def remove_punctuations(raw):
    processed = raw
    for punctuation in replacement_map:
        processed = processed.replace(punctuation, replacement_map[punctuation])
        processed = ''.join((x for x in processed if not x.isdigit()))
    return processed


def read_and_process_file(filename):
    text = []
    with open(filename, "r", encoding="UTF8") as f:
        for line in f:
            if len(line) > 1:
                line = line.split(" ", maxsplit=1)[1]  # remove line numbers (see txt file)
                line_preprocessed = remove_punctuations(line.lower())
                for word in line_preprocessed.split(" "):
                    if word != "":
                        text.append(word)
    return text


def split_text_into_sequences(text, length):
    sequences = list()
    for i in range(length, len(text)):
        sequence = text[i - length:i]
        line = ' '.join(sequence)
        sequences.append(line)
    return sequences


def save_sequences_in_txt(sequences, path):
    with open(path, "w", encoding="UTF8") as f:
        for sequence in sequences:
            f.write(sequence + "\n")

def save_text_in_txt(text, path):
    with open(path, "w", encoding="UTF8") as f:
        f.write(" ".join(text))

def load_sequences_from_txt(path):
    sequences = []
    with open(path, "r") as f:
        for line in f:
           sequences.append(line)
    return sequences


def load_text_from_txt(path):
    with open(path, "r", encoding="UTF8") as f:
        return f.read().split(" ")


def generate_text():
    sequence_length = 50 + 1
    num_words_to_generate = 2000

    path_processed_data_task1 = path_processed_data + "/processed_data_task1.txt"
    if os.path.exists(path_processed_data_task1):
        text = load_text_from_txt(path_processed_data_task1)
        print("Loaded processed text")
    else:
        print("Process raw text")
        text = read_and_process_file("data/data_stage_1.txt")
        save_text_in_txt(text, path_processed_data_task1)
        print("Saved processed text")


    path_sequenced_data_task1 = path_processed_data + "/sequenced_data_task1.txt"
    if os.path.exists(path_sequenced_data_task1):
        sequences = load_sequences_from_txt(path_sequenced_data_task1)
        print("Loaded sequences")

    else:
        print("Create sequences")
        sequences = split_text_into_sequences(text, sequence_length)
        save_sequences_in_txt(sequences, path_sequenced_data_task1)
        print("Saved sequences")

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sequences)

    sequences_tokenized = tokenizer.texts_to_sequences(sequences)
    vocab_size = len(tokenizer.word_index) + 1
    sequences_tokenized = np.array(sequences_tokenized)
    X, y = sequences_tokenized[:, :-1], sequences_tokenized[:, -1]
    y = to_categorical(y, num_classes=vocab_size)
    seq_length = X.shape[1]

    # load model
    if os.path.exists(path_models + "/simple_rnn.keras"):
        model = load_model(path_models + "/simple_rnn.keras")
        print(f"Loaded model from disc ({path_models}/simple_rnn.keras)")
    # train model
    else:
        print("Train model")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, 50, input_length=seq_length),
            tf.keras.layers.SimpleRNN(128),
            tf.keras.layers.Dense(vocab_size, activation="softmax")
        ])
        epochs = 50

        model.compile(loss="categorical_crossentropy", optimizer="adam")
        model.fit(X, y, epochs=epochs)
        model.save(path_models + "/simple_rnn.keras")
        print("Saved model")

    # prediction
    generated_text = text[-seq_length:]
    with Bar('Predicting…', max=num_words_to_generate) as bar:
        for i in range(num_words_to_generate):
            X = [' '.join(generated_text[-seq_length:])]
            input_seq = np.array(tokenizer.texts_to_sequences(X)[0])
            input_seq = input_seq.reshape(1, -1)
            next_word_index = np.argmax(model.predict(input_seq, verbose=0))
            next_word = tokenizer.sequences_to_texts(np.array(next_word_index).reshape(1, -1))
            generated_text.append(next_word[0])
            bar.next()

    generated_text = ' '.join(generated_text[seq_length + 1:])
    filename = path_results + "/group24_stage1_generation"
    with open(filename, "w", encoding="UTF8") as f:
        f.write(generated_text)
    print(generated_text)


def task1():
    if not os.path.exists(path_processed_data):
        os.mkdir(path_processed_data)
    if not os.path.exists(path_models):
        os.mkdir(path_models)
    if not os.path.exists(path_results):
        os.mkdir(path_results)
    generate_text()