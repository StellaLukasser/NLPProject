import os
from progress.bar import Bar
from keras.src.preprocessing.text import Tokenizer
from keras.src.saving.saving_api import load_model
from keras.src.utils import to_categorical
import numpy as np
import tensorflow as tf
from evaluation_metrics import bleu

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


def read_file(filename):
    text = []
    with open(filename, "r", encoding="UTF8") as f:
        for line in f:
            if len(line) > 1:
                line = line.split(" ", maxsplit=1)[1]
                text.append(line)
    return text


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


def save_single_line_text_in_txt(text, path):
    with open(path, "w", encoding="UTF8") as f:
        f.write(" ".join(text))


def load_multi_line_text_from_txt(path):
    sequences = []
    with open(path, "r") as f:
        for line in f:
           sequences.append(line)
    return sequences


def load_single_line_text_from_txt_tokenized(path):
    with open(path, "r", encoding="UTF8") as f:
        return f.read().split(" ")


def eval_task1():
    reference = read_file("data/data_stage_1.txt")
    generated_text = read_file(path_results + "/group24_stage1_generation_model2.txt")
    print(bleu(reference, generated_text, (0.25, 0.25, 0.25, 0.25)))
    print(bleu(reference, generated_text, (0.5, 0.5, 0, 0)))
    print(bleu(reference, generated_text, (0.5, 0.25, 0.25, 0)))

    generated_text = read_file(path_results + "/group24_stage1_generation_model2_with_punc.txt")
    print(bleu(reference, generated_text, (0.25, 0.25, 0.25, 0.25)))
    print(bleu(reference, generated_text, (0.5, 0.5, 0, 0)))
    print(bleu(reference, generated_text, (0.5, 0.25, 0.25, 0)))


def insert_punctuations():
    reference = read_file("data/data_stage_1.txt")
    # mean and std of how often punctuations appear
    nums = []
    for line in reference:
        last = 0
        tmp = line.split(" ")
        for i, word in enumerate(tmp):
            if word.__contains__("!") or word.__contains__(".") or word.__contains__(":") or word.__contains__("?"):
                nums.append(i - last)
                last = i
    mean = np.array(nums).mean()
    std = np.array(nums).std()
    print(mean)
    print(std)

    # insert random punctuations
    generated_text = read_file(path_results + "/group24_stage1_generation_model2.txt")
    gen = generated_text[0].split(" ")
    tmp = [abs(np.random.normal(mean, std))]
    for i in range(1, 2000):
        random_num = max(5, abs(np.random.normal(mean, std)))
        tmp.append(tmp[i-1] + random_num)
        num = tmp[i-1]
        if len(gen) - 1 < num:
            break
        gen[int(num)] = gen[int(num)] + "."
    gen[len(gen) - 1] += "."
    generated_text = ' '.join(gen)
    print(generated_text)
    filename = path_results + "/group24_stage1_generation_model2_with_punc.txt"
    with open(filename, "w", encoding="UTF8") as f:
        f.write(generated_text)


def create_directories():
    if not os.path.exists(path_processed_data):
        os.mkdir(path_processed_data)
    if not os.path.exists(path_models):
        os.mkdir(path_models)
    if not os.path.exists(path_results):
        os.mkdir(path_results)


def generate_text():
    sequence_length = 30
    num_words_to_generate = 2000

    # process data by removing punctuations and tokenizing sentences
    path_processed_data_task1 = path_processed_data + "/processed_data_task1_model2.txt"
    if os.path.exists(path_processed_data_task1):  # if this file already exists load it
        text = load_single_line_text_from_txt_tokenized(path_processed_data_task1)
        print("Loaded processed text")
    else:  # otherwise process data
        print("Process raw text")
        text = read_and_process_file("data/data_stage_1.txt")
        save_single_line_text_in_txt(text, path_processed_data_task1)  # save processed data
        print("Saved processed text")

    # split data into sequences
    path_sequenced_data_task1 = path_processed_data + "/sequenced_data_task1_model2.txt"
    if os.path.exists(path_sequenced_data_task1):  # if this file already exists load it
        sequences = load_multi_line_text_from_txt(path_sequenced_data_task1)
        print("Loaded sequences")
    else:  # otherwise sequence data
        print("Create sequences")
        sequences = split_text_into_sequences(text, sequence_length + 1)  # +1 for word which should be predicted
        save_sequences_in_txt(sequences, path_sequenced_data_task1)  # save sequenced data
        print("Saved sequences")

    # fit tokenizer on sequences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sequences)

    # tokenize sequences
    sequences_tokenized = tokenizer.texts_to_sequences(sequences)

    vocab_size = len(tokenizer.word_index) + 1

    # create X and y for training
    sequences_tokenized = np.array(sequences_tokenized)
    X, y = sequences_tokenized[:, :-1], sequences_tokenized[:, -1]
    y = to_categorical(y, num_classes=vocab_size)

    # load model if exists
    path = path_models + "/simple_rnn_100_epochs_batchsize_64_dropout_20.keras"
    if os.path.exists(path):
        model = load_model(path)
        print(f"Loaded model from disc ({path})")
    # else train model
    else:
        print("Train model")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, 50, input_length=sequence_length),
            tf.keras.layers.SimpleRNN(128),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(vocab_size, activation="softmax")
        ])
        epochs = 100
        batch_size = 64

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(X, y, batch_size=batch_size, epochs=epochs)
        model.save(path)
        print("Saved model")

    # prediction
    generated_text = text[-sequence_length:]
    with Bar('Predicting…', max=num_words_to_generate) as bar:
        for i in range(num_words_to_generate):
            X = [' '.join(generated_text[-sequence_length:])]
            input_seq = np.array(tokenizer.texts_to_sequences(X)[0])
            input_seq = input_seq.reshape(1, -1)
            next_word_index = np.argmax(model.predict(input_seq, verbose=0))
            next_word = tokenizer.sequences_to_texts(np.array(next_word_index).reshape(1, -1))
            generated_text.append(next_word[0])
            bar.next()

    generated_text = ' '.join(generated_text[sequence_length + 1:])
    filename = path_results + "/group24_stage1_generation_model2.txt"
    with open(filename, "w", encoding="UTF8") as f:
        f.write(generated_text)
    print(generated_text)


def task1():
    create_directories()
    generate_text()

