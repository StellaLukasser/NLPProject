#https://www.scaler.com/topics/deep-learning/text-generation/

import os
import re
import numpy as np
import tensorflow as tf
from keras.src.saving.saving_api import load_model
from bert_score import BERTScorer

from archive.evaluation.evaluation_metrics_Updated import bleu1, bleu_ludi, rouge_n

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


def split_text_into_sequences(text, length):
    sequences = list()
    for i in range(length, len(text)):
        sequence = text[i - length:i]
        line = ' '.join(sequence)
        sequences.append(line)
    return sequences

def text_process(text, seed, num_chars, model_name):
    # The unique characters in the file
    vocab = sorted(set(text))
    #print(f'{len(vocab)} unique characters')
    #print(vocab)

    #convert strings to numerical representation
    char_to_int = {ch:i for i, ch in enumerate(vocab)}
    int_to_char = {i:ch for i, ch in enumerate(vocab)}

    # Set the maximum sequence length (max_len) to be the length of the longest sequence
    max_len = max([len(s) for s in text])
    print(max_len)

    path_model = path_models + "/" + model_name + ".keras"
    if os.path.exists(path_model):
        model = load_model(path_model)
        #print(f"Loaded model from disc ({path_model})")
    else:
        # Create training examples and labels
        X = []
        y = []

        for i in range(0, len(text)-max_len, 1):
            X.append([char_to_int[ch] for ch in text[i:i+max_len]])
            y.append(char_to_int[text[i+max_len]])
        X = np.array(X)
        y = np.array(y)

        # Pad the examples
        X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_len, padding='post')

        # Convert labels to categorical format
        #y = tf.keras.utils.to_categorical(y)
        y = y.reshape(y.shape[0], 1)

        # Define the model architecture
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(input_dim=len(vocab), output_dim=50, input_length=max_len))
        model.add(tf.keras.layers.RNN(units=128))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(units=len(vocab), activation='softmax'))

        loss1 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        model.compile(optimizer='adam', loss=loss1)

        # Train the model
        model.fit(X, y, epochs=80, batch_size=64)
        model.save(path_model)
        print("Saved model")

    # Initialize the generated text
    generated_text = seed

    # Encode the seed as integers
    encoded_seed = [char_to_int[ch] for ch in seed]

    # Pad the seed
    padded_seed = tf.keras.preprocessing.sequence.pad_sequences([encoded_seed], maxlen=max_len, padding='post')

    # Generate characters
    for i in range(num_chars):
        # Get the next character probabilities
        probs = model.predict(padded_seed)[0]
        # Get the index of the character with the highest probability
        index = np.argmax(probs)
        # Add the character to the generated text
        generated_text.append(int_to_char[index])
        # Update the padded seed with the latest character
        padded_seed = np.append(padded_seed[0][1:], index)
        padded_seed = tf.keras.preprocessing.sequence.pad_sequences([padded_seed], maxlen=max_len, padding='post')

    generated_text = ' '.join(generated_text[25:]) + "."
    punctuations = ['!', ',', '.', ';', '?']
    for punctuation in punctuations:
        generated_text = generated_text.replace(" " + punctuation, punctuation)
    #generated_text = generated_text.replace(".", ".\n")
    return generated_text


def create_directories():
    if not os.path.exists(path_processed_data):
        os.mkdir(path_processed_data)
    if not os.path.exists(path_models):
        os.mkdir(path_models)
    if not os.path.exists(path_results):
        os.mkdir(path_results)


def read_file_eval(filename):
    text = []
    with open(filename, "r", encoding="UTF8") as f:
        for line in f:
            if len(line) > 1:
                line = line.split(" ", maxsplit=1)[1]
                text.append(line)
    return text


def eval_task2(generated_text_kickl, generated_text_kogler):
    reference = read_file_eval(path + "/data_stage_2/data_stage2_2_kickl.txt")
    print("kickl")
    print("bleu1:", bleu1(reference, generated_text_kickl, (0.25, 0.25, 0.25, 0.25)))
    print("bleu1:", bleu1(reference, generated_text_kickl, (0.5, 0.5, 0, 0)))
    print("bleu1:", bleu1(reference, generated_text_kickl, (0.5, 0.25, 0.25, 0)))

    take = bleu_ludi(reference, generated_text_kickl)
    print("bleu2:", take)

    take2_recall, take2_f1_score = rouge_n(reference, generated_text_kickl)
    print("rouge_n recall ", take2_recall, ", rouge_n  f1 score ", take2_f1_score)

    # BERTScore calculation
    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score(generated_text_kickl, reference)
    print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")


    reference = read_file_eval(path + "/data_stage_2/data_stage2_1_kogler.txt")
    print("kogler")
    print("bleu1:", bleu1(reference, generated_text_kogler, (0.25, 0.25, 0.25, 0.25)))
    print("bleu1:", bleu1(reference, generated_text_kogler, (0.5, 0.5, 0, 0)))
    print("bleu1:", bleu1(reference, generated_text_kogler, (0.5, 0.25, 0.25, 0)))

    take = bleu_ludi(reference, generated_text_kogler)
    print("bleu2:", take)

    take2_recall, take2_f1_score = rouge_n(reference, generated_text_kogler)
    print("rouge_n recall ", take2_recall, ", rouge_n  f1 score ", take2_f1_score)

    P_2, R_2, F1_2 = scorer.score(generated_text_kogler, reference)
    print(f"BERTScore Precision: {P_2.mean():.4f}, Recall: {R_2.mean():.4f}, F1: {F1_2.mean():.4f}")
    print('hello')


def generate_text(text_kickl, text_kogler):
    generated_text = text_process(text_kickl, text_kickl[len(text_kickl) - 24:], 290, 'kickl_rnn128_100_epochs_batchsize_64_dropout_20')
    print(generated_text)
    text_kickl = ' '.join(text_kickl) + "."
    punctuations = ['!', ',', '.', ';', '?']
    for punctuation in punctuations:
        text_kickl = text_kickl.replace(" " + punctuation, punctuation)
    print(text_kickl)
    filename = path_results + "/group24_stage2_generation2.txt"
    with open(filename, "w", encoding="UTF8") as f:
        f.write(generated_text)

    generated_text = text_process(text_kogler, text_kogler[len(text_kogler) - 24:], 290, 'kogler_rnn128_100_epochs_batchsize_64_dropout_20')
    print(generated_text)
    text_kogler = ' '.join(text_kogler) + "."
    punctuations = ['!', ',', '.', ';', '?']
    for punctuation in punctuations:
        text_kogler = text_kogler.replace(" " + punctuation, punctuation)
    print(text_kogler)
    filename = path_results + "/group24_stage2_generation1.txt"
    with open(filename, "w", encoding="UTF8") as f:
        f.write(generated_text)


def main():
    # create_directories()
    # #read text
    # text_kogler = read_file(file1_kogler)
    # text_kickl = read_file(file2_kickl)
    # #process data

    # text_kogler = pre_processing(text_kogler)
    # text_kickl = pre_processing(text_kickl)

    # # Generate text
    # generate_text(text_kickl, text_kogler)
    generated_text_kickl = read_file_eval(path_results + "/group24_stage2_generation2.txt")
    generated_text_kogler = read_file_eval(path_results + "/group24_stage2_generation1.txt")
    eval_task2(generated_text_kickl, generated_text_kogler)


if __name__ == '__main__':
    main()