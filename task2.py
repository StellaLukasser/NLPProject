#https://www.scaler.com/topics/deep-learning/text-generation/

import os
import re
from datetime import datetime
import numpy as np
import tensorflow as tf
from keras.src.saving.saving_api import load_model
from evaluation_metrics import bleu_score_, rouge_score_, bert_score_

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

    # Convert strings to numerical representation
    char_to_int = {ch:i for i, ch in enumerate(vocab)}
    int_to_char = {i:ch for i, ch in enumerate(vocab)}

    # Set the maximum sequence length (max_len) to be the length of the longest sequence
    max_len = max([len(s) for s in text])

    path_model = path_models + "/" + model_name + ".keras"
    if os.path.exists(path_model):
        model = load_model(path_model)
        print(f"Loaded model from disc ({path_model})")
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
        model.fit(X, y, epochs=100, batch_size=64)
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

    # join text and add dot at the end
    generated_text = ' '.join(generated_text[25:]) + "."

    # remove space before punctuations
    punctuations = ['!', ',', '.', ';', '?']
    for punctuation in punctuations:
        generated_text = generated_text.replace(" " + punctuation, punctuation)
    return generated_text


def create_directories():
    if not os.path.exists(path_processed_data):
        os.mkdir(path_processed_data)
    if not os.path.exists(path_models):
        os.mkdir(path_models)
    if not os.path.exists(path_results):
        os.mkdir(path_results)


def generate_text(text_kickl, text_kogler, model_name):
    generated_text = text_process(text_kickl, text_kickl[len(text_kickl) - 24:], 290, 'kickl_' + model_name)
    print(generated_text)
    text_kickl = ' '.join(text_kickl)
    if text_kickl[-1] != ".":
        text_kickl += "."
    punctuations = ['!', ',', '.', ';', '?']
    for punctuation in punctuations:
        text_kickl = text_kickl.replace(" " + punctuation, punctuation)
    print(text_kickl)
    filename = path_results + "/group24_stage2_generation2" + datetime.now().strftime("_%m%d_%H%M") + ".txt"
    with open(filename, "w", encoding="UTF8") as f:
        f.write(generated_text)

    generated_text = text_process(text_kogler, text_kogler[len(text_kogler) - 24:], 290, 'kogler_' + model_name)
    print(generated_text)
    text_kogler = ' '.join(text_kogler)
    if text_kogler[-1] != ".":
        text_kogler += "."
    punctuations = ['!', ',', '.', ';', '?']
    for punctuation in punctuations:
        text_kogler = text_kogler.replace(" " + punctuation, punctuation)
    print(text_kogler)
    filename = path_results + "/group24_stage2_generation1" + datetime.now().strftime("_%m%d_%H%M") + ".txt"
    with open(filename, "w", encoding="UTF8") as f:
        f.write(generated_text)


def prep_file_eval(text_kogler, text_kickl, generated_text_kogler, generated_text_kickl):
    text_kogler = ' '.join(text_kogler)
    text_kickl = ' '.join(text_kickl)

    text_kogler = text_kogler.replace(" ,", "")
    text_kickl = text_kickl.replace(" ,", "")
    punctuations = ['!', '.', ';', '?']
    for punctuation in punctuations:
        text_kogler = text_kogler.replace(" " + punctuation + " ", "\n")
        text_kickl = text_kickl.replace(" " + punctuation + " ", "\n")

    text_kogler = [i.split(" ") for i in text_kogler.split("\n")]
    text_kickl = [i.split(" ") for i in text_kickl.split("\n")]

    generated_text_kogler = generated_text_kogler.replace(",", "")
    generated_text_kickl = generated_text_kickl.replace(",", "")
    punctuations = ['!', '.', ';', '?']
    for punctuation in punctuations:
        generated_text_kogler = generated_text_kogler.replace(punctuation + " ", "\n")
        generated_text_kickl = generated_text_kickl.replace(punctuation + " ", "\n")

    generated_text_kogler = [i.split(" ") for i in generated_text_kogler.split("\n")]
    generated_text_kickl = [i.split(" ") for i in generated_text_kickl.split("\n")]

    return text_kogler, text_kickl, generated_text_kogler, generated_text_kickl


def eval_task2(text_kogler, text_kickl, generated_text_kogler, generated_text_kickl):
    print(f"Kogler bleu score (0.25, 0.25, 0.25, 0.25): {bleu_score_(text_kogler, generated_text_kogler, (0.25, 0.25, 0.25, 0.25))}")
    print(f"Kogler bleu score (0.50, 0.50, 0.00, 0.00): {bleu_score_(text_kogler, generated_text_kogler, (0.5, 0.5, 0, 0))}")
    print(f"Kogler bleu score (0.50, 0.25, 0.25, 0.00): {bleu_score_(text_kogler, generated_text_kogler, (0.5, 0.25, 0.25, 0))}")

    print(f"Kickl bleu score (0.25, 0.25, 0.25, 0.25): {bleu_score_(text_kickl, generated_text_kickl, (0.25, 0.25, 0.25, 0.25))}")
    print(f"Kickl bleu score (0.50, 0.50, 0.00, 0.00): {bleu_score_(text_kickl, generated_text_kickl, (0.5, 0.5, 0, 0))}")
    print(f"Kickl bleu score (0.50, 0.25, 0.25, 0.00): {bleu_score_(text_kickl, generated_text_kickl, (0.5, 0.25, 0.25, 0))}")

    text_kogler = [' '.join(sentence) for sentence in text_kogler]
    generated_text_kogler = [' '.join(sentence) for sentence in generated_text_kogler]
    text_kickl = [' '.join(sentence) for sentence in text_kickl]
    generated_text_kickl = [' '.join(sentence) for sentence in generated_text_kickl]

    print(f"Kogler ROUGE score: {rouge_score_(text_kogler, generated_text_kogler)}")
    print(f"Kickl ROUGE score: {rouge_score_(text_kickl, generated_text_kickl)}")

    print(f"Kickl BERTScore: {bert_score_(text_kickl, generated_text_kickl)}")
    print(f"Kickl BERTScore: {bert_score_(text_kogler, generated_text_kogler)}")


def main():
    create_directories()
    #read text
    text_kogler = read_file(file1_kogler)
    text_kickl = read_file(file2_kickl)
    #process data
    text_kogler = pre_processing(text_kogler)
    text_kickl = pre_processing(text_kickl)

    # set to True to generate new text, see generate text for model parameters
    generate = False

    if generate:
        generate_text(text_kickl, text_kogler, model_name='rnn128_100_epochs_batchsize_64_dropout_20')

    # set to True to evaluate text, adapt file names below
    evaluate = True

    if evaluate:
        # Evaluate text
        # RNN
        generated_text_kogler = read_file(path_results + "/testsystem_1805/group24_stage2_generation1.txt")
        generated_text_kickl = read_file(path_results + "/testsystem_1805/group24_stage2_generation2.txt")
        text_kogler_eval, text_kickl_eval, gen_kogler_eval, gen_kickl_eval = prep_file_eval(text_kogler, text_kickl,
                                                                                            generated_text_kogler,
                                                                                            generated_text_kickl)
        eval_task2(text_kogler_eval, text_kickl_eval, gen_kogler_eval, gen_kickl_eval)

        # Markov advanced
        generated_text_kogler = read_file(os.curdir + "/results/task2/markov_adv_kogler_maxseq20.txt")
        generated_text_kickl = read_file(os.curdir + "/results/task2/markov_adv_kickl_maxseq20.txt")
        text_kogler_eval, text_kickl_eval, gen_kogler_eval, gen_kickl_eval = prep_file_eval(text_kogler, text_kickl,
                                                                                            generated_text_kogler,
                                                                                            generated_text_kickl)
        eval_task2(text_kogler_eval, text_kickl_eval, gen_kogler_eval, gen_kickl_eval)

        # LSTM with temperature
        generated_text_kogler = read_file(os.curdir + "/results/task2/lstm_kogler_temp1.0_100epochs.txt")
        generated_text_kickl = read_file(os.curdir + "/results/task2/lstm_kickl_temp1.0_100epochs.txt")
        text_kogler_eval, text_kickl_eval, gen_kogler_eval, gen_kickl_eval = prep_file_eval(text_kogler, text_kickl,
                                                                                            generated_text_kogler,
                                                                                            generated_text_kickl)
        eval_task2(text_kogler_eval, text_kickl_eval, gen_kogler_eval, gen_kickl_eval)


if __name__ == '__main__':
    main()