#https://www.scaler.com/topics/deep-learning/text-generation/

import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from keras.preprocessing.text import Tokenizer


path = os.curdir + "/data"
file1_kogler = path + "/data_stage_2/data_stage2_1_kogler.txt"
file2_kickl = path + "/data_stage_2/data_stage2_2_kickl.txt"

def read_file(file):
    f = open(file, "r", encoding="utf-8")
    text = f.read()
    f.close()
    return text

def pre_processing(text):
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

def text_process(text, seed, num_chars):
    # The unique characters in the file
    vocab = sorted(set(text))
    print(f'{len(vocab)} unique characters')
    print(vocab)

    #convert strings to numerical representation
    char_to_int = {ch:i for i, ch in enumerate(vocab)}
    int_to_char = {i:ch for i, ch in enumerate(vocab)}

    # Set the maximum sequence length (max_len) to be the length of the longest sequence
    max_len = max([len(s) for s in text])
    print(max_len)

    # Create training examples and labels
    X = []
    y = []

    for i in range(0, len(text)-max_len, 1):
        X.append([char_to_int[ch] for ch in text[i:i+max_len]])
        y.append(char_to_int[text[i+max_len]])

    print(len(X))
    print("y is ", len(y))
    # Pad the examples
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_len, padding='post')

    # Convert labels to categorical format
    y = tf.keras.utils.to_categorical(y)

    # Define the model architecture
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=len(vocab), output_dim=50, input_length=max_len))
    model.add(tf.keras.layers.LSTM(units=128))
    model.add(tf.keras.layers.Dense(units=len(vocab), activation='softmax'))

    loss1 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss1)#'sparse_categorical_crossentropy')

    # Train the model
    model.fit(X, y, epochs=10, batch_size=64)
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
        generated_text += int_to_char[index]
        # Update the padded seed with the latest character
        padded_seed = np.append(padded_seed[0][1:], index)
        padded_seed = tf.keras.preprocessing.sequence.pad_sequences([padded_seed], maxlen=max_len, padding='post')

    return generated_text


def main():
    #read text
    text_kogler = read_file(file1_kogler)
    text_kickl = read_file(file2_kickl)
    #process data
    text_kogler = pre_processing(text_kogler)
    text_kickl = pre_processing(text_kickl)
    #print(text_kickl)
    #print(text_kogler)

    # Generate text
    generated_text = text_process(text_kickl, 'Kickl', 100)
    print(generated_text)


    # #sequences
    # sequence_length = 30
    # seq_kogler = split_text_into_sequences(text_kogler, sequence_length + 1)
    # seq_kickl = split_text_into_sequences(text_kickl, sequence_length + 1)
    # #print(seq_kogler[:10])
    # #print(seq_kickl[:10])

    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(seq_kogler)

    # # tokenize sequences
    # sequences_tokenized_kogler = tokenizer.texts_to_sequences(seq_kogler)
    # print(sequences_tokenized_kogler[:10])
    # sequences_tokenized_kickl = tokenizer.texts_to_sequences(seq_kickl)
    # print(sequences_tokenized_kickl[:10])

if __name__ == '__main__':
    main()