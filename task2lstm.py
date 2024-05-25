import numpy as np
import os
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense

path = os.curdir + "/data"
file1_kogler = path + "/data_stage_2/data_stage2_1_kogler.txt"
file2_kickl = path + "/data_stage_2/data_stage2_2_kickl.txt"
path_results = os.curdir + "/results"
path_models = os.curdir + "/models/task2"


# Function to read the file
def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def save_text(generated_text, title):
    filename = path_results + "/" + title + ".txt"
    with open(filename, "w", encoding="UTF8") as f:
        f.write(generated_text)

def save_model(model, title):
    model.save(path_models + "/" + title)
    print("Model saved")

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


def lstm_model(text_data):
    # Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text_data])
    total_words = len(tokenizer.word_index) + 1

    word_index = tokenizer.word_index

    # Print all the tokenized words
    for word, index in word_index.items():
        if len(word) == 1:
            print(f"{word}: {index}")

    # Convert text to sequences of integers
    input_sequences = []
    for line in text_data.split('.'):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    # Pad sequences to ensure uniform length
    max_sequence_len = max(len(seq) for seq in input_sequences)
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # Create predictors and label
    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = to_categorical(y, num_classes=total_words)

    # Define the model
    model = Sequential([
        Embedding(total_words, 100, input_length=max_sequence_len-1),
        LSTM(150, return_sequences=True),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(total_words, activation='softmax')
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X, y, epochs=100, verbose=1)
    model.summary()

    return model, max_sequence_len, tokenizer



# Function to generate text
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / temperature  # Adding small constant to avoid log(0)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(seed_text, next_words, model, max_sequence_len, tokenizer, temperature=1.0):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        preds = model.predict(token_list, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_word = tokenizer.index_word[next_index]
        seed_text += " " + next_word

    return seed_text





# Generate text

##temperature of 1.0 Should not change prediction
##temperature < 1.0: Sharpens probability distribution
##temperature > 1.0: Flattens probability distribution
## -> difference to use of temperature in ChatGPT, others

text_data1 = read_file(file1_kogler)
processed_text1 = pre_processing(text_data1)
model1, max_sequence_len1, tokenizer1 = lstm_model(" ".join(processed_text1))

seed_text1 = "kogler:"
next_words1 = 150
temperature1 = 1.0
generated_text1 = generate_text(seed_text1, next_words1, model1, max_sequence_len1, tokenizer1, temperature1)
save_text(generated_text1, "lstm_kogler_temp1.0_100epochs")
save_model(model1, "kogler_lstm_100epochs_temp1.0.keras")
print(generated_text1)

# Process second text
text_data2 = read_file(file2_kickl)
processed_text2 = pre_processing(text_data2)
model2, max_sequence_len2, tokenizer2 = lstm_model(" ".join(processed_text2))

seed_text2 = "kickl:"
next_words2 = 150
temperature2 = 1.0
generated_text2 = generate_text(seed_text2, next_words2, model2, max_sequence_len2, tokenizer2, temperature2)
save_text(generated_text2, "lstm_kickl_temp1.0_100epochs")
save_model(model2, "kickl_lstm_100epochs_temp1.0.keras")
print(generated_text2)
