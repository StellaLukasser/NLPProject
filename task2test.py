import numpy as np
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense


# Function to read the file
def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


# Paths
path = os.curdir + "/data"
file1_kogler = path + "/data_stage_2/data_stage2_1_kogler.txt"
file2_kickl = path + "/data_stage_2/data_stage2_2_kickl.txt"
text_data = read_file(file2_kickl)
path_results = os.curdir + "/results"

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_data])
total_words = len(tokenizer.word_index) + 1

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
model.summary()

# Train the model
history = model.fit(X, y, epochs=100, verbose=1)


# Function to generate text
def generate_text(seed_text, next_words, model, max_sequence_len, temperature=1.0):
    def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds + 1e-10) / temperature  # Adding small constant to avoid log(0)
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        preds = model.predict(token_list, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_word = tokenizer.index_word[next_index]

        seed_text += " " + next_word

    return seed_text

def save_text(generated_text, title):
    filename = path_results + "/" + title + ".txt"
    with open(filename, "w", encoding="UTF8") as f:
        f.write(generated_text)


# Generate text
seed_text = "kickl:"
next_words = 150
temperature = 1.0
generated_text = generate_text(seed_text, next_words, model, max_sequence_len, temperature)
save_text(generated_text, "group24_stage2_generation_testtemp1.0")
print(generated_text)
