import os
import re
from keras.src.preprocessing.text import Tokenizer


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


def main():
    #read text
    text_kogler = read_file(file1_kogler)
    text_kickl = read_file(file2_kickl)
    #process data
    text_kogler = pre_processing(text_kogler)
    text_kickl = pre_processing(text_kickl)
    print(text_kickl)
    print(text_kogler)

    #sequences
    sequence_length = 30
    seq_kogler = split_text_into_sequences(text_kogler, sequence_length + 1)
    seq_kickl = split_text_into_sequences(text_kickl, sequence_length + 1)
    print(seq_kogler[:10])
    print(seq_kickl[:10])

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(seq_kogler)

    # tokenize sequences
    sequences_tokenized_kogler = tokenizer.texts_to_sequences(seq_kogler)
    print(sequences_tokenized_kogler[:10])
    sequences_tokenized_kickl = tokenizer.texts_to_sequences(seq_kickl)
    print(sequences_tokenized_kickl[:10])

if __name__ == '__main__':
    main()