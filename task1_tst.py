import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

path_processed_text = os.curdir + "/processed_data" + "/sequenced_data_task1.txt"
path_processed_data = os.curdir + "/processed_data"
path_results = os.curdir + "/results"
path_generated_text = os.curdir + "/results" + "/group24_stage1_generation"

tokenizer = AutoTokenizer.from_pretrained("ainize/gpt2-spongebob-script-large")
model = AutoModelForCausalLM.from_pretrained("ainize/gpt2-spongebob-script-large")


def load_sequences_from_txt(path):
    sequences = []
    with open(path, "r") as f:
        for line in f:
           sequences.append(line)
    return sequences


def load_sequenced_text(path):

    if os.path.exists(path):
        sequences = load_sequences_from_txt(path)
        print("Loaded sequences")
    else: 
        print("sequenced data missing")
    
    return sequences


def load_text(path):
    with open(path, "r", encoding="UTF8") as f:
        return f.read()
    

def transfer_style_sequences(sequences):

    transferred_texts = []
    i = 0
    for text in sequences:
        input_ids = tokenizer.encode(text, return_tensors="pt")

        if input_ids.size(1) == 0:
            transferred_texts.append("Input sequence is empty or contains only padding tokens.")
            continue
        
        output_ids = model.generate(input_ids, max_length=len(input_ids[0])+50, num_return_sequences=1)[0]
        transferred_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        transferred_texts.append(transferred_text)
        i += 1
        if i == 10:
            break


    return transferred_texts


def sample_random_chunks(input_text, N, chunksize):

    num_chunks = (len(input_text) + chunksize - 1) // chunksize
    chunks = [input_text[i*chunksize:(i+1)*chunksize] for i in range(num_chunks)]
    
    if len(chunks) <= N:
        return chunks
    
    random_chunks = random.sample(chunks, N)
    return random_chunks


def transfer_style_generated_random(input_text):
    
    #number of chunks from text to sample
    N = 10

    #length of each chunk
    chunksize = 200

    sequences = sample_random_chunks(input_text, N, chunksize)

    transferred_chunks = []

    for text in sequences:
        inputs = tokenizer(text, return_tensors="pt", max_length=len(text))
        print("Tokenized Inputs:", inputs)

        output_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=chunksize*2, num_return_sequences=1)[0]
        print("Output IDs:", output_ids)
        
        transferred_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        print(transferred_text)

        transferred_chunks.append(transferred_text)
    
    transferred_text = " ".join(transferred_chunks)

    return transferred_text




def transfer_style_generated(input_text, max_chunk_length=8):
    
    transferred_chunks = []

    for start_idx in range(0, len(input_text), max_chunk_length):
        end_idx = min(start_idx + max_chunk_length, len(input_text))
        print("Chunk Start Index:", start_idx)
        print("Chunk End Index:", end_idx)
        
        input_chunk = input_text[start_idx:end_idx]
        print("Input Chunk:", input_chunk)

        inputs = tokenizer(input_chunk, return_tensors="pt", max_length=max_chunk_length, truncation=True)
        print("Tokenized Inputs:", inputs)

        output_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=max_chunk_length + 50, num_return_sequences=1)[0]
        print("Output IDs:", output_ids)
        
        transferred_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        transferred_chunks.append(transferred_text)

    transferred_text = " ".join(transferred_chunks)
    
    return transferred_text


def sample_random_sequence(input_text, N):

    max_start_idx = len(input_text) - N
    
    if max_start_idx <= 0:
        return input_text
    else:
        start_idx = random.randint(0, max_start_idx)
        random_sequence = input_text[start_idx:start_idx+N]
        return random_sequence

def transfer_style_generated_sequence(input_text):
    
    N = 100
    #sample a random sequence of size N from the input text
    input_sequence = sample_random_sequence(input_text, N)

    i = 0

    new_text = ""

    while i < 50:
        
        inputs = tokenizer(input_sequence, return_tensors="pt", max_length=N, truncation=True)
        output_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=N, num_return_sequences=1)[0]
        transferred_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        new_text += transferred_text.replace(input_sequence, "")

        random_new_sequence = sample_random_sequence(input_text, N//2)
        input_sequence = random_new_sequence + transferred_text[len(transferred_text) - N//2:]
        
        i += 1
        
        print("tt:", transferred_text)
        print("new_text:", new_text)
        print("input:", input_sequence)

    return new_text


#print(type(load_sequenced_text(path_processed_text)))
 
#print(transfer_style(load_sequenced_text(path_processed_text)))

#print(type(load_text(path_generated_text)))

#print(transfer_style_generated(load_text(path_generated_text)))

#print(sample_random_chunks(load_text(path_generated_text), 10, 50))

#print(transfer_style_generated_random(load_text(path_generated_text)))

print("final:", transfer_style_generated_sequence(load_text(path_generated_text)))