import numpy as np
import os
import random
import pandas as pd
import openpyxl
import torch
import re
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import replicate
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional
import transformers
import torch
from torch import cuda, bfloat16

from transformers import BitsAndBytesConfig
from task3preprocessing import preprocessing
pd.set_option('display.max_colwidth', 170)

#https://armandolivares.tech/2022/09/16/how-to-create-a-tweet-generator-with-gpt-2/

print(torch.cuda.is_available())

path = os.curdir + "/data"
file1_musk = path + "/data_stage_3/data_stage3_1_musk.xlsx"
file2_trump = path + "/data_stage_3/data_stage3_2_trump.xlsx"
path_results = os.curdir + "/results/task3"
path_models = os.curdir + "/models/task3"

model_musk_path = path_models + "/model_musk"
model_trump_path = path_models + "/model_trump"
token_musk_path = path_models + "/token_musk"
token_trump_path = path_models + "/token_trump"
output_musk_path = path_models + "/output_musk"
output_trump_path = path_models + "/output_trump"


def save_data(data, path):

    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not isinstance(data, str):
        data = '\n'.join(data)

    with open(path, 'w') as file:
        file.write(data)


def fine_tune(text_data, output_dir, model_path, token_path, epochs=5, batch_size=4, gradient_accumulation_steps=4,
              learning_rate=1e-5, warmup_steps=500, max_length=128, account="elonmusk"):

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2',
                                              bos_token='<|startoftext|>',
                                              eos_token='<|endoftext|>',
                                              pad_token='<|pad|>')

    # Prepare the text data

    text_data = f'<|startoftext|>' + account + ": " + text_data['tweets'] + '<|endoftext|>'
    text_data = pd.DataFrame(text_data)

    dataset = Dataset.from_pandas(text_data)

    #print(dataset)

    # Tokenize the data
    def preprocess(example):
        return tokenizer(example['tweets'], truncation=True)

    dataset = dataset.map(preprocess, batched=False)
    train_test_split = dataset.train_test_split(train_size=.8)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    # Load the GPT-2 model
    model = GPT2LMHeadModel.from_pretrained('gpt2', n_layer=5)
    model.resize_token_embeddings(len(tokenizer))

    # Clear cache
    torch.cuda.empty_cache()

    # Initialize the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=batch_size,
        load_best_model_at_end=True,
        logging_steps=5,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        fp16=True,  # Enable mixed precision training
        seed=38,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator
    )
    trainer.train()
    model.save_pretrained(model_path)

    # Save the tokenizer
    tokenizer.save_pretrained(token_path)
    return


def test_model(prompt, model, tokenizer):
    #########testing###########
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    prompt = "<|startoftext|>" + prompt + "<|endoftext|><|startoftext|>I would answer:"

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)

    print(generated)

    sample_outputs = model.generate(
        generated,
        do_sample=True,
        top_k=20,
        max_length=300,
        top_p=0.95,
        early_stopping=True,
        no_repeat_ngram_size=2,
        num_return_sequences=5,
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=torch.ones_like(generated)
    )

    for i, sample_output in enumerate(sample_outputs):
        print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))


def generate_tweet(prompt, model, tokenizer):
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)

    #print(generated)

    output = model.generate(
        generated,
        do_sample=True,
        top_k=20,
        max_new_tokens=300,
        top_p=0.92,
        early_stopping=False,
        no_repeat_ngram_size=2,
        num_beams=5,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=torch.ones_like(generated)
    )

    generated_tweet = tokenizer.decode(output[0], skip_special_tokens=True)
    # take part of string after "I would answer:"
    generated_tweet = generated_tweet.split("I would answer ")[1]
    return generated_tweet


def discuss(text1, text2, n=5):

    model1 = GPT2LMHeadModel.from_pretrained(model_musk_path)
    tokenizer1 = GPT2Tokenizer.from_pretrained(token_musk_path)

    model2 = GPT2LMHeadModel.from_pretrained(model_trump_path)
    tokenizer2 = GPT2Tokenizer.from_pretrained(token_trump_path)

    if model1.config.n_positions == model2.config.n_positions:
        max_length = model1.config.n_positions
    else:
        print("Models have different max length!!")
        max_length = model1.config.n_positions

    generated_tweets = []

    history = ""

    with open("../data/data_stage_3/initial_tweet_musk.txt", "r") as file:
        first_prompt = file.read().replace('\n', '')

    start_prompt = "<|startoftext|>elonmusk: " + first_prompt + "I would answer realDonaldTrump:"
    tweet1 = generate_tweet(start_prompt, model2, tokenizer2)
    generated_tweets.append("elonmusk: " + first_prompt)
    generated_tweets.append(tweet1)
    history += "elonmusk: " + first_prompt + " " + tweet1

    for _ in range(n):

        new_prompt = "<|startoftext|>Given the chat: " + history + "\n" + "I would answer elonmusk:"
        if len(new_prompt) > max_length:
            new_prompt = new_prompt[-max_length:]

        tweet2 = generate_tweet(new_prompt, model1, tokenizer1)
        generated_tweets.append(tweet2)
        history += " " + tweet2

        new_prompt = "<|startoftext|>Given the chat: " + history + "\n" + "I would answer realDonaldTrump:"
        if len(new_prompt) > max_length:
            new_prompt = new_prompt[-max_length:]

        tweet2 = generate_tweet(new_prompt, model2, tokenizer2)
        generated_tweets.append(tweet2)
        history += " " + tweet2

    for tweet in generated_tweets:
        print(tweet)

    return generated_tweets


def main():

    #preprocessing()

    tweets_cleaned_musk = pd.read_csv('tweets_cleaned_musk.csv', encoding='utf-8', sep=':')
    tweets_cleaned_trump = pd.read_csv('tweets_cleaned_trump.csv', encoding='utf-8', sep=':')

    #print(tweets_cleaned_musk.head[50])
    #print(tweets_cleaned_trump.head[50])

    fine_tune(tweets_cleaned_musk, output_musk_path, model_musk_path, token_musk_path, epochs=25, account="elonmusk")
    fine_tune(tweets_cleaned_trump, output_trump_path, model_trump_path, token_trump_path, epochs=25, account="realDonaldTrump")

    model1 = GPT2LMHeadModel.from_pretrained(model_musk_path)
    tokenizer1 = GPT2Tokenizer.from_pretrained(token_musk_path)

    model2 = GPT2LMHeadModel.from_pretrained(model_trump_path)
    tokenizer2 = GPT2Tokenizer.from_pretrained(token_trump_path)

    #test_model("Too much concentration is the only real issue imo.", model1, tokenizer1)
    #test_model("Too much concentration is the only real issue imo.", model2, tokenizer2)


    generated_tweets = discuss(tweets_cleaned_musk, tweets_cleaned_trump, n=100)
    #
    # for i, tweet in enumerate(generated_tweets_raw):
    #     print(i)
    #     print(tweet)
    #
    save_data(generated_tweets, path_results + "/generated_tweets.txt")


if __name__ == '__main__':
    main()
