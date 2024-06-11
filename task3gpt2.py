import numpy as np
import os
import random
import pandas as pd
import openpyxl
import torch
import re
import ast
import accelerate
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
pd.set_option('display.max_colwidth', 170)

#https://armandolivares.tech/2022/09/16/how-to-create-a-tweet-generator-with-gpt-2/

print(torch.cuda.is_available())

path = os.curdir + "/data"
file1_musk = path + "/data_stage_3/data_stage3_1_musk.xlsx"
file2_trump = path + "/data_stage_3/data_stage3_2_trump.xlsx"
path_results = os.curdir + "/results"
path_models = os.curdir + "/models/task3"

model_musk_path = path_models + "/model_musk"
model_trump_path = path_models + "/model_trump"
token_musk_path = path_models + "/token_musk"
token_trump_path = path_models + "/token_trump"
output_musk_path = path_models + "/output_musk"
output_trump_path = path_models + "/output_trump"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


#preprocessing#
#copypasted from edina

def clean_tweet_musk(tweet):
    tweet = re.sub(r'\n', ' ', tweet)
    original = tweet
    tweet = tweet + "\n"
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'@\w+', '', tweet)
    #tweet = re.sub(r'#\w+', '', tweet)

    tweet = re.sub(r'(\d+(\.|,|K| )*)+\n', ' ', tweet)
    tweet = re.sub("\w+\.com", " ", tweet)
    tweet = re.sub("\s+", " ", tweet).strip()
    tweet = re.sub(r'Replying to (and )*', '', tweet)
    # print(original)
    # print(tweet)
    # print()
    return tweet


def clean_tweet_trump(tweet):
    tweet = re.sub(r'\n', ' ', tweet)
    original = tweet
    tweet = tweet + "\n"
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'@\w+', '', tweet)
    #tweet = re.sub(r'#\w+', '', tweet)

    tweet = re.sub(r'(\d+(\.|,|K| )*)+\n', ' ', tweet)
    tweet = re.sub("\w+\.com", " ", tweet)
    tweet = re.sub("\s+", " ", tweet).strip()
    tweet = re.sub(r'RT : *', '', tweet)
    # â€™ as '
    tweet = re.sub(r"â€™", "'", tweet)
    tweet = re.sub(r"â€“", "—", tweet)
    tweet = re.sub(r"â€”", "–", tweet)
    tweet = re.sub(r"â€˜", "‘", tweet)
    #tweet = re.sub(r"â€œ", "“", tweet)#“
    #tweet = re.sub(r"â€", "”", tweet)#”
    tweet = re.sub(r"â€¦", "...", tweet)
    #tweet = bytes(tweet, "cp1252").decode("utf-8")
    # &amp => "An ampersand" => and
    tweet = re.sub(r'&amp', 'and', tweet)
    #remove all non-alphanumeric and _
    #tweet = re.sub(r'\W+', ' ', tweet)
    #remove new line
    tweet = tweet.rstrip()
    # Tweets longer than 50 chars
    #tweet = tweet[tweet.map(len) > 50]
    # print(original)
    # print(tweet)
    # print()
    return tweet



###############


def get_capitals(textdata):
    capital_words = []

    for text in textdata:
        words = text.split()
        for word in words:
            if word[0].isupper():
                capital_words.append(word)

    if not capital_words:
        print("No capitals!")
        return None

    return capital_words


def fine_tune(text_data, output_dir, model_path, token_path, epochs=2, batch_size=2, gradient_accumulation_steps=4,
              learning_rate=5e-5, warmup_steps=500, max_length=128):
    # Initialize the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

    # Prepare the text data
    text_data = pd.DataFrame({'text': [f' {text}' for text in text_data]})
    dataset = Dataset.from_pandas(text_data)

    # Tokenize the data
    def preprocess(example):
        return tokenizer(example['text'], truncation=True, padding='max_length', max_length=max_length)

    dataset = dataset.map(preprocess, batched=True)

    # Data collator for dynamic padding
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Split the dataset
    train_test_split = dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']

    # Load the GPT-2 model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
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

def generate_tweet(prompt, model, tokenizer):

    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate the tweet
    # max_length: Specifies the maximum length of the generated tweet.
    # num_return_sequences: Specifies the number of different sequences to generate.
    # no_repeat_ngram_size: Specifies the size of n-grams to avoid repeating in the generated sequences.
    # early_stopping: Stops generation when the model predicts an end-of-text token.
    outputs = model.generate(inputs, num_return_sequences=1, no_repeat_ngram_size=2, num_beams=5, early_stopping=True, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id, attention_mask=torch.ones_like(inputs))

    generated_tweet = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_tweet


def discuss(text1, text2, n=5):

    model1 = GPT2LMHeadModel.from_pretrained(model_musk_path)
    tokenizer1 = GPT2Tokenizer.from_pretrained(token_musk_path)

    model2 = GPT2LMHeadModel.from_pretrained(model_trump_path)
    tokenizer2 = GPT2Tokenizer.from_pretrained(token_trump_path)

    capitals_musk = get_capitals(text1)
    capitals_trump = get_capitals(text2)

    generated_tweets = []
    history = ""
    last_prompt = ""

    start_with = random.choice(["musk", "trump"])
    if start_with == "musk":
        start_prompt = "Elon tweets: " + random.choice(capitals_musk)
        tweet1 = generate_tweet(start_prompt, model1, tokenizer1)
        generated_tweets.append(tweet1)
        history += tweet1
        last_prompt = tweet1
    else:
        start_prompt = "Donald tweets: " + random.choice(capitals_trump)
        tweet2 = generate_tweet(start_prompt, model2, tokenizer2)
        generated_tweets.append(tweet2)
        history += tweet2
        last_prompt = tweet2

    for _ in range(n):

        if start_with == "musk":
            print(last_prompt)
            new_prompt = last_prompt + "\n\n" + "Donald tweets: " + random.choice(capitals_trump)
            tweet2 = generate_tweet(new_prompt, model2, tokenizer2)
            tweet2 = tweet2[len(last_prompt):]
            generated_tweets.append(tweet2)
            history += tweet2
            last_prompt = tweet2
            print(last_prompt)

            new_prompt = last_prompt + "\n\n" + "Elon tweets: " + random.choice(capitals_musk)
            tweet1 = generate_tweet(new_prompt, model1, tokenizer1)
            tweet1 = tweet1[len(last_prompt):]
            generated_tweets.append(tweet1)
            history += tweet1
            last_prompt = tweet1

        else:
            print(last_prompt)

            new_prompt = last_prompt + "\n\n" + "Elon tweets: " + random.choice(capitals_musk)
            tweet1 = generate_tweet(new_prompt, model1, tokenizer1)
            tweet1 = tweet1[len(last_prompt):]
            generated_tweets.append(tweet1)
            history += tweet1
            last_prompt = tweet1
            print(last_prompt)

            new_prompt = last_prompt + "\n\n" + "Donald tweets: " + random.choice(capitals_trump)
            tweet2 = generate_tweet(new_prompt, model2, tokenizer2)
            tweet2 = tweet2[len(last_prompt):]
            generated_tweets.append(tweet2)
            history += tweet2
            last_prompt = tweet2


    print(history)

    return generated_tweets

def discuss2(text1, text2, n=5):

    model1 = GPT2LMHeadModel.from_pretrained(model_musk_path)
    tokenizer1 = GPT2Tokenizer.from_pretrained(token_musk_path)

    model2 = GPT2LMHeadModel.from_pretrained(model_trump_path)
    tokenizer2 = GPT2Tokenizer.from_pretrained(token_trump_path)

    capitals_musk = get_capitals(text1)
    capitals_trump = get_capitals(text2)

    generated_tweets = []
    history = ""
    last_prompt = ""

    start_with = random.choice(["musk", "trump"])
    if start_with == "musk":
        start_prompt = "Elon Musk tweets: " + random.choice(capitals_musk)
        tweet1 = generate_tweet(start_prompt, model1, tokenizer1)
        generated_tweets.append(tweet1)
        history += tweet1
        last_prompt = tweet1
    else:
        start_prompt = "Donald Trump tweets: " + random.choice(capitals_trump)
        tweet2 = generate_tweet(start_prompt, model2, tokenizer2)
        generated_tweets.append(tweet2)
        history += tweet2
        last_prompt = tweet2

    for _ in range(n):

        if start_with == "musk":
            print(last_prompt)
            new_prompt = last_prompt + "\n\n" + "Donald Trump answers: "
            tweet2 = generate_tweet(new_prompt, model2, tokenizer2)
            tweet2 = tweet2[len(last_prompt):]
            generated_tweets.append(tweet2)
            history += tweet2
            last_prompt = tweet2
            print(last_prompt)

            new_prompt = last_prompt + "\n\n" + "Elon Musk answers: "
            tweet1 = generate_tweet(new_prompt, model1, tokenizer1)
            tweet1 = tweet1[len(last_prompt):]
            generated_tweets.append(tweet1)
            history += tweet1
            last_prompt = tweet1

        else:
            print(last_prompt)

            new_prompt = last_prompt + "\n\n" + "Elon Musk answers: "
            tweet1 = generate_tweet(new_prompt, model1, tokenizer1)
            tweet1 = tweet1[len(last_prompt):]
            generated_tweets.append(tweet1)
            history += tweet1
            last_prompt = tweet1
            print(last_prompt)

            new_prompt = last_prompt + "\n\n" + "Donald Trump answers: "
            tweet2 = generate_tweet(new_prompt, model2, tokenizer2)
            tweet2 = tweet2[len(last_prompt):]
            generated_tweets.append(tweet2)
            history += tweet2
            last_prompt = tweet2


    print(history)

    return generated_tweets




def main():


    df_musk = pd.read_excel('data/data_stage_3/data_stage3_1_musk.xlsx', header=None, engine='openpyxl')
    tweets_musk = df_musk.iloc[:, 0]

    df_trump = pd.read_excel('data/data_stage_3/data_stage3_2_trump.xlsx', header=None, engine='openpyxl')
    tweets_trump = df_trump.iloc[:, 0]

    tweets_cleaned_musk = tweets_musk.apply(clean_tweet_musk)
    tweets_cleaned_trump = tweets_trump.apply(clean_tweet_trump)
    tweets_cleaned_musk.replace('', np.nan, inplace=True)
    tweets_cleaned_trump.replace('', np.nan, inplace=True)
    tweets_cleaned_musk.dropna(inplace=True)
    tweets_cleaned_trump.dropna(inplace=True)

    print(tweets_cleaned_musk.head(50))
    print(tweets_cleaned_trump.head(50))



    #fine_tune(tweets_cleaned_musk, output_musk_path, model_musk_path, token_musk_path)
    #fine_tune(tweets_cleaned_trump, output_trump_path, model_trump_path, token_trump_path)


    #print(generate_tweet("SpaceX", model, tokenizer))


    tweets = discuss(tweets_cleaned_musk, tweets_cleaned_trump)
    #for tweet in tweets:
    #    print(tweet)

    #print(generate_tweet(prompt_musk))





    #initial_tweet = random.choice(tweets_musk)


    #new_tweet_musk = generate_tweet("do something")


    #style_transferred_tweet = new_tweet_musk

    # Generate a new tweet based on the style-transferred tweet in Trump's style
    #new_tweet_trump = generate_tweet(style_transferred_tweet)

    # Print the results
    #print("Initial Tweet (Elon Musk):", initial_tweet)
    #print("Generated Tweet (Musk's Style):", new_tweet_musk)
    #print("Style-Transferred Tweet (Trump's Style):", style_transferred_tweet)
    #print("Generated Tweet (Trump's Style):", new_tweet_trump)



if __name__ == '__main__':
    main()