import numpy as np
import os
import random
import pandas as pd
import torch
import re
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
pd.set_option('display.max_colwidth', 170)

#https://armandolivares.tech/2022/09/16/how-to-create-a-tweet-generator-with-gpt-2/

#print(torch.cuda.is_available())

path = os.curdir + "/data"
file1_musk = path + "/data_stage_3/data_stage3_1_musk.xlsx"
file2_trump = path + "/data_stage_3/data_stage3_2_trump.xlsx"

path_results = os.curdir + "/results/task3"
path_models = os.curdir + "/models/task3"

path_processed_data_musk = os.curdir + "/processed_data/task3/tweets_cleaned_musk.txt"
path_processed_data_trump = os.curdir + "/processed_data/task3/tweets_cleaned_trump.txt"

model_musk_path = path_models + "/model_musk"
model_trump_path = path_models + "/model_trump"
token_musk_path = path_models + "/token_musk"
token_trump_path = path_models + "/token_trump"
output_musk_path = path_models + "/output_musk"
output_trump_path = path_models + "/output_trump"
initial_tweet_path = path + "/data_stage_3/initial_tweet_musk.txt"


def save_data(data, path):

    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not isinstance(data, str):
        data = '\n'.join(data)

    with open(path, 'w') as file:
        file.write(data)

def text_file_to_dataframe(file_path, column_name='Column'):

    with open(file_path, 'r') as file:
        lines = file.read().splitlines()

    # Create DataFrame
    df = pd.DataFrame({
        column_name: lines
    })

    return df

def read_file(filename):
    text = ""
    with open(filename, "r", encoding="UTF8") as f:
        for line in f:
            if len(line) > 1:
                text += line
    return text

def clean_tweet_musk(tweet):
    tweet = re.sub(r'\n', ' ', tweet)                                         # remove line breaks
    tweet = tweet + "\n"                                                      # add break at the end (helper)
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE) # remove links
    tweet = re.sub(r'@\w+', '', tweet)                                        # remove taggings
    #tweet = re.sub(r'#\w+', '', tweet)                                        # remove hashtags
    tweet = re.sub(r'(\d+(\.|,|K| )*)+\n', ' ', tweet)                        # remove numbers at the end of a line
    tweet = re.sub("\w+\.com", " ", tweet)                                    # remove websites
    tweet = re.sub("\s+", " ", tweet).strip()                                 # remove unnecessary whitespaces
    tweet = re.sub(r'Replying to (and )*', '', tweet)                         # remove "Replying to ... (and ...)"

    return tweet


def clean_tweet_trump(tweet):
    tweet = re.sub(r'\n', ' ', tweet)                                         # remove line breaks
    tweet = tweet + "\n"                                                      # add break at the end (helper)
    tweet = re.sub(r'RT @realDonaldTrump:', '', tweet)
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE) # remove links
    tweet = re.sub(r'@\w+', '', tweet)                                        # remove taggings
    #tweet = re.sub(r'#\w+', '', tweet)                                        # remove hashtags
    tweet = re.sub(r'(\d+(\.|,|K| )*)+\n', ' ', tweet)                        # remove numbers at the end of a line
    tweet = re.sub("\w+\.com", " ", tweet)                                    # remove websites
    tweet = re.sub("\s+", " ", tweet).strip()                                 # remove unnecessary whitespaces
    tweet = re.sub(r":", "", tweet)                                           # remove ":"

    # handle weird twitter export api characters
    tweet = re.sub(r"â€™", "'", tweet)
    tweet = re.sub(r"â€“", "—", tweet)
    tweet = re.sub(r"â€”", "–", tweet)
    tweet = re.sub(r"â€˜", "‘", tweet)

    tweet = re.sub(r"â€¦", "...", tweet)

    # &amp => "An ampersand" => and
    tweet = re.sub(r'&amp', 'and', tweet)

    # remove leading dots, commas and whitespaces
    tweet = tweet.lstrip(" .,")
    tweet = tweet.rstrip()

    return tweet

def preprocessing():
    # read data
    df_musk = pd.read_excel('data/data_stage_3/data_stage3_1_musk.xlsx', header=None, engine='openpyxl')
    tweets_musk = df_musk.iloc[:, 0]

    df_trump = pd.read_excel('data/data_stage_3/data_stage3_2_trump.xlsx', header=None, engine='openpyxl')
    tweets_trump = df_trump.iloc[:, 0]

    # clean data
    tweets_cleaned_musk = tweets_musk.apply(clean_tweet_musk)
    tweets_cleaned_trump = tweets_trump.apply(clean_tweet_trump)
    tweets_cleaned_musk.replace('', np.nan, inplace=True)
    tweets_cleaned_trump.replace('', np.nan, inplace=True)
    tweets_cleaned_musk.dropna(inplace=True)
    tweets_cleaned_trump.dropna(inplace=True)

    # ---------TRUMP------------
    # clean unnecessary chars from html encoding
    chars = sorted(list(set(''.join(tweets_cleaned_trump))))
    char_indices = dict((cd, i) for i, cd in enumerate(chars))
    print("Chars in trump text", char_indices)

    # by removing unwanted chars 163-85 trump
    for c in chars[-78:]:
        tweets_cleaned_trump = tweets_cleaned_trump.str.replace(c,'')

    # Remove tweets shorter than 20 chars
    tweets_cleaned_trump = tweets_cleaned_trump[tweets_cleaned_trump.map(len) > 20]

    # ---------MUSK------------
    # clean unnecessary chars from html encoding
    chars = sorted(list(set(''.join(tweets_cleaned_musk))))
    char_indices = dict((cd, i) for i, cd in enumerate(chars))
    print("Chars in musk text", char_indices)

    # by removing unwanted chars 205-90 trump
    for c in chars[-116:]:
        tweets_cleaned_musk = tweets_cleaned_musk.str.replace(c,'')

    # Remove tweets shorter than 20 chars
    tweets_cleaned_musk = tweets_cleaned_musk[tweets_cleaned_musk.map(len) > 20]

    # remove now empty lines
    tweets_cleaned_musk.dropna(inplace=True)
    tweets_cleaned_trump.dropna(inplace=True)

    # remove retweets which are not from trump
    tweets_cleaned_trump_filtered = tweets_cleaned_trump[~tweets_cleaned_trump.str.startswith('RT')]

    tweets_cleaned_musk = tweets_cleaned_musk.str.strip('"')
    tweets_cleaned_musk.name = 'tweets'

    tweets_cleaned_trump_filtered = tweets_cleaned_trump_filtered.str.strip('"')
    tweets_cleaned_trump_filtered.name = 'tweets'

    # save in csv
    save_data(tweets_cleaned_trump_filtered, path_processed_data_trump)
    save_data(tweets_cleaned_musk, path_processed_data_musk)



###############


def get_capitals(textdata):
    capital_words = []
    pattern = re.compile("^[a-zA-Z]+$")

    for tweet in textdata.tweets:
        words = tweet.split()
        if words:  # Check if the tweet is not empty
            first_word = words[0]
            if first_word[0].isupper() and pattern.match(first_word):
                capital_words.append(first_word)

    if not capital_words:
        print("No capitals!")
        return None

    return capital_words


def fine_tune(text_data, output_dir, model_path, token_path, epochs=5, batch_size=2, gradient_accumulation_steps=4,
              learning_rate=1e-5, warmup_steps=1000, max_length=128):

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2',
                                              bos_token='<|startoftext|>',
                                              eos_token='<|endoftext|>',
                                              pad_token='<|pad|>')

    # Prepare the text data

    text_data = f'<|startoftext|> ' + text_data['tweets'] + '<|endoftext|>'
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
    return


def test_model(prompt, model, tokenizer):
    #########testing###########
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    prompt = "<|startoftext|> " + prompt

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
        num_beams=10,
        num_return_sequences=10,
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=torch.ones_like(generated)
    )

    for i, sample_output in enumerate(sample_outputs):
        print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))


def generate_tweet(prompt, sampled_word, model, tokenizer, name):
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    prompt_padded = "<|startoftext|>" + prompt
    prompt_addition = "\n" + f"{name} continues: "
    generated = torch.tensor(tokenizer.encode(prompt_padded + prompt_addition + sampled_word)).unsqueeze(0)
    generated = generated.to(device)

    print("Gen Prompt:")
    print(tokenizer.decode(generated[0]))
    #print(generated)

    output = model.generate(
        generated,
        do_sample=True,
        top_k=15,
        max_new_tokens=300,
        top_p=0.95,
        temperature=1.3,
        early_stopping=True,
        no_repeat_ngram_size=2,
        num_beams=10,
        num_return_sequences=10,
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=torch.ones_like(generated)
    )

    generated_tweet = None

    for sequence in output:
        tweet = tokenizer.decode(sequence, skip_special_tokens=True)
        tweet = tweet[len(prompt) + len(prompt_addition):].strip()

        if len(tweet) > 0:
            generated_tweet = tweet
            break

    if generated_tweet is None:
        generated_tweet = generate_tweet(prompt, sampled_word, model, tokenizer, name)

    return generated_tweet

def style_transfer(prompt, sampled_word, model, tokenizer, name):
    device = torch.device("cuda")

    model.to(device)
    model.eval()

    prompt_padded = f"<|startoftext|>" + prompt
    prompt_addition = "\n" + f"{name} repeats but in his own words: "

    generated = torch.tensor(tokenizer.encode(prompt_padded + prompt_addition + sampled_word)).unsqueeze(0)
    generated = generated.to(device)

    print("Style Prompt:")
    print(tokenizer.decode(generated[0]))
    #print(generated)

    output = model.generate(
        generated,
        do_sample=True,
        top_k=10,
        max_new_tokens=300,
        top_p=0.95,
        temperature=0.9,
        early_stopping=True,
        no_repeat_ngram_size=2,
        num_beams=10,
        num_return_sequences=10,
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=torch.ones_like(generated)
    )

    generated_tweet = None

    for sequence in output:
        tweet = tokenizer.decode(sequence, skip_special_tokens=True)
        tweet = tweet[len(prompt) + len(prompt_addition):].strip()

        if len(tweet) > 0:
            generated_tweet = tweet
            break

    if generated_tweet is None:
        generated_tweet = style_transfer(prompt, sampled_word, model, tokenizer, name)

    return generated_tweet


def task3(text1, text2, n=5):

    model1 = GPT2LMHeadModel.from_pretrained(model_musk_path)
    tokenizer1 = GPT2Tokenizer.from_pretrained(token_musk_path)

    model2 = GPT2LMHeadModel.from_pretrained(model_trump_path)
    tokenizer2 = GPT2Tokenizer.from_pretrained(token_trump_path)

    if model1.config.n_positions == model2.config.n_positions:
        max_length = model1.config.n_positions
    else:
        print("Models have different max length!!")
        max_length = model1.config.n_positions

    capitals_musk = get_capitals(text1)
    capitals_trump = get_capitals(text2)

    generated_tweets = []
    generated_tweets_transferred = []

    history = ""

    initial_tweet = read_file(initial_tweet_path)
    #save initial tweet
    start_prompt = initial_tweet
    history += "Elon Musk @elonmusk\n" + initial_tweet + "\n\n"


    for _ in range(n):

        # generate new musk tweet
        sampled_word = random.choice(capitals_musk)
        tweet_generated = generate_tweet(start_prompt, sampled_word, model1, tokenizer1, "Elon Musk")
        generated_tweets.append("musk: " + tweet_generated)
        # style transfer to trump
        sampled_word = random.choice(capitals_trump)
        tweet_transferred = style_transfer(tweet_generated, sampled_word, model2, tokenizer2, "Donald Trump")
        generated_tweets_transferred.append("trump: " + tweet_transferred)
        history += "Donald J Trump @realDonaldTrump\nreplying to @elonmusk\n" + tweet_transferred + "\n\n"

        #set transferred tweet as new start prompt
        start_prompt = tweet_transferred

        # generate new trump tweet
        sampled_word = random.choice(capitals_trump)
        tweet_generated = generate_tweet(start_prompt, sampled_word, model2, tokenizer2, "Donald Trump")
        generated_tweets.append("trump: " + sampled_word + tweet_generated)
        # style transfer to musk
        sampled_word = random.choice(capitals_musk)
        tweet_transferred = style_transfer(tweet_generated, sampled_word, model1, tokenizer1, "Elon Musk")
        generated_tweets_transferred.append("musk: " + tweet_transferred)
        history += "Elon Musk @elonmusk\nreplying to @realDonaldTrump\n" + tweet_transferred + "\n\n"

        # set transferred tweet as new start prompt
        start_prompt = tweet_transferred

    print(history)

    return generated_tweets, generated_tweets_transferred, history

def main():

    preprocessing()

    tweets_cleaned_musk = text_file_to_dataframe(path_processed_data_musk, 'tweets')
    tweets_cleaned_trump = text_file_to_dataframe(path_processed_data_trump, 'tweets')

    fine_tune(tweets_cleaned_musk, output_musk_path, model_musk_path, token_musk_path, epochs=2)
    fine_tune(tweets_cleaned_trump, output_trump_path, model_trump_path, token_trump_path, epochs=2)

    model1 = GPT2LMHeadModel.from_pretrained(model_musk_path)
    tokenizer1 = GPT2Tokenizer.from_pretrained(token_musk_path)

    model2 = GPT2LMHeadModel.from_pretrained(model_trump_path)
    tokenizer2 = GPT2Tokenizer.from_pretrained(token_trump_path)


    generated_tweets, generated_tweets_transferred, history = task3(tweets_cleaned_musk, tweets_cleaned_trump, n=100)

    save_data(generated_tweets, path_results + "/generated_tweets.txt")
    save_data(generated_tweets_transferred, path_results + "/generated_tweets_transferred.txt")
    save_data(history, path_results + "/history.txt")



if __name__ == '__main__':
    main()