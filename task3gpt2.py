import numpy as np
import os
import random
import pandas as pd
import torch
import re
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from evaluation_metrics import bleu_score_, rouge_score_

pd.set_option('display.max_colwidth', 170)

#https://armandolivares.tech/2022/09/16/how-to-create-a-tweet-generator-with-gpt-2/

#print(torch.cuda.is_available())

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
initial_tweet_path = path + "/data_stage_3/initial_tweet_musk.txt"


#preprocessing#
#copypasted from edina


def save_data(data, path):

    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not isinstance(data, str):
        data = '\n'.join(data)

    with open(path, 'w') as file:
        file.write(data)

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
    tweets_cleaned_trump_filtered.to_csv('tweets_cleaned_trump.csv', index=False, encoding='utf-8', sep=':')
    tweets_cleaned_musk.to_csv('tweets_cleaned_musk.csv', index=False, encoding='utf-8', sep=':')



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


def fine_tune(text_data, output_dir, model_path, token_path, epochs=5, batch_size=4, gradient_accumulation_steps=4,
              learning_rate=1e-5, warmup_steps=500, max_length=128):

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


def generate_tweet(prompt, model, tokenizer):
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    prompt_padded = "<|startoftext|>" + prompt

    generated = torch.tensor(tokenizer.encode(prompt_padded)).unsqueeze(0)
    generated = generated.to(device)

    #print(generated)

    output = model.generate(
        generated,
        do_sample=True,
        top_k=20,
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
        tweet = tweet[len(prompt):].strip()

        if len(tweet) > 0:
            generated_tweet = tweet
            break

    if generated_tweet is None:
        generated_tweet = generate_tweet(prompt, model, tokenizer)

    return generated_tweet

def style_transfer(prompt, model, tokenizer, style):
    device = torch.device("cuda")

    model.to(device)
    model.eval()

    prompt_padded = f"<|startoftext|><{style}>" + prompt

    generated = torch.tensor(tokenizer.encode(prompt_padded)).unsqueeze(0)
    generated = generated.to(device)

    #print(generated)

    output = model.generate(
        generated,
        do_sample=True,
        top_k=50,
        max_new_tokens=300,
        top_p=0.90,
        temperature=1.2,
        early_stopping=True,
        no_repeat_ngram_size=2,
        num_beams=10,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=torch.ones_like(generated)
    )

    generated_tweet = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_tweet = generated_tweet[len(prompt):].strip()

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

    capitals_musk = get_capitals(text1)
    capitals_trump = get_capitals(text2)

    generated_tweets = []
    generated_tweets_raw = []

    history = ""
    last_prompt = ""

    start_with = random.choice(["musk", "trump"])
    if start_with == "musk":
        start_prompt = "<|startoftext|>Elon Musk @elonmusk\n" + random.choice(capitals_musk)
        tweet1 = generate_tweet(start_prompt, model1, tokenizer1)
        generated_tweets.append(start_prompt[len("<|startoftext|>"):] + tweet1)
        generated_tweets_raw.append(start_prompt[len("<|startoftext|>Elon Musk @elonmusk\n"):] + tweet1)
        history += start_prompt[len("<|startoftext|>"):] + tweet1
    else:
        start_prompt = "<|startoftext|>Donald J Trump @realDonaldTrump\n" + random.choice(capitals_trump)
        tweet2 = generate_tweet(start_prompt, model2, tokenizer2)
        generated_tweets.append(start_prompt[len("<|startoftext|>"):] + tweet2)
        generated_tweets_raw.append(start_prompt[len("<|startoftext|>Donald J Trump @realDonaldTrump\n"):] + tweet2)
        history += start_prompt[len("<|startoftext|>"):] + tweet2
    for _ in range(n):

        if start_with == "musk":

            sampled_word = random.choice(capitals_trump)
            new_prompt = history + "\n\n" + "Donald J Trump @realDonaldTrump\nreplying to @elonmusk\n" + sampled_word
            if len(new_prompt) > max_length:
                new_prompt = new_prompt[-max_length:]

            tweet2 = generate_tweet(new_prompt, model2, tokenizer2)
            generated_tweets.append("Donald J Trump @realDonaldTrump\nreplying to @elonmusk\n" + sampled_word + tweet2)
            generated_tweets_raw.append(sampled_word + tweet2)
            history += "\n\n" + "Donald J Trump @realDonaldTrump\nreplying to @elonmusk\n" + sampled_word + tweet2


            sampled_word = random.choice(capitals_musk)
            new_prompt = history + "\n\n" + "Elon Musk @elonmusk\nreplying to @realDonaldTrump\n" + sampled_word
            if len(new_prompt) > max_length:
                new_prompt = new_prompt[-max_length:]

            tweet1 = generate_tweet(new_prompt, model1, tokenizer1)
            generated_tweets.append("Elon Musk @elonmusk\nreplying to @realDonaldTrump\n" + sampled_word + tweet1)
            generated_tweets_raw.append(sampled_word + tweet1)
            history += "\n\n" + "Elon Musk @elonmusk\nreplying to @realDonaldTrump\n" + sampled_word + tweet1

        else:
            sampled_word = random.choice(capitals_musk)
            new_prompt = history + "\n\n" + "Elon Musk @elonmusk\nreplying to @realDonaldTrump\n" + sampled_word
            if len(new_prompt) > max_length:
                new_prompt = new_prompt[-max_length:]

            tweet1 = generate_tweet(new_prompt, model1, tokenizer1)
            generated_tweets.append("Elon Musk @elonmusk\nreplying to @realDonaldTrump\n" + sampled_word + tweet1)
            generated_tweets_raw.append(sampled_word + tweet1)
            history += "\n\n" + "Elon Musk @elonmusk\nreplying to @realDonaldTrump\n" + sampled_word + tweet1


            sampled_word = random.choice(capitals_trump)
            new_prompt = history + "\n\n" + "Donald J Trump @realDonaldTrump\nreplying to @elonmusk\n" + sampled_word
            if len(new_prompt) > max_length:
                new_prompt = new_prompt[-max_length:]

            tweet2 = generate_tweet(new_prompt, model2, tokenizer2)
            generated_tweets.append("Donald J Trump @realDonaldTrump\nreplying to @elonmusk\n" + sampled_word + tweet2)
            generated_tweets_raw.append(sampled_word + tweet2)
            history += "\n\n" + "Donald J Trump @realDonaldTrump\nreplying to @elonmusk\n" + sampled_word + tweet2

    print(history)

    return generated_tweets, generated_tweets_raw

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
        tweet_generated = generate_tweet(start_prompt + " " + sampled_word, model1, tokenizer1)
        generated_tweets.append("musk: " + sampled_word + tweet_generated)
        # style transfer to trump
        tweet_transferred = generate_tweet(tweet_generated, model2, tokenizer2)
        generated_tweets_transferred.append("trump: " + tweet_transferred)
        history += "Donald J Trump @realDonaldTrump\nreplying to @elonmusk\n" + tweet_transferred + "\n\n"

        #set transferred tweet as new start prompt
        start_prompt = tweet_transferred

        # generate new trump tweet
        sampled_word = random.choice(capitals_trump)
        tweet_generated = generate_tweet(start_prompt + " " + sampled_word, model2, tokenizer2)
        generated_tweets.append("trump: " + sampled_word + tweet_generated)
        # style transfer to musk
        tweet_transferred = generate_tweet(tweet_generated, model1, tokenizer1)
        generated_tweets_transferred.append("musk: " + tweet_transferred)
        history += "Elon Musk @elonmusk\nreplying to @realDonaldTrump\n" + tweet_transferred + "\n\n"

        # set transferred tweet as new start prompt
        start_prompt = tweet_transferred

    print(history)

    return generated_tweets, generated_tweets_transferred, history

def main():

    #preprocessing()

    tweets_cleaned_musk = pd.read_csv('tweets_cleaned_musk.csv', encoding='utf-8', sep=':')
    tweets_cleaned_trump = pd.read_csv('tweets_cleaned_trump.csv', encoding='utf-8', sep=':')

    #print(tweets_cleaned_musk.head[50])
    #print(tweets_cleaned_trump.head[50])

    #fine_tune(tweets_cleaned_musk, output_musk_path, model_musk_path, token_musk_path, epochs=22)
    #fine_tune(tweets_cleaned_trump, output_trump_path, model_trump_path, token_trump_path, epochs=40)

    #fine_tune(tweets_cleaned_musk, output_musk_path, model_musk_path, token_musk_path, epochs=3)
    #fine_tune(tweets_cleaned_trump, output_trump_path, model_trump_path, token_trump_path, epochs=2)

    model1 = GPT2LMHeadModel.from_pretrained(model_musk_path)
    tokenizer1 = GPT2Tokenizer.from_pretrained(token_musk_path)

    model2 = GPT2LMHeadModel.from_pretrained(model_trump_path)
    tokenizer2 = GPT2Tokenizer.from_pretrained(token_trump_path)

    #initial_tweet = read_file(initial_tweet_path)
    #test_model("<|startoftext|>If major Dogecoin holders sell most of their coins, it will get my full support. Too much concentration is the only real issue imo.", model1, tokenizer1)
    #test_model("This", model2, tokenizer2)


    #print(generate_tweet("SpaceX", model1, tokenizer1))

    #generated_tweets, generated_tweets_raw = discuss(tweets_cleaned_musk, tweets_cleaned_trump, n=20)

    #for i, tweet in enumerate(generated_tweets_raw):
    #    print(i)
    #    print(tweet)

    #save_data(generated_tweets, path_results + "/generated_tweets.txt")
    #save_data(generated_tweets_raw, path_results + "/generated_tweets_raw.txt")


    generated_tweets, generated_tweets_transferred, history = task3(tweets_cleaned_musk, tweets_cleaned_trump, n=100)

    for i, (tweet_, tweet__) in enumerate(zip(generated_tweets, generated_tweets_transferred)):
        print(i)
        print(tweet_)
        print(tweet__)

    save_data(generated_tweets, path_results + "/generated_tweets.txt")
    save_data(generated_tweets_transferred, path_results + "/generated_tweets_transferred.txt")
    save_data(history, path_results + "/history.txt")

    #initial_tweet = random.choice(tweets_musk)


    #new_tweet_musk = generate_tweet("do something")


    # Generate a new tweet based on the style-transferred tweet in Trump's style
    #new_tweet_trump = generate_tweet(style_transferred_tweet)

    # Print the results
    #print("Initial Tweet (Elon Musk):", initial_tweet)
    #print("Generated Tweet (Musk's Style):", new_tweet_musk)
    #print("Style-Transferred Tweet (Trump's Style):", style_transferred_tweet)
    #print("Generated Tweet (Trump's Style):", new_tweet_trump)



if __name__ == '__main__':
    main()