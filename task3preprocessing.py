import numpy as np
import os
import pandas as pd
import re


path = os.curdir + "/data"
file1_musk = path + "/data_stage_3/data_stage3_1_musk.xlsx"
file2_trump = path + "/data_stage_3/data_stage3_2_trump.xlsx"
path_results = os.curdir + "/results/task3"
path_models = os.curdir + "/models/task3"



def read_file(filename):
    text = ""
    with open(filename, "r", encoding="UTF8") as f:
        for line in f:
            if len(line) > 1:
                text += line
    return text

def clean_tweet_musk(tweet):
    tweet = re.sub(r'\n', ' ', tweet)                                         # remove line breaks
    tweet = tweet + "\n"                                                                  # add break at the end (helper)
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE) # remove links
    tweet = re.sub(r'@\w+', '', tweet)                                        # remove taggings
    #tweet = re.sub(r'#\w+', '', tweet)                                                   # remove hashtags
    tweet = re.sub(r'(\d+(\.|,|K| )*)+\n', ' ', tweet)                        # remove numbers at the end of a line
    tweet = re.sub("\w+\.com", " ", tweet)                                    # remove websites
    tweet = re.sub(r"\d others", "", tweet)                                   # remove "x others"
    tweet = re.sub("\s+", " ", tweet).strip()                                 # remove unnecessary whitespaces
    tweet = re.sub(r'Replying to (and )*', '', tweet)                         # remove "Replying to ... (and ...)"

    return tweet


def clean_tweet_trump(tweet):
    tweet = re.sub(r'\n', ' ', tweet)                                         # remove line breaks
    tweet = tweet + "\n"                                                                  # add break at the end (helper)
    tweet = re.sub(r'RT @realDonaldTrump:', '', tweet)
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE) # remove links
    tweet = re.sub(r'@\w+', '', tweet)                                        # remove taggings
    #tweet = re.sub(r'#\w+', '', tweet)                                                   # remove hashtags
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