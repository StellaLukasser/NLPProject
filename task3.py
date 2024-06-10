import numpy as np
import pandas as pd
import re
import random

pd.set_option('display.max_colwidth', 170)


def clean_tweet_musk(tweet):
    tweet = re.sub(r'\n', ' ', tweet)
    tweet = tweet + "\n"
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#\w+', '', tweet)

    tweet = re.sub(r'(\d+(\.|,|K| )*)+\n', ' ', tweet)
    tweet = re.sub("\w+\.com", " ", tweet)
    tweet = re.sub("\s+", " ", tweet).strip()
    tweet = re.sub(r'Replying to (and )*', '', tweet)

    return tweet


def clean_tweet_trump(tweet):
    tweet = re.sub(r'\n', ' ', tweet)
    tweet = tweet + "\n"
    tweet = re.sub(r'RT @realDonaldTrump:', '', tweet)
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#\w+', '', tweet)

    tweet = re.sub(r'(\d+(\.|,|K| )*)+\n', ' ', tweet)
    tweet = re.sub("\w+\.com", " ", tweet)
    tweet = re.sub("\s+", " ", tweet).strip()
    #tweet = re.sub(r'RT : *', '', tweet)
    tweet = re.sub(r":", "", tweet)

    tweet = re.sub(r"â€™", "'", tweet)
    tweet = re.sub(r"â€“", "—", tweet)
    tweet = re.sub(r"â€”", "–", tweet)
    tweet = re.sub(r"â€˜", "‘", tweet)

    tweet = re.sub(r"â€¦", "...", tweet)

    # &amp => "An ampersand" => and
    tweet = re.sub(r'&amp', 'and', tweet)

    tweet = tweet.lstrip(" .,")
    tweet = tweet.rstrip()

    return tweet

def preprocessing():
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

    #clean unnecessary chars from html encoding
    chars = sorted(list(set(''.join(tweets_cleaned_trump))))
    char_indices = dict((cd, i) for i, cd in enumerate(chars))
    print("Chars in trump text", char_indices)

    #removing unwanted chars 163-85 trump
    for c in chars[-78:]:
        tweets_cleaned_trump = tweets_cleaned_trump.str.replace(c,'')

    # Remove tweets shorter than 20 chars
    tweets_cleaned_trump = tweets_cleaned_trump[tweets_cleaned_trump.map(len) > 20]

    chars = sorted(list(set(''.join(tweets_cleaned_musk))))
    char_indices = dict((cd, i) for i, cd in enumerate(chars))
    print("Chars in musk text", char_indices)

    #removing unwanted chars 205-90 trump
    for c in chars[-116:]:
        tweets_cleaned_musk = tweets_cleaned_musk.str.replace(c,'')

    # Remove tweets shorter than 20 chars
    tweets_cleaned_musk = tweets_cleaned_musk[tweets_cleaned_musk.map(len) > 20]

    tweets_cleaned_musk.dropna(inplace=True)
    tweets_cleaned_trump.dropna(inplace=True)

    tweets_cleaned_trump_filtered = tweets_cleaned_trump[~tweets_cleaned_trump.str.startswith('RT')]

    tweets_cleaned_trump_filtered.to_csv('tweets_cleaned_trump.csv', index=False, encoding='utf-8')
    tweets_cleaned_musk.to_csv('tweets_cleaned_musk.csv', index=False, encoding='utf-8')
    return tweets_cleaned_musk, tweets_cleaned_trump


def main():

    preprocessing()
    trump = pd.read_csv('tweets_cleaned_trump.csv', encoding='utf-8', sep=':')
    musk = pd.read_csv('tweets_cleaned_trump.csv', encoding='utf-8', sep=':')


if __name__ == '__main__':
    main()
