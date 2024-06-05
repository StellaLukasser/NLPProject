import numpy as np
import pandas as pd
import re


def clean_tweet_musk(tweet):
    tweet = re.sub(r'\n', ' ', tweet)
    original = tweet
    tweet = tweet + "\n"
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#\w+', '', tweet)

    tweet = re.sub(r'(\d+(\.|,|K| )*)+\n', ' ', tweet)
    tweet = re.sub("\w+\.com", " ", tweet)
    tweet = re.sub("\s+", " ", tweet).strip()
    tweet = re.sub(r'Replying to (and )*', '', tweet)
    print(original)
    print(tweet)
    print()
    return tweet


def clean_tweet_trump(tweet):
    tweet = re.sub(r'\n', ' ', tweet)
    original = tweet
    tweet = tweet + "\n"
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#\w+', '', tweet)

    tweet = re.sub(r'(\d+(\.|,|K| )*)+\n', ' ', tweet)
    tweet = re.sub("\w+\.com", " ", tweet)
    tweet = re.sub("\s+", " ", tweet).strip()
    tweet = re.sub(r'RT : *', '', tweet)
    print(original)
    print(tweet)
    print()
    return tweet


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


if __name__ == '__main__':
    main()
