import numpy as np
import pandas as pd
import re
import random

pd.set_option('display.max_colwidth', 170)


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
    tweet = re.sub(r'#\w+', '', tweet)

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


def main():
    # with open('data/data_stage_3/data_stage3_2_trump.xlsx', encoding='utf-8') as f:
    #     df_trump = pd.read_excel(f)

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

    chars = sorted(list(set(''.join(tweets_cleaned_trump))))
    char_indices = dict((cd, i) for i, cd in enumerate(chars))
    print("Chars in trup text", char_indices)

    #removing unwated 163-85
    for c in chars[-78:]:
        tweets_cleaned_trump = tweets_cleaned_trump.str.replace(c,'')

    print(tweets_cleaned_trump.head(50))


if __name__ == '__main__':
    main()
