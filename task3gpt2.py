import numpy as np
import os
import random
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer

path = os.curdir + "/data"
file1_musk = path + "/data_stage_3/data_stage3_1_musk.xlsx"
file2_trump = path + "/data_stage_3/data_stage3_2_trump.xlsx"
path_results = os.curdir + "/results"
path_models = os.curdir + "/models/task3"

# Load the XLSX file
musk = pd.read_excel(file1_musk)
trump = pd.read_excel(file2_trump)

# Extract tweets from the DataFrame
tweets_musk = musk.iloc[:, 0].tolist()
tweets_trump = trump.iloc[:, 0].tolist()

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Function to generate new tweets based on input tweet
def generate_tweet(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_new_tokens=10, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Sample an initial tweet randomly from Elon Musk's tweets
initial_tweet = random.choice(tweets_musk)

# Generate a new tweet based on the style of the initial tweet
new_tweet_musk = generate_tweet("do something")

# Style-transfer the new tweet to Trump's style
# For demonstration purposes, I'll use the same tweet (new_tweet_musk) as Trump's style
# You can replace it with a style transfer algorithm if available
style_transferred_tweet = new_tweet_musk

# Generate a new tweet based on the style-transferred tweet in Trump's style
new_tweet_trump = generate_tweet(style_transferred_tweet)

# Print the results
print("Initial Tweet (Elon Musk):", initial_tweet)
print("Generated Tweet (Musk's Style):", new_tweet_musk)
print("Style-Transferred Tweet (Trump's Style):", style_transferred_tweet)
print("Generated Tweet (Trump's Style):", new_tweet_trump)