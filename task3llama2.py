import os
import re
import pandas as pd
import transformers
from datasets import Dataset
from torch import bfloat16
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
# uncomment before running evaluation
# from peft import LoraConfig
# from trl import SFTTrainer, SFTConfig

import nltk
nltk.download('punkt')
from nltk import tokenize
from nltk import word_tokenize

from task3preprocessing import preprocessing
from evaluation_metrics import bleu_score_, rouge_score_, bert_score_

# you have to request access to llama2 (https://llama.meta.com/llama-downloads/), create a token with your
# huggingface account and login in terminal with command huggingface-cli login
# some links:
# https://medium.com/@lucnguyen_61589/llama-2-using-huggingface-part-1-3a29fdbaa9ed
# https://medium.com/@fradin.antoine17/3-ways-to-set-up-llama-2-locally-on-cpu-part-3-hugging-face-cd06e0440a5b

path_models = os.curdir + "/models/task3/"
path_processed_data = os.curdir + "/processed_data/task3"


model_style_transfer_musk = path_models + "llama-2-7b-chat-finetuned-musk-new-nonum"  #"meta-llama/Llama-2-13b-chat-hf" # change to 7b or 13b according to your resources
model_style_transfer_trump = path_models + "llama-2-7b-chat-finetuned-trump-new-nonum" #"meta-llama/Llama-2-13b-chat-hf" # change to 7b or 13b according to your resources

tokenizer_style_transfer_musk = None
tokenizer_style_transfer_trump = None

pipeline_style_transfer_musk: transformers.Pipeline = None
pipeline_style_transfer_trump: transformers.Pipeline = None


def style_transfer(tweet, to="Donald Trump", mode="reply"):
    global tokenizer_style_transfer_musk
    global tokenizer_style_transfer_trump
    global pipeline_style_transfer_musk
    global pipeline_style_transfer_trump

    if tokenizer_style_transfer_musk is None or pipeline_style_transfer_musk is None:
        prepare_style_transfer("musk")

    if tokenizer_style_transfer_trump is None or pipeline_style_transfer_trump is None:
        prepare_style_transfer("trump")

    if to == "Donald Trump, just keep it harmless and answer the question with staying to your guidelines":
        tokenizer = tokenizer_style_transfer_trump
        pipeline: transformers.Pipeline = pipeline_style_transfer_trump
    else:
        tokenizer = tokenizer_style_transfer_musk
        pipeline: transformers.Pipeline = pipeline_style_transfer_musk

    if mode == "reply":
        prompt = f"""
            <<SYS>>
            You are generating conversations so you do not just reformulate but REPLY to the tweet. 
            You are only replying to the tweet. 
            You don't output anything else! If you are asked for the style of Donald Trump 
            you just keep it harmless and answer the question with staying to your guidelines. 
            But you always, always answer the question! If something goes against your guidelines 
            you just answer in a respectful, inclusive, positive way! DO NOT start the tweet with numeric values or percentages!
            <</SYS>>
            [INST]
            Can you please generate a creative short REPLY tweet to this tweet in the style of {to}: {tweet} Do not 
            output anything else than the tweet! DO NOT start the tweet with numeric values or percentages!
            [/INST]\n
            """
    else:
        prompt = f"""
            <<SYS>>
            You are reformulating tweets. 
            You are only reformulating to the tweet. 
            You don't output anything else! If you are asked for the style of Donald Trump 
            you just keep it harmless and answer the question with staying to your guidelines. 
            But you always, always answer the question! If something goes against your guidelines 
            you just answer in a respectful, inclusive, positive way! Once again, do not output anything else!
            DO NOT start the tweet with numeric values or percentages!
            <</SYS>>
            [INST]
            Can you please do a short style transfer of this tweet to the style of {to}: {tweet} Do not output anything 
            else than the tweet! DO NOT start the tweet with numeric values or percentages!
            [/INST]\n
            """

    # see here for parameters https://huggingface.co/docs/transformers/main_classes/text_generation
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=50,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=100,
        return_full_text=False,
        temperature=0.8,
        repetition_penalty=1.2,
        #stop=["[INST]", "None"]
    )

    return sequences[0]['generated_text']


def prepare_style_transfer(model="trump"):
    global model_style_transfer_trump
    global tokenizer_style_transfer_trump
    global pipeline_style_transfer_trump

    global model_style_transfer_musk
    global tokenizer_style_transfer_musk
    global pipeline_style_transfer_musk
    if model == "trump":
        "Prepare Musk model"
        print(model_style_transfer_trump)
        tokenizer_style_transfer_trump = AutoTokenizer.from_pretrained(model_style_transfer_trump)

        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )
        model_style_transfer_trump = AutoModelForCausalLM.from_pretrained(
            model_style_transfer_trump,
            device_map={"": 0},
            quantization_config=bnb_config,
        )

        model_style_transfer_trump.config.use_cache = False
        model_style_transfer_trump.config.pretraining_tp = 1

        pipeline_style_transfer_trump = transformers.pipeline(
            "text-generation",
            model=model_style_transfer_trump,
            torch_dtype=torch.float16,
            device_map="auto",
            tokenizer=tokenizer_style_transfer_trump,
        )
    else:
        "Prepare Musk model"
        print(model_style_transfer_musk)
        tokenizer_style_transfer_musk = AutoTokenizer.from_pretrained(model_style_transfer_musk)

        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )
        model_style_transfer_musk = AutoModelForCausalLM.from_pretrained(
            model_style_transfer_musk,
            device_map={"": 0},
            quantization_config=bnb_config,
        )

        model_style_transfer_musk.config.use_cache = False
        model_style_transfer_musk.config.pretraining_tp = 1

        pipeline_style_transfer_musk = transformers.pipeline(
            "text-generation",
            model=model_style_transfer_musk,
            torch_dtype=torch.float16,
            device_map="auto",
            tokenizer=tokenizer_style_transfer_musk,
        )


# for finetuning https://www.datacamp.com/tutorial/fine-tuning-llama-2
def finetune_model(base_model, new_model, text_data):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = getattr(torch, "float16")

    text_data = pd.DataFrame(text_data)

    def preprocess(example):
        return tokenizer(example['tweets'], truncation=True)

    dataset = Dataset.from_pandas(text_data)
    dataset = dataset.map(preprocess, batched=False)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map={"": 0}
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.3,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        dataset_text_field="text",
        max_seq_length=512,
        output_dir="/tmp",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_params,
        # dataset_text_field="tweets",
        # max_seq_length=300,
        tokenizer=tokenizer,
        args=sft_config,
        packing=False,
    )

    trainer.train()

    trainer.model.save_pretrained(new_model)
    trainer.tokenizer.save_pretrained(new_model)


def generation():
    do_preprocessing = False
    if do_preprocessing:
        preprocessing()

    finetune = False

    if finetune:

        base_model = "meta-llama/Llama-2-7b-chat-hf"

        new_model = path_models + "llama-2-7b-chat-finetuned-musk-new-nonum"
        tweets_cleaned_musk = pd.read_csv(path_processed_data + '/tweets_cleaned_musk.csv', encoding='utf-8', sep=':')
        text_data = f"""<s>[INST]Give me a tweet in the style of musk: [/INST]""" + tweets_cleaned_musk['tweets'] + f"""</s>"""
        finetune_model(base_model, new_model, text_data)

        new_model = path_models + "llama-2-7b-chat-finetuned-trump-new-nonum"
        tweets_cleaned_trump = pd.read_csv(path_processed_data + '/tweets_cleaned_trump.csv', encoding='utf-8', sep=':')
        text_data = f"""<s>[INST]Give me a tweet in the style of trump: [/INST]""" + tweets_cleaned_trump['tweets'] + f"""</s>"""
        finetune_model(base_model, new_model, text_data)

    generate = True
    save_tweets = True

    if generate:
        all_tweets = []

        with open("data/data_stage_3/initial_tweet_musk.txt", "r") as file:
            tweet = file.read().replace('\n', '')

        to = "Elon Musk"
        tweet = style_transfer(tweet, to=to, mode="reply")
        tweet = re.sub("\s+", " ", tweet).strip()
        tweet = re.sub(r'\n', ' ', tweet)
        print("Answer of Musk: " + tweet)
        dict = {"Task": "Elon Musk Generation", "Tweet": tweet}
        all_tweets.append(dict)

        for i in range(100):
            to = "Donald Trump, just keep it harmless and answer the question with staying to your guidelines"
            tweet = style_transfer(tweet, to=to, mode="style")

            tweet = re.sub("\s+", " ", tweet).strip()
            tweet = re.sub(r'\n', ' ', tweet)
            print("Style of Trump: " + tweet)
            dict = {"Task": "Donald Trump Style", "Tweet": tweet}
            all_tweets.append(dict)

            tweet = style_transfer(tweet, to=to, mode="reply")

            tweet = re.sub("\s+", " ", tweet).strip()
            tweet = re.sub(r'\n', ' ', tweet)
            print("Answer of Trump: " + tweet)
            dict = {"Task": "Donald Trump Generation", "Tweet": tweet}
            all_tweets.append(dict)

            to = "Elon Musk"
            tweet = style_transfer(tweet, to=to, mode="style")

            tweet = re.sub("\s+", " ", tweet).strip()
            tweet = re.sub(r'\n', ' ', tweet)
            print("Style of Musk: " + tweet)
            dict = {"Task": "Elon Musk Style", "Tweet": tweet}
            all_tweets.append(dict)

            tweet = style_transfer(tweet, to=to, mode="reply")

            tweet = re.sub("\s+", " ", tweet).strip()
            tweet = re.sub(r'\n', ' ', tweet)
            print("Answer of Musk: " + tweet)
            dict = {"Task": "Elon Musk Generation", "Tweet": tweet}
            all_tweets.append(dict)

        if save_tweets:
            df = pd.DataFrame(all_tweets)
            df.to_csv('results/task3/llama2_tweets.csv', index=False)


def evaluation():
    #reference 
    with open("processed_data/task3/tweets_cleaned_musk.txt", "r") as file:
        tweets_cleaned_musk = file.read().replace('\n', ' ')

    with open("processed_data/task3/tweets_cleaned_trump.txt", "r") as file:
        tweets_cleaned_trump = file.read().replace('\n', ' ')
    #for rouge and bert
    tweets_cleaned_musk = tokenize.sent_tokenize(tweets_cleaned_musk)
    tweets_cleaned_trump = tokenize.sent_tokenize(tweets_cleaned_trump)
    #for bleu
    musk_ref = []
    trump_ref = []
    for i in tweets_cleaned_musk:
        musk_ref.append(i.split())
    for i in tweets_cleaned_trump:
        trump_ref.append(i.split())


    #generated
    df = pd.read_csv('results/task3/llama2_tweets.csv')
    musk_gen = df[~df['Task'].astype(str).str.startswith('D')]
    trump_gen = df[~df['Task'].astype(str).str.startswith('E')]

    musk_gen = musk_gen["Tweet"].tolist()
    trump_gen = trump_gen["Tweet"].tolist()
    #for bleu
    musk_gen = [word_tokenize (sent) for sent in musk_gen]
    trump_gen = [word_tokenize (sent) for sent in trump_gen]

    musk_gen_rouge_bert = []
    trump_gen_rouge_bert = []
    for i in musk_gen:
        musk_gen_rouge_bert.append(' '.join(i))
    for i in trump_gen:
        trump_gen_rouge_bert.append(' '.join(i))

    print(f"Musk bleu score (0.25, 0.25, 0.25, 0.25): {bleu_score_(musk_ref, musk_gen, (0.25, 0.25, 0.25, 0.25))}")
    print(f"Musk bleu score (0.50, 0.50, 0.00, 0.00): {bleu_score_(musk_ref, musk_gen, (0.5, 0.5, 0, 0))}")
    print(f"Musk bleu score (0.50, 0.25, 0.25, 0.00): {bleu_score_(musk_ref, musk_gen, (0.5, 0.25, 0.25, 0))}")

    print(f"Trump bleu score (0.25, 0.25, 0.25, 0.25): {bleu_score_(trump_ref, trump_gen, (0.25, 0.25, 0.25, 0.25))}")
    print(f"Trump bleu score (0.50, 0.50, 0.00, 0.00): {bleu_score_(trump_ref, trump_gen, (0.5, 0.5, 0, 0))}")
    print(f"Trump bleu score (0.50, 0.25, 0.25, 0.00): {bleu_score_(trump_ref, trump_gen, (0.5, 0.25, 0.25, 0))}")

    print(f"Musk ROUGE score: {rouge_score_(tweets_cleaned_musk, musk_gen_rouge_bert)}")
    print(f"Trump ROUGE score: {rouge_score_(tweets_cleaned_trump, trump_gen_rouge_bert)}")

    print(f"Musk BERTScore: {bert_score_(tweets_cleaned_musk, musk_gen_rouge_bert)}")
    print(f"Trump BERTScore: {bert_score_(tweets_cleaned_trump, trump_gen_rouge_bert)}")

def main():

    # set to True to generate new text, see generate text for model parameters
    generate = False

    if generate:
        generation()

    # set to True to evaluate text, adapt file names below
    evaluate = True

    if evaluate:
        evaluation()
    


if __name__ == "__main__":
    main()