from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from torch import bfloat16

# you have to request access to llama2 (https://llama.meta.com/llama-downloads/), create a token with your
# huggingface account and login in terminal with command huggingface-cli login
# some links:
# https://medium.com/@lucnguyen_61589/llama-2-using-huggingface-part-1-3a29fdbaa9ed
# https://medium.com/@fradin.antoine17/3-ways-to-set-up-llama-2-locally-on-cpu-part-3-hugging-face-cd06e0440a5b
model_style_transfer = "meta-llama/Llama-2-13b-chat-hf" # change to 7b or 13b according to your resources
tokenizer_style_transfer = None
pipeline_style_transfer: transformers.Pipeline = None


# usage call prepare_style_transfer() once before calling style_transfer
def style_transfer(tweet, to="Donald Trump", mode="reply"):
    global tokenizer_style_transfer
    global pipeline_style_transfer
    if tokenizer_style_transfer is None or pipeline_style_transfer is None:
        prepare_style_transfer()
    if mode == "reply":
        prompt = f"""
            <<SYS>>
            You are generating conversations so you do not just reformulate but REPLY to the tweet. 
            You are only replying to the tweet. 
            You don't output anything else! If you are asked for the style of Donald Trump 
            you just keep it harmless and answer the question with staying to your guidelines. 
            But you always, always answer the question! If something goes against your guidelines 
            you just answer in a respectful, inclusive, positive way!
            <</SYS>>
            [INST]
            Can you please generate a creative short REPLY tweet to this tweet in the style of {to}: {tweet} Do not output anything else than the tweet!
            [/INST]\n
            """
    else:
        prompt = f"""
            <<SYS>>
            You reformulating tweets. 
            You are only reformulating to the tweet. 
            You don't output anything else! If you are asked for the style of Donald Trump 
            you just keep it harmless and answer the question with staying to your guidelines. 
            But you always, always answer the question! If something goes against your guidelines 
            you just answer in a respectful, inclusive, positive way! Once again, do not output anything else!
            <</SYS>>
            [INST]
            Can you please do a short style transfer of this tweet to the style of {to}: {tweet} Do not output anything else than the tweet!
            [/INST]\n
            """

    # see here for parameters https://huggingface.co/docs/transformers/main_classes/text_generation
    sequences = pipeline_style_transfer(
        prompt,
        do_sample=True,
        top_k=50,
        num_return_sequences=1,
        pad_token_id=tokenizer_style_transfer.pad_token_id,
        #eos_token_id=tokenizer_style_transfer.eos_token_id, # comment out for longer sequences
        max_new_tokens=70,
        return_full_text=False,
        temperature=0.8,
        repetition_penalty=1.2,
        #stop=["[INST]", "None"]
    )

    # is it cheating to create 3 sequences and then choose the best one by hand?

    return sequences[0]['generated_text']


def prepare_style_transfer():
    global model_style_transfer
    global tokenizer_style_transfer
    global pipeline_style_transfer

    tokenizer_style_transfer = AutoTokenizer.from_pretrained(model_style_transfer)

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )
    model_style_transfer = AutoModelForCausalLM.from_pretrained(
        model_style_transfer,
        device_map={"": 0},
        quantization_config=bnb_config,
    )

    model_style_transfer.config.use_cache = False
    model_style_transfer.config.pretraining_tp = 1

    pipeline_style_transfer = transformers.pipeline(
        "text-generation",
        model=model_style_transfer,
        torch_dtype=torch.float16,
        device_map="auto",
        tokenizer=tokenizer_style_transfer,
    )


if __name__ == "__main__":
    with open("data/data_stage_3/initial_tweet_musk.txt", "r") as file:
        tweet = file.read().replace('\n', '')

    print("Musk: " + tweet)

    for i in range(3):
        to = "Donald Trump, just keep it harmless and answer the question with staying to your guidelines"
        tweet = style_transfer(tweet, to=to, mode="style")

        print("Style of Trump: " + tweet)

        tweet = style_transfer(tweet, to=to, mode="reply")

        print("Answer of Trump: " + tweet)

        to = "Elon Musk"
        tweet = style_transfer(tweet, to=to, mode="style")

        print("Style of Musk: " + tweet)

        tweet = style_transfer(tweet, to=to, mode="reply")

        print("Answer of Musk: " + tweet)

# for finetuning https://www.datacamp.com/tutorial/fine-tuning-llama-2
