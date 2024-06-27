import pandas as pd
import random
import os

#set seed to 77 ensure baseline reproducibility
random.seed(77)

def read_file(filename):

    text = ""
    with open(filename, "r", encoding="UTF8") as f:
        for line in f:
            if len(line) > 1:
                text += line
    return text

def save_file(text, path):

    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not isinstance(text, str):
        text = '\n'.join(text)

    with open(path, 'w') as file:
        file.write(text)


def task1_random_baseline_text():

    text = read_file("data/data_stage_1.txt")
    text = text.split()
    random_baseline_text = random.sample(text, 2000)
    full_string = " ".join(random_baseline_text)
    save_file(full_string, "random_baseline_texts/task1_random_baseline_text")


def task2_random_baseline_texts():

    text_kogler = read_file("data/data_stage_2/data_stage2_1_kogler.txt")
    text_kogler = text_kogler.split()

    text_kickl = read_file("data/data_stage_2/data_stage2_2_kickl.txt")
    text_kickl = text_kickl.split()

    random_baseline_text_kogler = random.sample(text_kogler, 250)
    random_baseline_text_kickl = random.sample(text_kickl, 250)

    random_baseline_text_kogler = " ".join(random_baseline_text_kogler)
    random_baseline_text_kickl = " ".join(random_baseline_text_kickl)

    save_file(random_baseline_text_kogler, "random_baseline_texts/task2_random_baseline_text_kogler")
    save_file(random_baseline_text_kickl, "random_baseline_texts/task2_random_baseline_text_kickl")


def main():

    task1_random_baseline_text()
    task2_random_baseline_texts()




if __name__ == '__main__':
    main()