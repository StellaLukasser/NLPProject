import os
import re

path = os.curdir + "/data"
file1_kogler = path + "/data_stage_2/data_stage2_1_kogler.txt"
file2_kickl = path + "/data_stage_2/data_stage2_2_kickl.txt"

def read_file(file):
    f = open(file, "r", encoding="utf-8")
    text = f.read()
    f.close()
    return text

def pre_processing(text):
    #removing brackets i.e., (Abg. Belakowitsch: Wo genau?)
    # or (Beifall bei Grünen und ÖVP.)
    text=re.sub("\(.*?\)","",text)
    #lower
    text = text.lower()
    #removing numbers
    text = re.sub("\d+", " ", text)
    #remove - 
    text = re.sub(r"-", " ", text)
    text = re.sub(r"–", " ", text)
    text = re.sub(r":", " ", text)
    #spaces before punctuation
    text = re.sub('([.,!?()])', r" \1", text)
    #removing multiple spaces
    text = re.sub("\s+", " ", text).strip()
    #convert words and punctuations to indices
    text = re.findall(r"[\w']+|[.,!?;]", text)
    print(text)

def main():
    #read text
    text_kogler = read_file(file1_kogler)
    text_kickl = read_file(file2_kickl)
    #process data
    text_kogler = pre_processing(text_kogler)
    text_kickl = pre_processing(text_kickl)
    #TODO sequences

if __name__ == '__main__':
    main()