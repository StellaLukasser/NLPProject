# NLP Project

All data, results, models for the tasks can be found in the respective folders. 
The directory testfiles includes all unused scripts in the final version.
 
The **model scripts** themselves **include** the **boolean parameters** "generate" and only in the main scripts "evaluate" or
"evaluate_our_models".
On default, only evaluation is set to True. 

If you want to run each model, including preprocessing and training, 
make sure to have all necessary dependencies installed 
(Check out requirements.txt for further details). 

Attention: Especially Stage3 models take a long time to run. 
LLama2 also needs access permission from meta see links in file.

## Stage 1: 

 * task1.py: main file for generation and evaluation

## Stage 2: 

 * task2.py: main file for generation with RNN and evaluation for all models
  
 * task2lstm.py: secondary model for generation with LSTM
  
 * task2markov.py: secondary model for generation with Markov Chain

## Stage 3:
  
 * task3llama2: main file for generation with LLama2 finetuned and evaluation

 * task3gpt2: secondary model for generation with GPT2 finetuned

## General files

 * evaluation_metrics.py: includes functions for all evaluation metrics used in the project
 * The whole code can be run with runalloneclick.py.

 * In the results folder the final generations and style transfer results can be found in the "test_system" folders.

Dropzone:
https://cloud.tugraz.at/index.php/s/2NfASLnwerAYWFW