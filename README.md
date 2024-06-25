# NLP Project

All data, results, models for the tasks can be found in the respective folders. testfiles includes all unused scripts in the final version.
The whole code can be run with runalloneclick.py. The model scripts themselves include the parameters "generate" and only in the main scripts "evaluate".
On default, only "evaluate" is set to True. If you want to run each model, including preprocessing and training, make sure to have all necessary dependencies installed (Check out requirements.txt for further details). Attention: Especially Stage3 models take a long time to run. 

* Stage 1: 

  task1.py: main file with generation False and evaluation True

* Stage 2: 

  task2.py: main file with generation False and evaluation True
  
  task2lstm.py: secondary model with generation False
  
  task2markov.py: secondary model with generation False

* Stage 3:
  
  task3llama2: main file with generation False and evaluation True

  task3gpt2: secondary model with generation False




Dropzone:
https://cloud.tugraz.at/index.php/s/2NfASLnwerAYWFW