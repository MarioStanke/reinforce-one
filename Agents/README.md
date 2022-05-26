## Agent Training

Instructions for use of the Agents are included in each file.  
Agent_PPO.py will not work without first following the guide in the script.  
Outputs are generated as follows:  
- "eval" folder contains evaluation metrics (read in tensorboard).
- "train" folder contains training metrics (read in tensorboard).
- "policy" folder contains saved models (test in Policy_Test.ipynb).  

Policies are only saved if they achieve a higher average return than -30.  
Then, only models that perform better during evaluation than previously saved ones are stored.  
The threshhold for saving is reset to -30 after 5000 iterations.
