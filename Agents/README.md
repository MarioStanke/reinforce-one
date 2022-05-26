## Agent Training

Instructions for use of the Agents are included in each file.  
Outputs are generated as follows:  
- "eval" folder contains evaluation metrics (read in tensorboard)
- "train" folder contains training metrics (read in tensorboard)
- "policy" folder contains saved models.  

Policies are only saved if they achieve a higher average return than -30.
Then, only models that perform better during evaluation than previously saved ones are stored.  
The threshhold for saving is reset to -30 after 5000 iterations.
