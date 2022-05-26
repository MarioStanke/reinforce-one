# REINFORCE One
Reinforcement Learning for One Health  
Code accompanying master thesis: "Optimizing policies for epidemic control using reinforcement learning".

# Documentation 
  - [Ideas for Phase 1 Environment (from call on 11.12.20).](docs/2020-12-11-Note.pdf)
  - "Environment" folder contains all environment variants and environment tests.
  - "Agent" folder contains RNN/ANN DDPG notebook "DDPG.ipynb" and Agent_PPO.py for ANN/RNN PPO.
  - "Policy_Test" folder contains a notebook that tests learned policies.
  - Usage guides can be found in each folder.
  
# Dependencies  
To use code, please run the following installations.  
  
pip install tensorflow==2.6.0  
pip install tf-agents==0.9.0  
pip install keras==2.6.0  
pip install tensorflow-probability==0.14.1

# Citation

Uses agents from [TF-Agents](https://github.com/tensorflow/agents).  
Authors: Sergio Guadarrama, Anoop Korattikara, Oscar Ramirez, Pablo Castro,  
Ethan Holly, Sam Fishman, Ke Wang, Ekaterina Gonina, Neal Wu, Efi Kokiopoulou,  
Luciano Sbaiz, Jamie Smith, Gábor Bartók, Jesse Berent, Chris Harris, Vincent Vanhoucke, Eugene Brevdo. 
Title: TF-Agents: A library for Reinforcement Learning in TensorFlow, 2018.
