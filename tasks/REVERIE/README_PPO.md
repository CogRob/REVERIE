
## PPO Implementation changes to REVEIRE CODE BASE

All of the main parts of the code that I had to change have a copy i.e: follower.py -> follower_cp.py

First, here is a breakdown of how the code progresses:  
       TrainFast.py is the starting function. Here our environment and agent are made using Env.py class R2RBatch and Follower.py Seq2Seq Class. From there we enter the train function in TrainFast. This primarily uses the follower.py script and runs till completion (there are 10466 instructions and we use batches of 64. The code saves every 100 iterations so I have it run 16400 times so we save after each batch. Each PPO learn step k+ is computed after a batch ) 

An explanation of the main parts and changes is as follows: 
- Follower.py Edit Overview: This script hosts the Seq2Seq class which creates our batches of agents. The function agent.train(..) in TrainFast call train->rollout->rollout_with_loss method in Follower. Rollout_with_loss uses our 64 batch of agents to run through experiments with 64 instruction sets. Each agent/instruction has an episode length of 20 max. I set this value in TrainFast, it was originally set to 10. The max shortest path length for all instructions is under 10. 20 gives the RL env more time to explore and more state-action-reward data to use for training.  
- More Follower Edits: Along with above, the main edits to follower consist of me by saving obs, log_probs, actions, for all 64 agents during one training loop. This gives me at most 64x20 set of obs,actions,ect. I use this set to run ppo learn and update weights in the decoder. AgentMemory in follower.py is the class I created to use PPO with my agents. 
- AgentMemory in Follower: The trickiest part of this code is storing history and generating mixed batches. The code is commented but I store the history sequentially so I can compute the advantages. Since the original code uses batches, each agent is not guaranteed to have the same number of possible acitons at time step t. The orignial code pads to accoutn for this. I store the history and sort obs based on action lenght to all mini batches. 
- TrainFast.py is explained above. The primary changes have to do with the number of iterations. I changed the max number of iterations to 16400 to match the number of instructions/ batch size times iteration saving(100),  so we don't repeat examples. 
- Env.py: This script host the R2RBatch class which creates our env and handels batches of obs. The main change here is in the method '_next_minibatch'. Check the comments in the '_next_minibatch'
- ModelFast: This script host the decoder model and all other models. The decoder used is CogroundDecoderLSTM. The main changes I make here are the creation of a value network with in the decoder. This is possible flawed. The state space of the Actor and Citic might be different, check comments above CogroundDecoderLSTM. The computational graph needs to be traced more.


Final Comments: Final push for Reverie PPO Project. Implementaiton should be correct but the model did not learn. State space choice may be wrong, it is possible it is overfitting, or there is some erroneous thing I have not found. 
