#!/usr/bin/env python
# coding: utf-8

# ## Test the environment `HerdEnv` of `herd_env.py`

# In[1]:


import numpy as np
import tensorflow as tf
from tf_agents.environments import utils
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories import TimeStep
from tf_agents.policies import scripted_py_policy
from tf_agents.policies import random_py_policy
from tf_agents.metrics import py_metrics
from tf_agents.drivers import py_driver
from tf_agents.utils import common


# In[2]:


from herd_env import HerdEnv


# In[3]:


# sanity check
henv_val = HerdEnv(herd_sizes = [64,64], rand_recovery_prob = 0.1, rand_infection_prob = 0.05)
utils.validate_py_environment(henv_val, episodes=10)


# In[4]:


# create Herd Environment instance to be trained for
max_episode_length=100
num_herds=2
henv = HerdEnv(herd_sizes = [32,32], expected_episode_length=50., max_episode_length=max_episode_length,
               rand_recovery_prob = 0.04, rand_infection_prob = 0.05)


# In[5]:


# show interor values of environment
time_step = henv.reset()
print(time_step)
cumulative_reward = time_step.reward
finished = False

while not finished:
  time_step = henv.step(0) # do nothing
  s = henv.get_state()
  print("state: ", s, "observation: ", time_step.observation, "\treward: ", time_step.reward)
  cumulative_reward += time_step.reward
  if time_step.step_type == StepType.LAST:
    finished = True

print('Final Reward = ', cumulative_reward)


# In[6]:


action_spec = henv.action_spec()
ts_spec = henv.time_step_spec()
print("action spec:\n", action_spec, "\n\ntime step spec:\n", ts_spec)


# ### Define scripted policies

# In[7]:


# do nothing policy: cull never
action_script0 = [(max_episode_length, 0)]

# cull first herd every 10th step and second herd every 20th step
action_script1 = [(9, 0), 
                 (1, 1),
                 (9, 0), 
                 (1, 3)] * int(1+max_episode_length/20)

manual_scripted_policy0 = scripted_py_policy.ScriptedPyPolicy(
    time_step_spec=ts_spec,
    action_spec=action_spec,
    action_script=action_script0)

manual_scripted_policy1 = scripted_py_policy.ScriptedPyPolicy(
    time_step_spec=ts_spec,
    action_spec=action_spec,
    action_script=action_script1)

init_policy_state = manual_scripted_policy0.get_initial_state()


# In[8]:


policy_state =  init_policy_state
ts0 = henv.reset()
for _ in range(21):
    action_step = manual_scripted_policy1.action(ts0, policy_state)
    policy_state = action_step.state
    print("action=", action_step.action, "\tpolicy_state", policy_state)
policy_state = manual_scripted_policy1.get_initial_state()


# ### ... and a random policy

# In[9]:


random_policy = random_py_policy.RandomPyPolicy(time_step_spec=ts_spec, action_spec=action_spec)


# ## Drive a rollout

# In[10]:


def compute_avg_return(environment, policy, num_episodes=50, verbose=False):
  total_return = 0.0
  cullsteps = 0
  for e in range(num_episodes):

    time_step = environment.reset()
    if isinstance(policy, scripted_py_policy.ScriptedPyPolicy):
        policy_state = policy.get_initial_state() # remember where in the script we were
    else:
        policy_state = None # other policies without memory
    episode_return = 0.0
    i=0
    while not time_step.is_last():
        i+=1
        action_step = policy.action(time_step, policy_state)
        if action_step.action > 0:
            cullsteps += 1
        policy_state = action_step.state
        time_step = environment.step(action_step.action)
        if isinstance(environment, HerdEnv):
            state = environment.get_state()
        else:
            state = None # TF environment from wrapper does not have get_state()
        episode_return += time_step.reward
        if verbose:
            print (f"episode {e:>2} step{i:>3} action: ", action_step.action, "state=", state, "obs=", time_step.observation, "reward=", time_step.reward)
    total_return += episode_return

  avg_return = total_return / num_episodes
  cullsteps /= num_episodes
  return avg_return, cullsteps


# In[11]:


random_reward, cullsteps = compute_avg_return(henv, random_policy)
print (f"average return of random policy: {random_reward:.3f} avg steps with culls per episode: {cullsteps}")


# In[12]:


# show states for one rollout of second scripted policy
compute_avg_return(henv, manual_scripted_policy1, num_episodes=1, verbose=True)


# In[13]:


manual_reward0, cullsteps = compute_avg_return(henv, manual_scripted_policy0, num_episodes=200)
print (f"average return of do-nothing-policy: {manual_reward0:.3f} avg culls {cullsteps}")
manual_reward1, cullsteps = compute_avg_return(henv, manual_scripted_policy1, num_episodes=200)
print (f"average return of manual policy: {manual_reward1:.3f} avg culls {cullsteps}")


# ### Train a Deep-Q Agent

# In[14]:


from tf_agents.networks.sequential import Sequential
from tensorflow.keras.layers import Dense
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory


# In[15]:


num_iterations = 20000
replay_buffer_max_length = 10000
batch_size = 64
num_eval_episodes = 100
initial_collect_steps = 100
collect_steps_per_iteration = 128
log_interval = 100
eval_interval = 500


# In[16]:


# make actor network simple
num_actions = 2**num_herds # this does not scale, obviously
kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03)
q_net = Sequential([Dense(4, activation=None,
                          kernel_initializer = kernel_initializer)])


# In[17]:


train_step_counter = tf.Variable(0)

train_env = tf_py_environment.TFPyEnvironment(henv)
eval_env = tf_py_environment.TFPyEnvironment(henv)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    boltzmann_temperature = 0.005,
    epsilon_greedy = None,
    optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-3),
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()


# In[18]:


agent.policy.trainable_variables


# In[19]:


# manually initialize a reasonably good policy: kill both herds if the sum of observations is large
W = np.array([[0, 3 ,0, 2],[0, 0, 3, 2,]])
b = np.array([1, 0, 0, 0])
q_net.layers[0].set_weights([W,b])
agent.policy.trainable_variables


# In[20]:


replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)


# In[21]:


agent.collect_data_spec._fields


# In[22]:


def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)

collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)


# In[23]:


# Dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3)

iterator = iter(dataset)
dataset


# In[ ]:


# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

  # Collect a few steps using collect_policy and save to the replay buffer.
  collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss

  step = agent.train_step_counter.numpy()

  if step <= 1 or step % log_interval == 0:
    print('step = {0:4>}: loss = {1:.4f}'.format(step, train_loss), end="\t")

  if step <= 1 or (step < 100 and step % 20 == 0) or step % eval_interval == 0:
    avg_return, cullsteps = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step {0}: average return = {1:.1f} cullsteps = {2:.1f}'.format(step, avg_return.numpy().item(), cullsteps))
    returns.append(avg_return)


# In[ ]:


learned_reward, culleps = compute_avg_return(eval_env, agent.policy, num_episodes=200)
print ("reward of learned policy: ", learned_reward.numpy(), "culleps=", culleps)


# In[ ]:


init_ts = eval_env.reset()

def get_action(obs):
    """ execute the learned policy network 
       obs:  one float for each herd - the time since last culling
    """
    _ts = TimeStep(tf.constant([0.]),
                   tf.constant([0.]),
                   tf.constant([1]),
                   tf.constant([obs]))
    # a = agent.collect_policy.action(_ts) # just to see how much is explored versus exploited
    a = agent.policy.action(_ts)
    return a.action.numpy().item()


# In[ ]:


# what the learned policy does on a grid of observations (5 steps per row&col)
A = [[get_action([x,y])
 for y in np.arange(0.,1.,.05,np.float32)]
 for x in np.arange(0.,1.,.05,np.float32)]
A


# ### Play with parameters of manually designed q_network policy

# In[ ]:


W, b = agent.policy.trainable_variables
W = W.numpy()
b = b.numpy()
print ("weights\n", W, "\nbias", b)


# In[ ]:


def nn(obs):
    y = np.dot(obs, W)+b
    return y


# In[ ]:


nn([0.5,.2])


# In[ ]:




