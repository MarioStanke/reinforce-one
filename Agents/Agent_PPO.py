#!/usr/bin/env python
# coding: utf-8

'''
PPO Agent for epidemic control model  
This notebook will train an agent in an epidemic control environment using PPO with or without RNNs.  
For use, please complete the following steps:

- Edit PATH variable below to any folder where training outputs can be stored.  
- Create a folder titled 'policy' in PATH directory.  
- Edit the sys path insertions (below) to the directories where environments are stored.  

The default variant is ANN PPO, for RNN PPO edit "use_rnns" variable below.  
Default environment is EE0, for different environments see "Environment" section below (ll. 77-81).  
For more in-depth changes in hyperparameters edit the "flags" in the hyperparameter section.
''' 

# Output folder
PATH = '/home/jovyan/Masterarbeit/out/'

# Path to environment folder
import sys
sys.path.insert(1, '/home/jovyan/Masterarbeit/reinforce-one/Environments')
sys.path.insert(1, '/home/jovyan/Masterarbeit/reinforce-one/Environments/Variations')

# Decide whether to use RNN DDPG or ANN DDPG
use_rnns = False

# Imports 
# Firstly, all relevant dependencies will be imported.  
# Comments indicate what imports are generally used for or related to.

# General Imports
import os
import tensorflow as tf 
import numpy as np
import functools
from absl import flags
from absl import app
from tf_agents.system import system_multiprocessing as multiprocessing
# Environment 
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import py_environment
from tf_agents.policies import scripted_py_policy
from tf_agents.policies import random_tf_policy
# Neural Networks 
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import value_network
from tf_agents.networks import value_rnn_network
from tf_agents.networks import sequential
from tf_agents.networks import nest_map
from tf_agents.keras_layers import inner_reshape
# Agent 
from tf_agents.agents.ppo import ppo_clip_agent
# Experience Replay
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
#Training
from tf_agents.utils import common
#Evaluation
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver
from tf_agents.trajectories import time_step

# Environment
# Next, an environment will be imported and initialized.
from EE0 import EE0
from EE0_5 import EE0_5
from EE0_NT import EE0_NT
from EE0_A import EE0_A
from EE1 import EE1
from EE1_A import EE1_A
from EE2 import EE2

# Please set all five parameters below to your liking. 
num_herds = 2
total_population = 300
average_episode_length = 200
fix_episode_length = True
env_fn = EE0

py_env = env_fn(num_herds = num_herds, total_population = total_population, fix_episode_length = fix_episode_length, 
                average_episode_length = average_episode_length)

# Transforms py environment into tensorflow environment (i/o are now tensors)
train_env = tf_py_environment.TFPyEnvironment(py_env)
eval_env = tf_py_environment.TFPyEnvironment(py_env)

# Training
# In this section, define a function for agent training and evaluation.  

# Hyperparameters  
# Set hyperparameters for PPO.

flags.DEFINE_string('directory', PATH,
                    'Root directory for saving policies.')
flags.DEFINE_integer('ppo_rb_capacity', 10000,
                     'Replay buffer capacity per env.')
flags.DEFINE_integer('ppo_num_parallel_environments', 50,  #30
                     'Number of environments to run in parallel')
flags.DEFINE_integer('ppo_num_iterations', 100000,
                     'Number of iterations to run before finishing.')
flags.DEFINE_integer('ppo_num_epochs', 100,  #25
                     'Number of epochs for computing policy updates.')
flags.DEFINE_integer(
    'ppo_collect_episodes_per_iteration', 50,   #30
    'The number of episodes to take in the environment before '
    'each update. This is the total across all parallel '
    'environments.')
flags.DEFINE_integer('ppo_num_eval_episodes', 200,
                     'The number of episodes to run eval on.')
flags.DEFINE_integer('ppo_eval_interval', 1000,  #500
                     'Eval interval.')
flags.DEFINE_boolean('use_rnns', use_rnns,
                     'If true, use RNN for policy and value function.')

FLAGS = flags.FLAGS

# PPO
# Define training function using tf-agent's ppo agent.
def PPO(num_iterations = 200000,
        directory = PATH,
        env_fn = env_fn,
        eval_interval = 500,
        eval_episodes = 200,
        summary_interval = 1000,
        threshhold_return = -30,
        threshhold_reset_interval = 5000,
        # RNN/ANN Hyperparameters
        use_rnns = False,
        lstm_size = (40,),
        actor_fc_layers=(400, 200),
        value_fc_layers=(400, 200),
        # Agent hyperparameters
        learning_rate = 1e-3,  #1e-3
        importance_ratio_clipping = 0.2,
        # Training hyperparameters
        num_epochs = 100,
        # Experience replay hyperparameters
        collect_episodes_per_iteration = 50,
        rb_capacity = 10000,
        batch_size = 50):
    
    
    # Create directories for summary output
    directory = os.path.expanduser(directory)
    train_dir = os.path.join(directory, 'train')
    eval_dir = os.path.join(directory, 'eval')
    policy_dir = os.path.join(directory, 'policy')
    
    # Global step tracks number of train steps
    global_step = tf.compat.v1.train.get_or_create_global_step()
    
    # Initialize summary writers 
    train_summary_writer = tf.compat.v2.summary.create_file_writer(
                               train_dir, flush_millis=10000)
    train_summary_writer.set_as_default()

    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
                              eval_dir, flush_millis=10000)
    
    with tf.compat.v2.summary.record_if(lambda: tf.math.equal(global_step % summary_interval, 0)):
        if batch_size > 1:
            ppo_env = tf_py_environment.TFPyEnvironment(
                          parallel_py_environment.ParallelPyEnvironment(
                              [env_fn] * batch_size
                          )
                      )
        else:
            ppo_env = train_env
    
        observation_spec = ppo_env.observation_spec()
        action_spec = ppo_env.action_spec()
    
        if use_rnns:
            actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
                            observation_spec,
                            action_spec,
                            input_fc_layer_params=actor_fc_layers,
                            output_fc_layer_params=None,
                            lstm_size=lstm_size)
            value_net = value_rnn_network.ValueRnnNetwork(
                            observation_spec,
                            input_fc_layer_params=value_fc_layers,
                            output_fc_layer_params=None,
                            lstm_size=lstm_size)
        else:
            actor_net = actor_distribution_network.ActorDistributionNetwork(
                            observation_spec,
                            action_spec,
                            fc_layer_params=actor_fc_layers,
                            activation_fn=tf.keras.activations.tanh)
            value_net = value_network.ValueNetwork(
                            observation_spec,
                            fc_layer_params=value_fc_layers,
                            activation_fn=tf.keras.activations.tanh)
        

        # PPO Agent
        agent = ppo_clip_agent.PPOClipAgent(ppo_env.time_step_spec(),
                                            action_spec,
                                            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
                                            actor_net = actor_net,
                                            value_net = value_net,
                                            entropy_regularization = 0.0,
                                            importance_ratio_clipping = importance_ratio_clipping,
                                            normalize_observations = False,
                                            normalize_rewards = False,
                                            use_gae = True,
                                            num_epochs = num_epochs,
                                            train_step_counter = global_step)
        agent.initialize()
        
        # Metrics to be tracked in the summary 
        environment_steps_metric = tf_metrics.EnvironmentSteps()
        step_metrics = [tf_metrics.NumberOfEpisodes(),
                        environment_steps_metric]
        train_metrics = step_metrics + [tf_metrics.AverageReturnMetric(batch_size=batch_size),
                                        tf_metrics.AverageEpisodeLengthMetric(batch_size=batch_size)]
    
        eval_metrics = [tf_metrics.AverageReturnMetric(buffer_size=eval_episodes), 
                        tf_metrics.AverageEpisodeLengthMetric(buffer_size=eval_episodes)]
    
        # Tools for evaluation
        eval_policy = agent.policy
        saver = policy_saver.PolicySaver(eval_policy)
        best_return = threshhold_return

        # Experience replay and sample collection tools
        collect_policy = agent.collect_policy
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(agent.collect_data_spec,
                                                                       batch_size = batch_size,
                                                                       max_length = rb_capacity)
    
        # Assign step driver to fill replay buffer 
        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(ppo_env,
                                                                     collect_policy,
                                                                     observers=[replay_buffer.add_batch] + train_metrics,
                                                                     num_episodes=collect_episodes_per_iteration)

        
        # TF functions speed up training process
        collect_driver.run = common.function(collect_driver.run)
        agent.train = common.function(agent.train)
    
        # Training starts
        train_loss = 0
    
        def train_step():
            trajectories = replay_buffer.gather_all()
            return agent.train(experience=trajectories)
        train_step = common.function(train_step)
    
        for _ in range(num_iterations):
            collect_driver.run()    

            train_loss, _ = train_step()
            replay_buffer.clear()
            
            for train_metric in train_metrics:
                train_metric.tf_summaries(train_step=global_step, step_metrics=train_metrics[:2])
                
            # Evaluation and policy storage
            if global_step.numpy() % eval_interval == 0:
                results = metric_utils.eager_compute(eval_metrics, 
                                                     eval_env,
                                                     eval_policy,
                                                     num_episodes=eval_episodes,
                                                     train_step=global_step,
                                                     summary_writer=eval_summary_writer,
                                                     summary_prefix='Metrics')
                metric_utils.log_metrics(eval_metrics)
                print('Global Step = {0}, Average Return = {1}.'.format(global_step.numpy(), results['AverageReturn'].numpy()))
                if results['AverageReturn'].numpy() > best_return:
                    best_return = results['AverageReturn'].numpy()
                    print('New best return: ', best_return)
                    dir_name = str(global_step.numpy()) + '_' + str(best_return) 
                    saver.save(os.path.join(policy_dir, dir_name))
            if global_step.numpy() % threshhold_reset_interval == 0:
                best_return = threshhold_return
            if global_step.numpy() % num_iterations == 0:
                break
    return train_loss


# # Run Functions (rename)  
# Now you can execute PPO using either artificial or recurrent NNs!

def main(_):
    tf.compat.v1.enable_v2_behavior()
    PPO(num_iterations = FLAGS.ppo_num_iterations,
           directory = FLAGS.directory,
           eval_interval = FLAGS.ppo_eval_interval,
           use_rnns = FLAGS.use_rnns,
           num_epochs = FLAGS.ppo_num_epochs,
           collect_episodes_per_iteration = FLAGS.ppo_collect_episodes_per_iteration,
           rb_capacity = FLAGS.ppo_rb_capacity,
           batch_size = FLAGS.ppo_num_parallel_environments)

# Addresses `UnrecognizedFlagError: Unknown command line flag 'f'`
sys.argv = sys.argv[:1]
# `app.run` calls `sys.exit`
try:
    app.run(lambda argv: None)
except:  
    pass

if __name__ == '__main__':
    multiprocessing.handle_main(functools.partial(app.run, main))




