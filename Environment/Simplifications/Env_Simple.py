# Second Phase Environment implemented based on Env_P1 but for n herds in a population
'''
Copy of Env (02.12.21), except observation is percentage of infectious for each herd and global time.
Thus tests don't play a role.
'''

import numpy as np
import os
import random
from scipy.stats import poisson
from scipy.stats import geom
from scipy.stats import hypergeom
from scipy.stats import bernoulli

from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType, TimeStep, termination, transition


class Env_S(py_environment.PyEnvironment):
    def __init__(self,
                num_herds = 2,
                total_population = 300,
                rand_recovery_prob = 0.01,
                rand_infection_prob = 0.02,
                fix_episode_length = True,
                average_episode_length = 270
                ):
        super(Env_S, self).__init__()
        # State: [population_herd_1, ... , population_herd_n, infectious_herd_1, ... , infectious_herd_n]
        self._state = np.zeros(((num_herds*2),), np.int32)
        # Observation: [%infectious_herd_1, ... , %infectious_herd_n]
        self._observation = np.zeros(((num_herds),), np.int32)
        # Discount & time
        self._discount = np.float32(1)
        self._time = 0
        # Episode length
        self._fix_episode_length = fix_episode_length
        self._episode_length = 0
        self._average_episode_length = np.int32(average_episode_length)
        # Reward function values
        self._reward = np.float32(0)
        self._cost_removed = 1.
        self._cost_infectious = 2. 
        # Model defining metrics
        self._num_herds = num_herds
        self._total_population = total_population
        self._rand_recovery_prob = rand_recovery_prob    #g from scrapsheet
        self._rand_infection_prob = rand_infection_prob    #q from scrapsheet
    
    def action_spec(self):
        # Actions: [num_tests_herd_1, ... , num_tests_herd_n, slaughter_herd_1, ... slaughter_herd_n]
        return BoundedArraySpec(((self._num_herds),), np.float32, minimum=np.float32(0), maximum=np.float32(1))
    
    
    def observation_spec(self):
        # Observation: [time_since_test_herd_1, negative_tests_herd_1, positive_tests_herd_1, ... , positive_tests_herd_n]
        return BoundedArraySpec(((self._num_herds),), np.float32, minimum=np.float32(0), maximum=np.float32(1))
    
    def _check_values(self):
        assert self._num_herds >= 2, "Please set num_herds to at least 2."
        assert self._total_population >= 10, "Please set total_population to at least 10."
        #assert self._exchanged_members <= (self._total_population/self._num_herds), "More subjects transferred than available."
        for i in range (0, np.size(self._observation)):
            assert 0 <= self._observation[i] <= 1, "Check observation values"
        for j in range (0, self._num_herds-1):
            assert self._state[j] == self._state[j+1], "Total herd population changes while transfers should be symmetrical."
        return True
        
    def _reset(self):
        '''
        Resets all variables the model could change over time.
        State will be reset to no infected except for random number of initial infected in herd 1 
        (between 1 and (1/8)*population_herd_1).
        Observation will be reset to zeros.
        Returns TimeStep object with StepType.First.
        '''
        self._state = np.zeros(((self._num_herds*2),), np.int32)
        for i in range (0, self._num_herds):
            self._state[i] = self._total_population / self._num_herds
        initial_infected_h1 = np.random.randint(low = 1, high = (self._state[0]/8))
        self._time = 0
        self._reward = np.float32(0)
        if self._fix_episode_length: 
            self._episode_length = self._average_episode_length
        else:
            self._episode_length = geom.rvs(p = 1/self._average_episode_length)
        self._state[self._num_herds] = initial_infected_h1    #infected h1
        self._observation = np.zeros(((self._num_herds),), np.float32)
        self._check_values()
        return TimeStep(StepType.FIRST, reward=self._reward,
                    discount=self._discount, observation = self._observation)
        
    def _model(self, action: np.ndarray):
        '''
        Completes one time step in a herd (i.e. excluding transfers and tests).
        In f(x), samples new infections from poisson dist with lambda = 0.05,
        also considers spontaneous infection and recovery factors.
        Then, depending on whether a herd is to be replaced by healthy subjects (action),
        calls f(x) or simply replaces all subjects by healthy subjects for each herd.
        '''
        #Model for one herd
        def f(S, I): # Susceptible, Infected
            new_infs = 0
            inf_rvs = poisson.rvs(0.05, size = I)
            rec_rvs = poisson.rvs(self._rand_recovery_prob, size = I)
            rand_rvs = poisson.rvs(self._rand_infection_prob, size = S)
            potential_infs = np.sum(inf_rvs)
            recoveries = np.sum(rec_rvs)
            rand_infs = np.sum(rand_rvs)
            potential_infs = min(potential_infs, S)
            # code draw whether subject to be infected is already infected or sus
            if potential_infs >= 1:
                new_infs = hypergeom.rvs(M = (S + I), n = S, N = potential_infs, size = None)
            new_I = I + new_infs + rand_infs - recoveries
            new_I = max(0, new_I)
            return new_I
        
        #One step for each herd
        step_results = self._state
        for i in range (self._num_herds, self._num_herds*2):
            if action[i-self._num_herds] == 1:
                step_results[i] = 0
            else:
                step_results[i] = f(S = (self._state[i-self._num_herds] - self._state[i]), I = self._state[i])
        return step_results
    
    def _reward_func(self, action: np.ndarray):
        '''
        Calculates and returns reward. 
        Negative Reward for infectious grows (or rather decreases) exponentially, 
        others have a linear decrease.
        
        Reward is scaled against episode length before output, 
        so it is comparable across different episode lengths.
        '''
        step_reward = 0.
        pop_herd = self._total_population / self._num_herds
        for i in range (0, self._num_herds):
            step_reward -= self._discount * (action[i] * self._state[i] * self._cost_removed ) 
            step_reward -= self._discount * (self._state[i+self._num_herds] * self._cost_infectious) 
        step_reward = step_reward / pop_herd
        return step_reward
    
    def _step(self, action: np.ndarray):
        '''
        Step completes one time step in the environment.
        First, transfers subjects if time interval dictates it.
        Then, calls model(action) to complete a time step in each herd.
        Afterwards, tests subjects if action dictates it and outputs testresults
        if time for testing has been concluded.
        Finally, calculates reward and returns a Time_Step object.
        TimeStep(StepType.MID, reward=step_reward, discount=self._discount, observation=self._observation)
        '''
        if self._current_time_step.is_last():
            return self.reset()
        
        # Converting continuous inputs into discrete actions while maintaining gradient
        for a in range (0, self._num_herds):
            action[a] = np.int32(bernoulli.rvs(action[a], size = None))
            assert 0 <= action[a] <= 1, 'Action exceeds reasonable bounds.'
             
        self._time += 1
            
        # Model should make a step in between transfer and test
        self._time += 1
        self._state = self._model(action)
                
        # Testing
        for i in range (0, self._num_herds):
            self._observation[i] = self._state[i+self._num_herds] / self._state[i]
            
         
        
        # Reward function
        self._reward += np.float32(self._reward_func(action))
        #step_reward = np.float32(0)
        #scaled_reward = np.float32(self._reward/self._time)
            
        # Check for errors
        self._check_values()
        
        # Output
        if self._time == self._episode_length:
            return termination(self._observation, self._reward)
        else:
            return transition(self._observation, self._reward, np.float32(self._discount))
       