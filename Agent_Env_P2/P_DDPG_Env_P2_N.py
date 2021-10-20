#!/usr/bin/env python
# coding: utf-8

# Second Phase Environment implemented based on Env_P1 but for more herds in a population

# In[1]:


import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.ioff() # for py work
import numpy as np
import os
import pandas as pd
import copy
import random
from scipy.stats import poisson
from scipy.stats import geom
from scipy.stats import hypergeom
from scipy.stats import bernoulli

from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep


class P_Env_P2_N(py_environment.PyEnvironment):
    def __init__(self,
                root_dir, #path for plotting
                global_step = 0, #ID for plotting
                num_herds = 10,
                total_population = 3000,
                split_even = True,
                population_range = None,
                num_transfers = 10,
                weeks_until_exchange = 3,
                rand_recovery_prob = 0.005,
                rand_infection_prob = 0.01
                ):
        super(P_Env_P2_N, self).__init__()
        self._state = np.zeros(((num_herds*2),), np.int32)
        self._observation = np.zeros(((num_herds*3),), np.int32)
        self._discount = np.float32(1)
        self._time = 0
        self._episode_length = 0
        self._tests = []
        self._reward = np.float32(0)
        self._c_tests = 0.5   #cost for each test
        self._c_prime_tests = 10    #organizational costs tests
        self._e_removed = 3   #individual replacement cost
        self._weeks_until_testresults = 3
        self._split_even = split_even
        self._population_range = population_range
        self._num_herds = num_herds
        self._num_transfers = num_transfers
        self._total_population = total_population
        self._exchanged_members = np.int32(np.round(total_population / (num_transfers*num_herds*5)))    #k from scrapsheet
        self._weeks_until_exchange = weeks_until_exchange    #T from scrapsheet
        self._rand_recovery_prob = rand_recovery_prob    #g from scrapsheet
        self._rand_infection_prob = rand_infection_prob    #q from scrapsheet
        
        self._a_and_s = []
        self._root_dir = root_dir
        self._global_step = global_step
    
    def action_spec(self):
        #Actions for: number of subjects to be tested h1, h2. number of subjects to be eliminated h1, h2
        max_array = np.ones(((self._num_herds*2),), np.int32)
        for i in range (0, self._num_herds):
            max_array[i] = self._state[i]
        return BoundedArraySpec(((self._num_herds*2),), np.float32, minimum=0, maximum=1)
    
    
    def observation_spec(self):
        # tau, x0, x1 for both herds
        max_array = np.ones(((self._num_herds),), np.int32)
        for i in range (0, self._num_herds):
            max_array[i] = self._state[i]
        obs_max = np.amax(max_array)
        return BoundedArraySpec(((self._num_herds*3),), np.int32, minimum=0, maximum=obs_max)
    
    
    def _reset(self):
        '''
        State consists of actual state of each herd (population and infected, state[1]),
        and observation the agent gets to see (state[0]).
        state[0] contains:
        number of steps since test has taken place,
        number of positive tests,
        number of negative tests
        for each herd.  
        '''
        self._a_and_s = []
        self._state = np.zeros(((self._num_herds*2),), np.int32)
        if self._split_even:
            for i in range (0, self._num_herds):
                self._state[i] = self._total_population / self._num_herds
        else:
            raise NameError('Work more Maurice.')
        
        initial_infected_h1 = np.random.randint(low = 1, high = (self._state[0]/8))
        self._tests = []
        self._time = 0
        self._reward = np.float32(0)
        self._episode_length = geom.rvs(p = 1/270)
        self._state[self._num_herds] = initial_infected_h1    #infected h1
        self._observation = np.zeros(((self._num_herds*3),), np.int32)
        return TimeStep(StepType.FIRST, reward=self._reward,
                    discount=self._discount, observation = self._observation)
    
    def _test(self, herd = -1, num_tests = 0):
        '''
        Randomly draws (without returning) num_tests subjects of a herd,
        then tests whether they are infected or not before returning testresults.
        '''
        if herd >= 0 and num_tests > 0:
            test_out = hypergeom.rvs(M = self._state[herd], n = self._state[herd+self._num_herds], N = int(num_tests), size = None)
            testresults = np.zeros(3, np.int32)
            testresults[1] = num_tests - test_out #negative tests
            testresults[2] = test_out #positive tests
            return testresults
        else:
            return np.zeros(3, np.int32)
        
    def _transfer(self, origin_herd = -1, target_herd = -1):
        ''' 
        Each self._weeks_until_exchange weeks, transfers subjects 
        from origin_herd to target_herd by randomly drawing (without return)
        self._exchanged_members subjects from all subjects of origin_herd.
        returns numbers of infected transfers and susceptible transfers.
        '''
        if origin_herd >= 0 and target_herd >=0 and self._time % self._weeks_until_exchange == 0:
            infected_transfers = hypergeom.rvs(M = self._state[origin_herd], 
                                                 n = self._state[origin_herd+self._num_herds], N = self._exchanged_members, size = None)
            susceptible_transfers = self._exchanged_members - infected_transfers
            return np.array([susceptible_transfers, infected_transfers])    
        else:
            return None
        
    def _model(self, action: np.ndarray):
        '''
        Completes one time step in a herd (i.e. excluding transfers and tests).
        In f(x), samples new infections from poisson dist with lambda = 0.05,
        also considers spontaneous infection and recovery factors.
        Then, depending on whether a herd is to be replaced by healthy subjects (action),
        calls f(x) or simply replaces all subjects by healthy subjects for each herd.
        '''
        #Model for one herd
        def f(x):
            S = np.int32(x[0])
            I = np.int32(x[1])
            new_infs = 0
            inf_rvs = poisson.rvs(0.05, size = I)
            rec_rvs = poisson.rvs(self._rand_recovery_prob, size = I)
            rand_rvs = poisson.rvs(self._rand_infection_prob, size = S)
            potential_infs = np.sum(inf_rvs)
            recoveries = np.sum(rec_rvs)
            rand_infs = np.sum(rand_rvs)
            potential_infs = min(potential_infs, S)
            #code draw whether subject to be infected is already infected or sus
            if potential_infs >= 1:
                new_infs = hypergeom.rvs(M = (S + I), n = S, N = potential_infs, size = None)
            new_I = I + new_infs + rand_infs - recoveries
            return new_I
        
        #One step for each herd
        step_results = self._state
        temp_x = np.zeros((2,), np.int32)
        for i in range (self._num_herds, self._num_herds*2):
            if action[i] == 1:
                step_results[i] = 0
            else:
                step_results[i] = f(np.array([(self._state[i-self._num_herds] - self._state[i]), self._state[i]]))
        return step_results
    
    def _reward_func(self, action: np.ndarray):
        '''
        Calculates and returns reward.
        '''
        for i in range (0, self._num_herds):
            self._reward -= self._discount * (action[i] * self._c_tests + min(action[i],1) * self._c_prime_tests) / (self._total_population/self._num_herds)
            self._reward -= self._discount * (action[i+self._num_herds] * self._state[i] * self._e_removed ) / (self._total_population/self._num_herds)
            self._reward -= self._discount * (self._state[i+self._num_herds])**1.5 / (self._total_population/self._num_herds)
        return self._reward
    
    def plot_actions_and_states(self):
        t = np.linspace(0, self._time, num=len(self._a_and_s))
        fig, (p1,p2,p3) = plt.subplots(1, 3, figsize=(20,10))
        fig.suptitle('Actions and Infectious over Time')
        p1.set_title('Average Tests over Time')
        p1.set_xlabel('Time')
        p1.set_ylabel('Average Number of Tests')
        p1.set_ylim(-0.5, (self._total_population/self._num_herds)+1)
        p2.set_title('Number of Herd Replacements over Time')
        p2.set_xlabel('Time')
        p2.set_ylabel('Replacements')
        p2.set_ylim(-0.5, self._num_herds+1)
        p3.set_title('Average Infectious over Time')
        p3.set_xlabel('Time')
        p3.set_ylabel('Average Infectious')
        p3.set_ylim(-0.5, (self._total_population/self._num_herds)+1)
        root_dir = self._root_dir
        root_dir = os.path.expanduser(root_dir) 
        fnm = os.path.join(root_dir, 'Act_Inf' + '_' + str(self._time) + '_' + str(self._global_step)) 
        n_tests, n_replace, av_infectious = [], [], []
        for i in range(len(self._a_and_s)):
            n_tests.append(self._a_and_s[i][0])
            n_replace.append(self._a_and_s[i][1])
            av_infectious.append(self._a_and_s[i][2])
        p1.plot(t, n_tests, color='black', label = 'Average Number of Tests', marker = '', linestyle = '-')
        p2.plot(t, n_replace, color='black', label = 'Number of Herd Replacements', marker = '', linestyle = '-')
        p3.plot(t, av_infectious, color='red', label = 'Average Infectious per Herd', marker = '', linestyle = '-')
        p1.legend()
        p2.legend()
        p3.legend()
        fig.savefig(fnm + '.jpg',bbox_inches='tight', dpi=150)
        plt.close('all')
        return None
    
    def _step(self, action: np.ndarray):
        '''
        Step completes one time step in the environment.
        First, transfers subjects if time interval dictates it.
        Then, calls model(action) to complete a time step in each herd.
        Afterwards, tests subjects if action dictates it and outputs testresults
        if time for testing has been concluded.
        Finally, calculates reward and returns a Time_Step object.
        TimeStep(StepType.MID, reward=reward, discount=self._discount, observation=self._observation)
        '''
        if self._current_time_step.is_last():
            return self.reset()
        
        #Converting continuous inputs into discrete actions while maintaining gradient
        for a in range (0, self._num_herds):
            floor = np.floor(action[a]*self._state[a])
            diff = (action[a]*self._state[a]) - floor
            action[a] = floor + np.int32(bernoulli.rvs(diff, size = None))
        for b in range (self._num_herds, self._num_herds*2):
            action[b] = np.int32(bernoulli.rvs(action[b], size = None))
            
        self._time += 1
        
        #Transfers
        indices = np.arange(self._num_herds)
        np.random.shuffle(indices)
        for i in range (0, self._num_transfers):
            origin_herd = indices[i % (self._num_herds-1)]
            if i % self._num_herds == 0:
                target_herd = indices[self._num_herds-1]
            else:
                target_herd = indices[(i % (self._num_herds-1))-1]
            transfers = self._transfer(origin_herd = origin_herd, target_herd = target_herd)
            back_transfers = self._transfer(origin_herd = target_herd, target_herd = origin_herd)
            if transfers is not None:
                self._state[origin_herd+self._num_herds] = self._state[origin_herd+self._num_herds] - transfers[1] + back_transfers[1]
                self._state[target_herd+self._num_herds] = self._state[target_herd+self._num_herds] + transfers[1] - back_transfers[1]
            
        #Model should make a step in between transfer and test
        self._state = self._model(action)
                
        #Testing
        for i in range (0, self._num_herds):
            self._tests.append(self._test(herd = i, num_tests = action[i]))
        lim = np.ma.size(self._tests, axis = 0)
        get_obs = False
        

        for i in reversed (range (0, lim)):
            if self._tests[i][0] < self._weeks_until_testresults:
                self._tests[i][0] += 1
            elif self._tests[i][0] > self._weeks_until_testresults:
                raise ValueError()
            elif self._tests[i][0] == self._weeks_until_testresults:
                get_obs = True
                break
        
        if get_obs:
            for j in range (0, self._num_herds):
                k = j*3
                if self._tests[j][1] == 0 and self._tests[j][2] == 0:
                    self._observation[k] += 1
                else:
                    self._observation[k] = self._tests[j][0]
                    self._observation[k+1] = self._tests[j][1]
                    self._observation[k+2] = self._tests[j][2]

            for k in range(0, self._num_herds):
                self._tests.pop(k)
        else:
            for l in range (0, np.size(self._observation), 3):
                self._observation[l] += 1
        

        #Reward function
        self._reward = self._reward_func(action)
        #step_reward = np.float32(0)
            
        #Plotting
        average = 0
        num_replace = 0
        total_infectious = 0
        for i in range (0, self._num_herds):
            average += action[i]
            
        average = average / self._num_herds
        for j in range (self._num_herds, self._num_herds*2):
            num_replace += action[j]
            total_infectious += self._state[j]
        av_infectious = total_infectious / self._num_herds
        act = [average, num_replace, av_infectious]
        self._a_and_s.append(act)
        
        #output
        if self._time == self._episode_length:
            self.plot_actions_and_states()
            #scaled_reward = np.float32(self._reward / self._episode_length)
            return TimeStep(StepType.LAST, reward=self._reward, discount=self._discount, observation=self._observation)
        else:
            return TimeStep(StepType.MID, reward=self._reward, discount=self._discount, observation=self._observation)
