#!/usr/bin/env python
# coding: utf-8

# First Phase Environment implemented according to scratch notes from call on 12/11/20

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
from scipy.integrate import odeint

from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep


# # Documentation
# Env_P1 is a class that represents an epidemic with two herds.  
# 
# <img src="Sketch.jpeg"
#      alt="Env_P1 Sketch"
#      style="float: left; margin-right: 5px;" />
# ## Variables:  
# The action $\in \mathbb{R}^4$ is a vector $(\tau_1, \tau_2, s_1, s_2)$.  
# $\tau_i$ are the number of tests to be done in herd $i$.  
# $s_i \in \{0,1\}$ determine whether a herd is to be completely replaced by healthy members.  
# $S_i$ is the number of susceptible herd members (subjects) of herd $i$, $i \in {0,1}$.  
# $I_i$ is the number of infectious subjects of herd $i$, $i \in {1,2}$.  
# $g$ is a small recovery probability.  
# $q$ is a small infection probability.  
# $B_i' = min(B_i,|S_i|-1)$ with $B_i = \sum_{I_i} A$ and $A \sim Poi(0.01)$, is the number of Susceptibles moving to Infectious per time step for herd $i$, $i \in {1,2}$.  
# ## State  
# The state contains two arrays of size six. 
# state[0] is the observation shown to the agent.  
# The observation are testresults for each herd $(\mu_i, x^i_0, x^i_1)$.  
# $\mu_i$ shows the number of time steps since the test has been done.  
# $x^i_0$ and $x^i_1$ correspond to negative and positive testresults respectively.  
# 
# state[1] instead contains the internal information about both herds.  
# state[1][i] shows the population size for herd i.  
# state[1][i+2] shows the total number of infected subjects for herd i.  
# ## Reward  
# Reward calculation respects costs for tests and herd replacement.  
# Let $R$ be the current reward (starts at zero). Then each time step:  
# 
# $R - \tau_i * c + 1_{\tau_i} * cprime, \forall i \in \{0,1\}$,  
# $R - s_i * state[1][i] * e, \forall i \in \{0,1\}$. 
#   
# Here, $c$, $cprime > c$ and $e$ are constants.  
# Also, reduces reward by number of infectious:  
# 
# $R - state[1][i+2], \forall i \in \{0,1\}$. 
# 

# In[2]:


class P_Env_P1(py_environment.PyEnvironment):
    def __init__(self,
                root_dir, #path for plotting
                global_step = 0, #ID for plotting
                population_herd1 = 200,
                population_herd2 = 50,
                exchanged_members = 5,
                weeks_until_exchange = 4,
                rand_recovery_prob = 0.005,
                rand_infection_prob = 0.01,
                ):
        super(P_Env_P1, self).__init__()
        self._state = None
        self._observation = None
        self._discount = np.float32(1)
        self._time = 0
        self._episode_length = 0
        self._tests = []
        self._reward = np.float32(0)
        self._c_tests = 0.5   #cost for each test
        self._c_prime_tests = 10    #organizational costs tests
        self._e_removed = 3   #individual replacement cost
        self._weeks_until_testresults = 3
        self._population_herd1 = population_herd1
        self._population_herd2 = population_herd2
        self._exchanged_members = exchanged_members    #k from scrapsheet
        self._weeks_until_exchange = weeks_until_exchange    #T from scrapsheet
        self._rand_recovery_prob = rand_recovery_prob    #g from scrapsheet
        self._rand_infection_prob = rand_infection_prob    #q from scrapsheet
        
        self._actions = []
        self._states = []
        self._root_dir = root_dir
        self._global_step = global_step
        
    def action_spec(self):
        #Actions for: number of subjects to be tested h1, h2. number of subjects to be eliminated h1, h2
        return BoundedArraySpec((4,), np.float32, minimum=0, maximum=1)
    
    
    def observation_spec(self):
        # tau, x0, x1 for both herds
        return BoundedArraySpec((6,), np.int32, minimum=0, maximum=2000)
    
    
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
        self._actions = []
        self._states = []
        self._tests = []
        self._state = np.zeros((4,), np.int32)
        initial_infected_h1 = np.random.randint(low = 1, high = (self._population_herd1/8))
        self._time = 0
        self._reward = np.float32(0)
        self._episode_length = geom.rvs(p = 1/270)
        self._state[3] = 0    #infected h2
        self._state[2] = initial_infected_h1    #infected h1
        self._state[1] = self._population_herd2
        self._state[0] = self._population_herd1
        self._observation = np.zeros((6,), np.int32)
        return TimeStep(StepType.FIRST, reward=self._reward,
                    discount=self._discount, observation = self._observation)
    
    def _test(self, herd = -1, num_tests = 0):
        '''
        Randomly draws (without returning) num_tests subjects of a herd,
        then tests whether they are infected or not before returning testresults.
        '''
        assert self._state[herd] >= num_tests, "More tests than herd members."
        if herd >= 0 and num_tests > 0:
            test_out = hypergeom.rvs(M = self._state[herd], n = self._state[herd+2], N = int(num_tests), size = None)
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

        assert self._state[origin_herd] > self._exchanged_members, "Population in origin herd too low."
        if origin_herd >= 0 and target_herd >=0 and self._time % self._weeks_until_exchange == 0:
            infected_transfers = hypergeom.rvs(M = self._state[origin_herd], 
                                                 n = self._state[origin_herd+2], N = self._exchanged_members, size = None)
            susceptible_transfers = self._exchanged_members - infected_transfers
            return np.array([susceptible_transfers, infected_transfers])    
        else:
            return None
        
    def _model(self, action: np.ndarray):
        '''
        Completes one time step in a herd (i.e. excluding transfers and tests).
        In f(x), samples new infections from poisson dist with lambda = 0.6,
        also considers spontaneous infection and recovery factors.
        Then, depending on whether a herd is to be replaced by healthy subjects (action),
        calls f(x) or simply replaces all subjects by healthy subjects for each herd.
        '''
        
        initial_state = self._state
        #Model for one herd
        def f(x):
            S = np.int32(x[0])
            I = np.int32(x[1])
            s_to_i = 0
            new_infs = 0
            for i in range (0, I):
                s_to_i += poisson.rvs(0.05)
            s_to_i = np.int32(min(s_to_i, S))
            #code draw whether subject to be infected is already infected or sus
            if s_to_i >= 1:
                new_infs = hypergeom.rvs(M = (S + I), n = S, N = s_to_i, size = None)
            dsdt = S - new_infs - np.int32(self._rand_infection_prob * S) + np.int32(self._rand_recovery_prob * I)
            didt = I + new_infs + np.int32(self._rand_infection_prob * S) - np.int32(self._rand_recovery_prob * I)
            return dsdt, didt
        
        #One step for each herd
        if action[2] == 1:
            initial_state_h2 = np.array([self._state[1]-self._state[3], self._state[3]])
            S1, I1 = self._state[0], 0
            S2, I2 = f(x = initial_state_h2)
        elif action[3] == 1: 
            initial_state_h1 = np.array([self._state[0]-self._state[2], self._state[2]])
            S1, I1 = f(x = initial_state_h1)
            S2, I2 = self._state[1], 0
        else:
            initial_state_h1 = np.array([self._state[0]-self._state[2], self._state[2]])
            initial_state_h2 = np.array([self._state[1]-self._state[3], self._state[3]])
            S1, I1 = f(x = initial_state_h1)
            S2, I2 = f(x = initial_state_h2)
        return np.array([S1 + I1, S2 + I2, I1, I2])
    
    def _reward_func(self, action: np.ndarray):
        '''
        Calculates and returns reward.
        R -= tau_i * C + Indicator_i * C_prime
        Where tau_i is number of tests in each herd, 
        Indicator_i is whether tau_i > 0 and C < C_prime.
        R -= s_i * population_herd_i * replacement_cost
        Where s_i is indicator for whether a herd was replaced by healthy subjects
        and replacement_cost is a constant representing the cost of replacing a single subject.
        '''
        tau_1 = np.int32(action[0] * self._state[0]) 
        tau_2 = np.int32(action[1] * self._state[1])
        indicator_1 = 0
        indicator_2 = 0
        if action[0] > 0:
            indicator_1 = 1
        if action[1] > 0:
            indicator_2 = 1
        self._reward -= self._discount * (tau_1 * self._c_tests + indicator_1 * self._c_prime_tests) / 10
        self._reward -= self._discount * (tau_2 * self._c_tests + indicator_2 * self._c_prime_tests) / 10
        self._reward -= self._discount * ((action[2] * self._state[0] + action[3] * self._state[1]) / 10) * self._e_removed
        self._reward -= self._discount * ((self._state[2] + self._state[3]) / 10)**1.4
        return self._reward
    
    def plot_actions_and_states(self):
        t = np.linspace(0, self._time, num=len(self._actions))
        fig, (p1,p2) = plt.subplots(1, 2, figsize=(20,10))
        fig2, (q1,q2) = plt.subplots(1, 2, figsize=(20,10))
        fig.suptitle('Actions over Time')
        p1.set_title('Tests over Time')
        p1.set_xlabel('Time')
        p1.set_ylabel('Number of Tests')
        p1.set_ylim(-0.5,1.5)
        p2.set_title('Herd Replacement over Time')
        p2.set_xlabel('Time')
        p2.set_ylabel('Replacement')
        p2.set_ylim(-0.5,1.5)
        fig2.suptitle('Tests and Infectious over Time')
        q1.set_title('Herd 1')
        q1.set_xlabel('Time')
        q1.set_ylabel('Tests and Infectious in %')
        q1.set_ylim(-0.5,1.5)
        q2.set_title('Herd 2')
        q2.set_xlabel('Time')
        q2.set_ylabel('Tests and Infectious in %')
        q2.set_ylim(-0.5,1.5)
        root_dir = self._root_dir
        root_dir = os.path.expanduser(root_dir) 
        fnm = os.path.join(root_dir, 'A' + '_' + str(self._time) + '_' + str(self._global_step)) 
        fnm2 = os.path.join(root_dir, 'I+A' + '_' + str(self._time) + '_' + str(self._global_step)) 
        ntests1, ntests2, sone, stwo = [], [], [], []
        S_1, S_2, I_1, I_2 = [], [], [], []
        for i in range(len(self._actions)):
            ntests1.append(self._actions[i][0])
            ntests2.append(self._actions[i][1])
            sone.append(self._actions[i][2])
            stwo.append(self._actions[i][3])
            I_1.append((self._states[i][2]/self._states[i][0]))
            #S_1.append((self._states[i][0]-self._states[i][2]))
            I_2.append((self._states[i][3]/self._states[i][1]))
            #S_2.append((self._states[i][1]-self._states[i][3]))                
        p1.plot(t, ntests1, color='blue', label = 'Tests Herd 1', marker = '.', linestyle = '')
        p1.plot(t, ntests2, color='red', label = 'Tests Herd 2', marker = '.', linestyle = '')
        p2.plot(t, sone, color='blue', label = 'Replace Herd 1', marker = '.', linestyle = '')
        p2.plot(t, stwo, color='red', label = 'Replace Herd 2', marker = '.', linestyle = '')
        q1.plot(t, I_1, color='lightgreen', label = 'Infectious Herd 1')
        q1.plot(t, ntests1, color='blue', label = 'Tests Herd 1')
        q2.plot(t, I_2, color='lightgreen', label = 'Infectious Herd 2')
        q2.plot(t, ntests2, color='blue', label = 'Tests Herd 2')
        p1.legend()
        p2.legend()
        q1.legend()
        q2.legend()
        fig.savefig(fnm + '.jpg',bbox_inches='tight', dpi=150)
        fig2.savefig(fnm2 + '.jpg',bbox_inches='tight', dpi=150)
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
        TimeStep(StepType.MID, reward=reward, discount=self._discount, observation=[self._state[0]])
        
        TODOS: Check chronology
        '''
        if self._current_time_step.is_last():
            return self.reset()
        
        if action[2] != np.float32(1) and action[2] != np.float32(0) and action[2] >= 1/2 and action[3] < 1/2:
            action[2] = np.float32(1)
            action[3] = np.float32(0)
        elif action[3] != np.float32(1) and action[3] != np.float32(0) and action[3] >= 1/2 and action[2] < 1/2:
            action[3] = np.float32(1)
            action[2] = np.float32(0)
        else: 
            action[2] = np.float32(0)
            action[3] = np.float32(0)
            
        self._time += 1
        origin_herd = 0
        target_herd = 1
        transfers = self._transfer(origin_herd = origin_herd, target_herd = target_herd)
        back_transfers = self._transfer(origin_herd = target_herd, target_herd = origin_herd)
        if transfers is not None:
            self._state[origin_herd] = self._state[origin_herd] - transfers[0] - transfers[1] + back_transfers[0] + back_transfers[1]
            self._state[target_herd] = self._state[target_herd] + transfers[0] + transfers[1] - back_transfers[0] - back_transfers[1]
            self._state[origin_herd+2] = self._state[origin_herd+2] - transfers[1] + back_transfers[1]
            self._state[target_herd+2] = self._state[target_herd+2] + transfers[1] - back_transfers[1]
            
        #interpreting actions
        num_test_h1 = np.int32(action[0] * self._state[0])
        num_test_h2 = np.int32(action[1] * self._state[1])
            
        #Model should make a step in between transfer and test
        self._state = self._model(action)
        #Testing 
        self._tests.append(self._test(herd = 0, num_tests = num_test_h1))
        self._tests.append(self._test(herd = 1, num_tests = num_test_h2))
        
        for i in range (0, np.ma.size(self._tests, axis = 0)):
            if self._tests[i][0] == self._weeks_until_testresults:
                self._observation[5] = self._tests[i+1][2]    #x1 tested pos h2
                self._observation[4] = self._tests[i+1][1]    #x0 tested neg h2
                self._observation[3] = self._weeks_until_testresults    
                self._observation[2] = self._tests[i][2]    #x1 tested pos h1
                self._observation[1] = self._tests[i][1]    #x0 tested neg h1
                self._observation[0] = self._weeks_until_testresults
                self._tests.pop(i)
                self._tests.pop(i)
                break
        for i in range (0, np.ma.size(self._tests, axis = 0)):
            self._tests[i][0] += 1 
            
        #Reward function
        self._reward = np.float32(self._reward_func(action))
        step_reward = np.float32(0)
        
        #Plotting
        
        self._actions.append(action)
        self._states.append(self._state)
        
        #output
        if self._time == self._episode_length:
            figure = self.plot_actions_and_states()
            return TimeStep(StepType.LAST, reward=self._reward, discount=self._discount, observation=self._observation)
            
        else:
            return TimeStep(StepType.MID, reward=step_reward, discount=self._discount, observation=self._observation)
