# Second Phase Environment implemented based on Env_P1 but for n herds in a population
'''
TODOS:
-Add Transfer Block Action
-Test output as percentage of herd population or total tests? (Total tests for now)
-Add global time as observation
-Figure out how to decide episode length (fixed length for now?)
-Clean up redundancies
-Add some assertion errors *Check(for now)
-Write proper tests
-split even issue?
-Check transfer pair finding  *Check
-Look through chat and notes for further improvements

'''
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import copy
import random
from scipy.stats import poisson
from scipy.stats import geom
from scipy.stats import truncnorm
from scipy.stats import hypergeom
from scipy.stats import bernoulli

from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType, TimeStep, termination, transition


class Env(py_environment.PyEnvironment):
    def __init__(self,
                num_herds = 10,
                total_population = 3000,
                split_even = True,
                population_range = None,
                num_transfers = 10,
                weeks_until_exchange = 3,
                rand_recovery_prob = 0.005,
                rand_infection_prob = 0.01,
                mean_episode_length = 270
                ):
        super(Env, self).__init__()
        # State: [population_herd_1, ... , population_herd_n, infectious_herd_1, ... , infectious_herd_n]
        self._state = np.zeros(((num_herds*2),), np.int32)
        # Observation: [time_since_test_herd_1, negative_tests_herd_1, positive_tests_herd_1, ... , positive_tests_herd_n]
        self._observation = np.zeros(((num_herds*3),), np.int32)
        # Some fixed values
        self._discount = np.float32(1)
        self._time = 0
        self._episode_length = 0#
        self._mean_episode_length = np.int32(mean_episode_length)
        # List of Tests for output
        self._tests = []
        # Reward Function values
        self._reward = np.float32(0)
        self._c_tests = 0.25   #cost for each test
        self._c_prime_tests = 5    #organizational costs tests
        self._e_removed = 2.5   #individual replacement cost
        self._weeks_until_testresults = 3
        # Params for a later feature with differently sized herds
        self._split_even = split_even
        self._population_range = population_range
        # Model defining metrics
        self._num_herds = num_herds
        self._num_transfers = num_transfers
        self._total_population = total_population
        self._exchanged_members = max(4, np.int32(np.round(total_population / (num_transfers*num_herds*5))))    #k from scrapsheet
        self._weeks_until_exchange = weeks_until_exchange    #T from scrapsheet
        self._rand_recovery_prob = rand_recovery_prob    #g from scrapsheet
        self._rand_infection_prob = rand_infection_prob    #q from scrapsheet
    
    def action_spec(self):
        # Actions: [num_tests_herd_1, ... , num_tests_herd_n, slaughter_herd_1, ... slaughter_herd_n]
        return BoundedArraySpec(((self._num_herds*2),), np.float32, minimum=np.float32(0), maximum=np.float32(1))
    
    
    def observation_spec(self):
        # Observation: [time_since_test_herd_1, negative_tests_herd_1, positive_tests_herd_1, ... , positive_tests_herd_n]
        return BoundedArraySpec(((self._num_herds*3),), np.float32, minimum=np.float32(0), maximum=np.float32(1))
    
    def _check_values(self):
        assert self._num_herds >= 2, "Please set num_herds to at least 2."
        assert self._total_population >= 10, "Please set total_population to at least 10."
        assert self._exchanged_members <= (self._total_population/self._num_herds), "More subjects transferred than available."
        for i in range (0, np.size(self._observation)):
            assert 0 <= self._observation[i] <= 1, "Check observation values"
        return True
        
    def _reset(self):
        '''
        Resets all variables the model could change over time.
        State will be reset to no infected except for random number of initial infected in herd 1 
        (between 1 and (1/8)*population_herd_1).
        Observation will be reset to zeros.
        Returns TimeStep object with StepType.First.
        '''
        self._check_values()
        self._state = np.zeros(((self._num_herds*2),), np.int32)
        if self._split_even:
            for i in range (0, self._num_herds):
                self._state[i] = self._total_population / self._num_herds
        else:
            raise NameError('Feature not implemented.')
        
        initial_infected_h1 = np.random.randint(low = 1, high = (self._state[0]/8))
        self._tests = []
        self._time = 0
        self._reward = np.float32(0)
        lower, upper = np.int32(self._mean_episode_length/2.5), np.int32(self._mean_episode_length*2.5)
        mu, sigma = self._mean_episode_length, 40
        X = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        self._episode_length = np.int32(X.rvs(size = None)) #min(50 + geom.rvs(p = 1/104), 500) 
        self._state[self._num_herds] = initial_infected_h1    #infected h1
        self._observation = np.zeros(((self._num_herds*3),), np.float32)
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
            if action[i] == 1:
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
        for i in range (0, self._num_herds):
            self._reward -= self._discount * (action[i] * self._c_tests + min(action[i],1) * self._c_prime_tests) / (self._total_population/self._num_herds)
            self._reward -= self._discount * (action[i+self._num_herds] * self._state[i] * self._e_removed ) / (self._total_population/self._num_herds)
            self._reward -= self._discount * (self._state[i+self._num_herds])**1.5 / (self._total_population/self._num_herds)
        return self._reward
    
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
        
        #Converting continuous inputs into discrete actions while maintaining gradient
        for a in range (0, self._num_herds):
            floor = np.floor(action[a]*self._state[a])
            diff = (action[a]*self._state[a]) - floor
            action[a] = floor + np.int32(bernoulli.rvs(diff, size = None))
        for b in range (self._num_herds, self._num_herds*2):
            action[b] = np.int32(bernoulli.rvs(action[b], size = None))
            
        self._time += 1
        
        #Transfers
        '''
        The Idea is to make an array of indices of herds.
        If indices[0] is picked for n herds, make the transfer target indices[n-1]. 
        In any other case, pick the previous indices entry, i.e. herd indices[i] transfers to herd indices[i-1].
        
        '''
        indices = np.arange(self._num_herds)
        np.random.shuffle(indices)
        for i in range (0, self._num_transfers):
            origin_herd = indices[i % self._num_herds]
            if i % self._num_herds == 0:  
                target_herd = indices[self._num_herds-1]
            else:
                target_herd = indices[(i % self._num_herds)-1]
            assert target_herd != origin_herd, 'Target herd and origin herd must not be the same.'
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
                    self._observation[k] += 1 / self._episode_length
                else:
                    self._observation[k] = self._tests[j][0] / self._episode_length
                    test_sum = self._tests[j][1] + self._tests[j][2]
                    self._observation[k+1] = self._tests[j][1] / test_sum
                    self._observation[k+2] = self._tests[j][2] / test_sum

            for k in range(0, self._num_herds):
                self._tests.pop(k)
        else:
            for l in range (0, np.size(self._observation), 3):
                self._observation[l] += 1 / self._episode_length
        

        #Reward function
        self._reward = np.float32(self._reward_func(action))
        #step_reward = np.float32(0)
            
        #Testing
        #self._check_values()
        
        #output
        if self._time == self._episode_length:
            return termination(self._observation, self._reward)
        else:
            return transition(self._observation, self._reward, np.float32(self._discount))
