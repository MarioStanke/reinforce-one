#!/usr/bin/env python
# coding: utf-8

# ReinforceOne
# PyEnvironment for infection transmission and its control in herds
# Simple version with discrete actions and without transmissions between herds.

import matplotlib
import matplotlib.pyplot as plt
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

class FREnv(py_environment.PyEnvironment):
    """
    Very simple environment for transmission within herds (no migration, no testing).
    The only action is culling the whole herd.
    """
    def __init__(self,
                herd_sizes = [32,32],
                expected_episode_length = 100, # if non-positive episodes all have max_episode_length (270)
                max_episode_length = 5000, #
                rand_recovery_prob = 0.008,
                rand_infection_prob = 0.01,
                culling_cost_herd  = 0.,   # herd replacement fixed costs
                culling_cost_individual  = 1.,   # individual replacement cost
                cost_infected = .025  # cost for each step and infected at end
                ):
        super(FREnv, self).__init__()
        self._discount = np.float32(1)
        self._time = 0
        self._episode_length = 0
        self._reward = np.float32(0)
        # overall limit of the otherwise random episode length
        self._max_episode_length = max_episode_length 
        self._expected_episode_length = expected_episode_length
        self._culling_cost_herd  = culling_cost_herd
        self._culling_cost_individual = culling_cost_individual
        self._cost_infected = cost_infected
        self._herd_sizes = herd_sizes
        self._num_herds = len(herd_sizes)
        self._total_population = np.array(herd_sizes).sum()
        self._rand_recovery_prob = rand_recovery_prob    #g from scrapsheet
        self._rand_infection_prob = rand_infection_prob    #q from scrapsheet
        
        # state: for each herd
        # - the current number of infected I
        # - the cumulative number of infectec since the episode start and
        # - the time lastC since last culling 
        # I[0], I[1], ..., lastC[0], lastC[1], .... 
        self._state = np.zeros(self._num_herds*3, np.int32)
        
        # For each herd only the times since last culling are observed
        # and returned as floats for learning.
        # time, lastC[0], lastC[1], ....
        self._observation = np.zeros(1+self._num_herds, np.float32) 
        
        
    def action_spec(self):
        # For each heard a probability for culling
        # ddpg requires continuous actions 
        return BoundedArraySpec(((self._num_herds),), dtype=np.float32,
                                minimum=0., maximum=1., name="action")
    
    def observation_spec(self):
        ''' For each herd the time since last culling is observed.'''
        return BoundedArraySpec((1+self._num_herds,), dtype=np.float32,
                                minimum=0,
                                maximum=1,
                                name="observation")
    
    
    def _reset(self):
        ''' at start new episode '''
        
        # assume 0 infected per herd in the beginning,
        # 0 steps since last culling
        self._state = np.array([0] * self._num_herds + # current infected
                               [0] * self._num_herds + # cumulative infected
                               [0] * self._num_herds,  # time since last culling
                               np.int32)        
        self._time = 0
        self._reward = np.float32(0)
        
        # determine episode length
        # either fixed at _max_episode_length or truncated geometric
        if self._expected_episode_length <= 0:
            self._episode_length = self._max_episode_length
        else:
            self._episode_length = geom.rvs(p = 1./self._expected_episode_length)
            if self._episode_length > self._max_episode_length:
                self._episode_length = self._max_episode_length
                
        # observe the global time ...
        self._observation[0] = self._time
        # ... and the last third of state - the culling times
        self._observation[1:] = self._state[2*self._num_herds:].astype(np.float32)
        return TimeStep(StepType.FIRST, reward=self._reward,
                    discount=self._discount, observation = self._observation)
        
    def _model(self, action):
        '''
        Completes one time step in a herd (i.e. excluding transfers and tests).
        '''
        # Model for one herd
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
        
        # One step for each herd
        for i in range (self._num_herds):
            if action[i] == 1: # bit of i-th herd is set in action integer
                self._state[i] = 0 # herd is culled, starts with 0 infections
                self._state[2*self._num_herds + i] = 0. # clock reset, 0 steps since last culling
            else:
                self._state[2*self._num_herds + i] += 1. # one more step since last culling
                I = self._state[i]
                S = self._herd_sizes[i] - I
                self._state[i] = f(S, I)
                if self._state[i] > self._herd_sizes[i]:
                    self._state[i] = self._herd_sizes[i]
        # accumulate: add numbers of currently infected to cumulative infected
        self._state[self._num_herds:2*self._num_herds] += self._state[:self._num_herds]

        self._observation[0] = self._time
        self._observation[1:] = self._state[2*self._num_herds:].astype(np.float32) 
        self._observation /= self._max_episode_length
        return self._state
    
    def _action_array(self, action:float):
        """ get the Boolean array of actions per herd from the probabilities"""
        action_array = np.zeros(self._num_herds, np.int32)
        for i in range (0, self._num_herds):#
            act = np.int32(bernoulli.rvs(action[i], size = None))
            assert (act == 0 or act == 1), "Action takes weird values:" + str(act)
            if (act == 1):
                action_array[i] = 1
        return action_array
    
    def _reward_func(self, action):
        '''
        Calculates and returns reward.
        '''
        r = 0
        for i in range (0, self._num_herds):
            # penalize culled herds with a per-animal cost
            if action[i] == 1: # i-th herd is completely slaughtered
                r += - self._discount * (
                    self._culling_cost_herd +
                     self._herd_sizes[i] * self._culling_cost_individual)
        cumI = self._state[self._num_herds:2*self._num_herds]
        r += - cumI.sum() * self._cost_infected
        return r
 
    def _step(self, action):
        '''
        Step completes one time step in the environment.
        Then, calls model(action) to complete a time step in each herd.
        Finally, calculates reward and returns a Time_Step object.
        TimeStep(StepType.MID, reward=reward, discount=self._discount, observation=self._observation)
        '''
        if self._current_time_step.is_last():
            return self.reset()
        action = self._action_array(action)
        
        self._time += 1
        self._model(action) # changes _state, _observation
        
        st = StepType.MID
        self._reward = self._reward_func(action)
        if self._time == self._episode_length:
            st = StepType.LAST
            return termination(self._observation, self._reward)
        else:
            return transition(self._observation, self._reward, self._discount)
        
    def get_state(self):
        return self._state