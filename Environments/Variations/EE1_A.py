
'''
Epidemic environment EE0 implemented for n herds.
Actions: Tests, culls (for each herd)
Observations: Time since test, test size, positive tests, time since last cull (for each herd)

EE1_A variant changes: 
- weeks_until_exchange 2 -> 10
- exchanged members 0.05 -> 0.08
- infection_rates [0.05,0.05] -> [0.02, 0.08]
- transfers only from herd 0 to herd 1
- self._cost_removed 1 -> 4
- action conversion like EE0_A
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


class EE1_A(py_environment.PyEnvironment):
    def __init__(self,
                 num_herds = 2,
                 total_population = 300,
                 infection_rates = [0.02, 0.08],
                 exchanged_members = 0.08,    
                 weeks_until_exchange = 10,
                 weeks_until_testresults = 0.,
                 rand_recovery_prob = 0.003,     #0.0005
                 rand_infection_prob = 0.00015,    #0.1
                 fix_episode_length = True,
                 average_episode_length = 200
                ):
        super(EE1_A, self).__init__()
        # State: [population_herd_1, ... , population_herd_n, infectious_herd_1, ... , infectious_herd_n]
        self._state = np.zeros(((num_herds*2),), np.int32)
        # Observation: [time_since_test_herd_1, number_tested_herd_1, positive_tests_herd_1, time_since_last_cull_herd_1 
        # ... , time_since_test_herd_n, number_tested_herd_n, positive_tests_herd_n, time_since_last_cull_herd_n]
        self._observation = np.zeros(((num_herds*4),), np.float32)
        # Discount & time
        self._discount = np.float32(1)
        self._time = 0
        # Episode length
        self._fix_episode_length = fix_episode_length
        self._episode_length = 0
        self._average_episode_length = np.int32(np.rint(average_episode_length))
        # List of tests for output
        self._tests = []
        self._weeks_until_testresults = weeks_until_testresults
        # Reward function values
        self._reward = np.float32(0)
        self._c_tests = 0.01   #cost for each test 0.05 
        self._c_prime_tests = 1.   #organizational costs tests for each herd
        self._cost_removed = 4.   #herd replacement cost
        self._cost_infectious = 2.  #'Cost' for infectious each step
        # Model defining parameters
        self._inf_rates = infection_rates
        self._num_herds = num_herds
        self._num_transfers = np.int32(num_herds/2)
        self._total_population = total_population
        self._exchanged_members = max(1, np.int32(np.rint((total_population/num_herds)*exchanged_members)))    #k from scrapsheet
        self._weeks_until_exchange = weeks_until_exchange    #T from scrapsheet
        self._rand_recovery_prob = rand_recovery_prob    #g from scrapsheet
        self._rand_infection_prob = rand_infection_prob    #q from scrapsheet
    
    def action_spec(self):
        # Actions: [num_tests_herd_1, ... , num_tests_herd_n, slaughter_herd_1, ... slaughter_herd_n]
        return BoundedArraySpec(((self._num_herds*2),), np.float32, minimum=np.float32(0), maximum=np.float32(1))
    
    
    def observation_spec(self):
        # Observation: [time_since_test_herd_1, percentage_tested_herd_1, percentage_positive_tests_herd_1, time_since_last_cull_herd_1, ... ,
        #               percentage_positive_tests_herd_n, time_since_last_cull_herd_n]
        return BoundedArraySpec(((self._num_herds*4),), np.float32, minimum=np.float32(0), maximum=np.float32(1))
    
    def _check_values(self):
        assert self._num_herds >= 2, "Please set num_herds to at least 2."
        assert (self._total_population / self._num_herds).is_integer(), "Please enter a population number that can be distributed evenly among herds."
        assert self._total_population >= 10, "Please set total_population to at least 10."
        assert self._exchanged_members <= (self._total_population/self._num_herds), "More subjects transferred than available."
        assert (len(self._inf_rates) == self._num_herds), 'Please enter one infection rate for each herd in a list.'
        for i in range (0, np.size(self._observation)):
            assert 0 <= self._observation[i] <= 1, "Check observation values"
        return True
    
    def get_state(self):
        return self._state
        
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

        self._tests = []
        self._time = 0
        self._reward = np.float32(0)
        if self._fix_episode_length: 
            self._episode_length = self._average_episode_length
        else:
            self._episode_length = geom.rvs(p = 1/self._average_episode_length)
        self._observation = np.zeros(((self._num_herds*4),), np.float32)
        self._check_values()
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
        if origin_herd >= 0 and target_herd >=0 and (self._time % self._weeks_until_exchange) == 0:
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
        # Model for one herd
        # S: Susceptible, I: Infected/Infectious
        def f(S, I, herd): 
            new_infs = 0
            herd_population = S+I
            inf_rvs = poisson.rvs(mu = self._inf_rates[herd], size = I)
            rec_rvs = 0
            if (I > 0):
                rec_rvs = poisson.rvs(mu = self._rand_recovery_prob*I)
            rand_rvs = poisson.rvs(mu = self._rand_infection_prob*S)
            potential_infs = np.sum(inf_rvs)
            recoveries = np.sum(rec_rvs)
            rand_infs = np.sum(rand_rvs)
            potential_infs = min(potential_infs, S)
            # Check whether subject to be infected is already infected or susceptible
            if potential_infs >= 1:
                new_infs = hypergeom.rvs(M = herd_population, n = S, N = potential_infs, size = None)
            new_I = I + new_infs + rand_infs - recoveries
            new_I = min(herd_population, max(0, new_I))
            return new_I
        
        # Complete one model step for each herd
        step_results = self._state
        for i in range (self._num_herds, self._num_herds*2):
            if action[i] == 1:
                step_results[i] = 0
            else:
                step_results[i] = f(S = (self._state[i-self._num_herds] - self._state[i]), I = self._state[i], herd = (i-self._num_herds))
        return step_results
    
    def _reward_func(self, action: np.ndarray):
        '''
        Calculates and returns reward. 
        Negative Reward for infectious grows (or rather decreases) exponentially, 
        others have a linear decrease.
        '''
        step_reward = 0.
        for i in range (0, self._num_herds):
            step_reward -= self._discount * (action[i] * self._c_tests + min(action[i],1) * self._c_prime_tests) / self._state[i]
            step_reward -= self._discount * (action[i+self._num_herds] * self._state[i] * self._cost_removed) / self._state[i]
            step_reward -= self._discount * (self._state[i+self._num_herds] * self._cost_infectious) / self._state[i]
        step_reward = step_reward / self._num_herds
        return step_reward
    
    # Get state for plotting infectious
    def get_states():
        return self._state
    
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
            if action[a] < 1/3:
                action[a] = 0
            elif action[a] < 2/3:
                action[a] = np.rint(self._state[a]/2)
            else: 
                action[a] = self._state[a]
        for b in range (self._num_herds, self._num_herds*2):
            if (action[b] >= 0.5):
                action[b] = 1
            else:
                action[b] = 0
            
        self._time += 1
        
        # Transfers
        '''
        The idea is to create an array of indices of herds and shuffle it randomly.
        If indices[0] is picked for n herds, make the transfer target indices[n-1]. 
        In any other case, pick the previous indices entry, i.e. herd indices[i] transfers to herd indices[i-1].
        
        '''
        indices = np.arange(self._num_herds)
        tmp_targets = indices[indices % 2 == 1]    # feeder_herds
        tmp_origins = indices[indices % 2 == 0]    # grower_herds
        if self._num_herds == 2:
            assert tmp_targets[0] == 1,'transfer implementation err 1'
            assert tmp_origins[0] == 0,'transfer implementation err 2'
        for i in range (0, self._num_transfers):
            np.random.shuffle(tmp_origins)
            np.random.shuffle(tmp_targets)
            origin_herd = tmp_origins[i % np.int32(tmp_origins.size)]
            target_herd = tmp_targets[i % np.int32(tmp_targets.size)]
            assert target_herd != origin_herd, 'Target herd and origin herd must not be the same.'
            transfers = self._transfer(origin_herd = origin_herd, target_herd = target_herd)
            if transfers is not None:
                replaced_inf = 0
                if (transfers[0] > 0) and (self._state[target_herd+self._num_herds] > 0):
                    replaced_inf = hypergeom.rvs(M = self._state[target_herd], 
                                                 n = self._state[target_herd+self._num_herds], N = (transfers[0]+transfers[1]), size = None)
                self._state[origin_herd+self._num_herds] = max(0, self._state[origin_herd+self._num_herds] - transfers[1])
                self._state[target_herd+self._num_herds] = max(0, min(self._state[target_herd+self._num_herds] + transfers[1] - replaced_inf, 
                                                               self._state[target_herd]))
        # Model completes a step in between transfers and tests
        self._state = self._model(action)     
        
        # Testing
        for i in range (0, self._num_herds):
            self._tests.append(self._test(herd = i, num_tests = action[i]))
        lim = np.ma.size(self._tests, axis = 0)
        get_obs = False
        

        for i in reversed (range (0, lim)):
            if self._tests[i][0] < self._weeks_until_testresults:
                self._tests[i][0] += 1
            elif self._tests[i][0] == self._weeks_until_testresults:
                get_obs = True
                break
            elif self._tests[i][0] > self._weeks_until_testresults:
                raise ValueError()
        
        if get_obs:
            for j in range (0, self._num_herds):
                k = (j*4)
                if self._tests[j][1] == 0 and self._tests[j][2] == 0:
                    self._observation[k] += np.float32(1 / self._episode_length)
                else:
                    self._observation[k] = np.float32(self._tests[j][0] / self._episode_length)
                    test_sum = self._tests[j][1] + self._tests[j][2]
                    self._observation[k+1] = np.float32(test_sum / self._state[j])  # number of tests / herd size
                    self._observation[k+2] = np.float32(self._tests[j][2] / test_sum)  # pos. tests / total tests
                if action[j+self._num_herds] == 1:
                    self._observation[k+3] = 0.
                else:
                    self._observation[k+3] += np.float32(1 / self._episode_length)

            for k in range(0, self._num_herds):
                test0 = self._tests.pop(0)
        else:
            for l in range (0, self._num_herds):
                k = (l*4)
                self._observation[(l*4)] += 1 / self._episode_length
                if action[j+self._num_herds] == 1:
                    self._observation[k+3] = 0.
                else:
                    self._observation[k+3] += np.float32(1 / self._episode_length)
        
        # Reward function
        step_reward = np.float32(self._reward_func(action))            
        
        # Output
        if self._time == self._episode_length:
            return termination(self._observation, step_reward)
        else:
            return transition(self._observation, step_reward, np.float32(self._discount))
       