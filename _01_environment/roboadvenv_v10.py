from _01_environment.universe import InvestUniverse

import gym
from gym.spaces import Tuple, Discrete, Box
import numpy as np
import pandas as pd
from pandas import Timestamp




class RoboAdvisorEnvV10(gym.Env):

    def __init__(self, universe : InvestUniverse):
        super(RoboAdvisorEnvV10, self).__init__()

        self.universe = universe

        self.trading_days_ser = pd.Series(self.universe.get_trading_days())

        self.step_counter = 0

        # first monday in 2017
        self.start_monday = pd.to_datetime("2017-01-02")
        self.current_evaluation_day: Timestamp = self._find_next_trading_day(self.start_monday)
        self.current_trading_day: Timestamp = self._find_next_trading_day(self.current_evaluation_day + pd.DateOffset(1))

    def reset(self):
        # state reseted
        return np.array([0])

    def step(self, actions):
        self.step_counter += 1

        # state, reward, done, ...
        return np.array([0]), 0, False, {}


    def _advance_time(self):
        current_monday = self.start_monday + pd.DateOffset(self.step_counter * 7)
        self.current_evaluation_day = self._find_next_trading_day(current_monday)
        self.current_trading_day = self._find_next_trading_day(self.current_evaluation_day + pd.DateOffset(1))

    def _find_next_trading_day(self, date: Timestamp):
        return self.trading_days_ser[self.trading_days_ser >= date].min()





        """ advances current date to next monday or next trading day if a monday has been a bank holiday"""
        #next_monday = self.start_monday + pd.DateOffset(days=(self.step_counter * 7))
