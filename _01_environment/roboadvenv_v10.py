import gym
from gym.spaces import Tuple, Discrete, Box
import numpy as np
import pandas as pd
import random

ROBO_ADVISOR_DATA_FILE = "D:/data_mt/09_training/robo_train_set.csv"


class RoboAdvisorEnvV10(gym.Env):

    def __init__(self):
        super(RoboAdvisorEnvV10, self).__init__()
        self.step_counter = 0

        self.data = self._load_data()

        # first monday in 2017
        self.start_monday = pd.to_datetime("2017-01-02")

    def reset(self):
        # state reseted
        return np.array([0])

    def step(self, actions):
        self.step_counter += 1

        # state, reward, done, ...
        return np.array([0]), 0, False, {}

    def _load_data(self):
        df = pd.read_csv(ROBO_ADVISOR_DATA_FILE, sep=',', encoding='utf-8', header=0)
        df['Date'] = pd.to_datetime(df.Date)
        df['day_of_week'] = df.Date.dt.dayofweek

        return df

    def _find_evaluation_day(self):
        """ advances current date to next monday or next trading day if a monday has been a bank holiday"""
        next_monday = self.start_monday + pd.DateOffset(days=(self.step_counter * 7))
