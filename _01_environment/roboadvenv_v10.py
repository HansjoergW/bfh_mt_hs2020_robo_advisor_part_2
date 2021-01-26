from _01_environment.universe import InvestUniverse
from _01_environment.portfolio import Portfolio

import gym
from gym.spaces import Tuple, Discrete, Box
import numpy as np
import pandas as pd
from pandas import Timestamp
from typing import List, Dict, Union

# Todo
# - as an additional input the "deviation" from prediction and real potential a year ago could be used.
#   with this value the agen might be able to take a guess on how good the prediction for a stock tends to be


class RoboAdvisorEnvV10(gym.Env):

    def __init__(self, universe : InvestUniverse):
        super(RoboAdvisorEnvV10, self).__init__()

        self.universe = universe
        self.trading_days_ser = pd.Series(self.universe.get_trading_days())
        self.start_cash = 100_000.0

        self.step_counter = 0
        
        # first friday in 2017
        self.start_friday = pd.to_datetime("2017-01-06")
        self.current_evaluation_day: Timestamp = self.universe.find_trading_day_or_before(self.start_friday)
        self.current_trading_day: Timestamp = self.universe.find_trading_day_or_after(self.current_evaluation_day + pd.DateOffset(1))

        self.portfolio: Union[Portfolio, None] = None

    def reset(self):
        self.step_counter = 0
        self.portfolio = Portfolio(self.universe, self.start_cash)
        # state reseted
        return np.array([0])

    def step(self, actions):
        # actions are executed on the current trading day
        
        # after that, we need to advance
        self.step_counter += 1
        self._advance_time()
        
        # and calculate the new state based on the new current evaluation day
        # .. todo create state
        
        # state, reward, done, ...
        return np.array([0]), 0, False, {}

    def _advance_time(self):
        current_friday = self.start_friday + pd.DateOffset(self.step_counter * 7)

        # generally, we evaluate the situation on Friday evening. So we have to find the last
        # trading information wich is either Friday or a day before Friday (in case of bank holidays
        self.current_evaluation_day = self.universe.find_trading_day_or_before(current_friday)

        # trades are then executed on the next trading day, which is the following monday or
        # the next trading day after, if this monday is a bank holidy
        self.current_trading_day = self.universe.find_trading_day_or_after(self.current_evaluation_day + pd.DateOffset(1))

    def _calculate_state(self, date: Timestamp):
        # state information exist on the following information for each ticker (company) in the universe
        # - the current prediction
        # - how many days ago that stock had been bought (or -1 if it isn't in the portfolio)
        # - value_change in % since bought (or 0 if not in the portfolio)
        # - prediction change in % since bought (or 0 if not in the portfolio)
        # - (the prediction at which it had been bought)
        # - (the value at which it had been bought)

        # returns [ticker,prediction]
        falsch hier möchte ich ticker und current_prediction und current close
        predictions = self.universe.get_predictions_per(date)

        # returns index = ['shares', 'buy_prediction', 'buy_price', 'buy_date']
        current_positions = self.portfolio.get_positions(date)

        print("")

        # we normalisieren? muss nicht perfekt sein
        # https://stats.stackexchange.com/questions/347623/using-non-normalized-data-for-learning-a-rl-agent-using-ppo


