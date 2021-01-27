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
#   with this value the agent might be able to take a guess on how good the prediction for a stock tends to be


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
        #
        # it would also be possible to add real values of prediction and buy price,
        # but in these case these values need to be normalized somehow. Therefore it might be better
        # to just use change for prediction and price

        # returns [ticker,prediction, close]
        current_values = self.universe.get_data_per(date, ['ticker', 'prediction', 'Close'])
        current_values.set_index('ticker', inplace=True)

        # returns index = ['shares', 'buy_prediction', 'buy_price', 'buy_date']
        current_positions = self.portfolio.get_positions(date)

        merged = pd.merge(current_values, current_positions, how="left", left_index=True, right_index=True)

        # holding_days, normalised as years
        merged['holding_days'] = date - merged.buy_date
        merged['holding_days'] = merged[~merged.holding_days.isna()].holding_days.dt.days / 365

        merged['price_change'] = (merged.Close - merged.buy_price) / merged.buy_price
        merged['prediction_change'] = (merged.prediction - merged.buy_prediction) / merged.buy_prediction

        with_shares = merged[~merged.shares.isna()]
        total_invest = (with_shares.shares * with_shares.Close).sum()
        merged.loc[:, 'portion'] = with_shares.shares * with_shares.Close / total_invest

        merged.holding_days.fillna(-1, inplace=True)
        merged.price_change.fillna(0, inplace=True)
        merged.prediction_change.fillna(0, inplace=True)
        merged.portion.fillna(0, inplace=True)




        print("")

        # we normalisieren? muss nicht perfekt sein
        # https://stats.stackexchange.com/questions/347623/using-non-normalized-data-for-learning-a-rl-agent-using-ppo


