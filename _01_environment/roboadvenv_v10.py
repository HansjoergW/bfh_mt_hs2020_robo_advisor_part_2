from _01_environment.universe import InvestUniverse
from _01_environment.portfolio import Portfolio

import gym
from gym.spaces import Tuple, MultiDiscrete, Box
import numpy as np
import pandas as pd
from pandas import Timestamp
from typing import List, Dict, Union
import random

# Todo
# - as an additional input the "deviation" from prediction and real potential a year ago could be used.
#   with this value the agent might be able to take a guess on how good the prediction for a certain stock tends to be
#   or in which direction it deviates

REWARD_AVERAGE_COUNT = 3


class RoboAdvisorEnvV10(gym.Env):

    def __init__(self, universe : InvestUniverse,
                    reward_average_count: int = REWARD_AVERAGE_COUNT,
                    start_cash: float = 100_000.0,
                    trading_cost: float = 40.0,
                    buy_volume: float = 5_000.0):
        super(RoboAdvisorEnvV10, self).__init__()

        self.universe = universe
        self.trading_days_ser = pd.Series(self.universe.get_trading_days())
        self.reward_average_count = reward_average_count

        self.portfolio_start_cash = start_cash
        self.portfolio_trading_cost = trading_cost
        self.portfolio_buy_volume = buy_volume

        self.step_counter = 0
        self.is_done = False
        
        # first friday in 2017
        self.start_friday = pd.to_datetime("2017-01-06")

        self.portfolio: Union[Portfolio, None] = None

        nr_of_companies = len(self.universe.get_companies())
        self.sorted_companies = list(self.universe.get_companies())
        self.sorted_companies.sort()
        self.sorted_companies_ser = pd.Series(self.sorted_companies,  name='ticker')

        # numpy array to hold the current value at every step
        # the size is large enough so that we could use a step-size of one day
        self.current_value_holder = np.zeros(len(self.trading_days_ser))

        # bounds based on the data we will prepare
        # we have 5 values per company: 'prediction', 'prediction_change', 'holding_days', 'price_change', 'portion'
        self.observation_space = Box(low=-2.0, high=3.0, shape=(nr_of_companies, 5), dtype=np.float32)

        # for every company there can be a sell, buy or do nothing action
        # https://github.com/openai/gym/blob/master/gym/spaces/multi_discrete.py
        self.action_space = MultiDiscrete([3] * len(self.universe.get_companies()))

        self.zero_state = np.zeros((nr_of_companies, 5), dtype = np.float32)

    def reset(self):
        self.step_counter = 0
        self.is_done = False
        self.portfolio = Portfolio(self.universe, self.portfolio_start_cash,
                                   self.portfolio_trading_cost, self.portfolio_buy_volume)
        self._advance_time()

        return self._calculate_state(self.current_evaluation_day)

    def step(self, actions):
        """ actions:
        0: Do nothing
        1: buy
        2: sell
        """
        if self.is_done:
            return self.zero_state, 0, self.is_done

        # actions are executed on the current trading day
        self._execute_actions(actions.tolist())

        # after that, we need to advance
        self.step_counter += 1
        self._advance_time()
        
        # and calculate the new state based on the new current evaluation day
        state = self._calculate_state(self.current_evaluation_day)

        # get the performance
        self.current_value_holder[self.step_counter] = \
            self.portfolio.get_current_evaluation(self.current_evaluation_day)

        # the reward is the average gain over the last 'reward_average_count' steps.
        # by using the average we hope to smooth it and make the training more stable
        reward = self._calculate_reward(self.step_counter, self.reward_average_count)

        self.is_done = self._calculate_is_done(self.current_value_holder[self.step_counter])

        # state, reward, done, ...
        return state, reward, self.is_done, {}

    def get_current_value(self):
        return self.current_value_holder[self.step_counter]

    def _execute_actions(self, actions: List[int]):
        action_ser = pd.Series(actions, name= 'action')
        action_pd = pd.concat([self.sorted_companies_ser, action_ser], axis= 1)
        action_pd.set_index('ticker', inplace= True)

        positions = self.portfolio.get_current_positions()

        # first: execute sell action - we can only sell shares we own
        to_sell = pd.merge(action_pd[action_pd.action == 2], positions, left_index=True, right_index=True)
        tickers_to_sell = to_sell.index.to_list()
        for ticker in tickers_to_sell:
            self.portfolio.add_sell_trade(ticker, self.current_trading_day)

        # second: execute buy action
        # how many titles can be bought based on the available cash?
        possible_buy_trades = self.portfolio.number_of_possible_buy_trades_based_on_cash()
        if possible_buy_trades == 0:  # if there is no cash left to buy .. return
            return

        tickers_to_buy = set(action_pd[action_pd.action == 1].index.to_list())

        # don't buy if already in portfolio
        tickers_to_buy = tickers_to_buy - set(positions.index.to_list())
        if len(tickers_to_buy) == 0:
            return

        # if there are more 'buy' suggestions than there is cash to buy all, we have to decide which to buy
        if len(tickers_to_buy) > possible_buy_trades:

            # we could either select completely randomly or we could for instance
            # use the prediction to narrow the possible candidates
            trading_day_predictions = self.universe.get_data_per(self.current_trading_day, ['ticker', 'prediction'])
            tickers_to_buy_predictions = trading_day_predictions[trading_day_predictions.ticker.isin(tickers_to_buy)]

            top_x_elements_to_select = min(possible_buy_trades * 3, len(tickers_to_buy))

            # sort by prediction (descending) and take the 'top_x_elements_to_select'.
            # out of them 'possible_buy_trades' are selected randomly
            buy_candidates = tickers_to_buy_predictions.sort_values('prediction', ascending=False).ticker[:top_x_elements_to_select].to_list()
            tickers_to_buy = random.choices(buy_candidates, k=possible_buy_trades)

        for ticker in tickers_to_buy:
            self.portfolio.add_buy_trade(ticker, self.current_trading_day)

    def _advance_time(self):
        current_friday = self.start_friday + pd.DateOffset(self.step_counter * 7)

        # generally, we evaluate the situation on  a Friday evening, resp. over the weekend. So we have to find the last
        # trading information which is either Friday or the last trading day before that Friday (in case of bank holidays
        self.current_evaluation_day = self.universe.find_trading_day_or_before(current_friday)

        # trades are then executed on the next trading day, which is the following monday or
        # the next trading day after, if this monday is a bank holidy
        self.current_trading_day = self.universe.find_trading_day_or_after(self.current_evaluation_day + pd.DateOffset(1))

    def _calculate_is_done(self, current_value: float):
        # done is either reached if we have reached the end of the training range
        # or if the current value is less than 20% of the start_value

        end_of_period = pd.isnull(self.current_trading_day)
        out_of_money = current_value < 0.3 * self.start_cash

        return end_of_period | out_of_money

    def _calculate_reward(self, current_step:int, reward_average_count:int) -> float:
        """ calculates the reward. it is the average profit/loss over the last 'reward_average_count' steps. """
        assert reward_average_count > 0
        if current_step < reward_average_count:
            return 0.0

        # attention: the diff array is shifted by one, so the difference if the actual step is 1 to stop 0
        # is at index 0 in the diff-array
        diff = self.current_value_holder[1:] - self.current_value_holder[:-1]
        reward = np.sum(diff[current_step - reward_average_count : current_step]) / reward_average_count
        return reward

    def _calculate_state(self, date: Timestamp):
        # state information exist on the following information for each ticker (company) in the universe
        # - the current prediction
        # - how many days ago that stock had been bought (or -1 if it isn't in the portfolio)
        # - value_change in % since bought (or 0 if not in the portfolio)
        # - prediction change in % since bought (or 0 if not in the portfolio)
        # - portion: proportion of one share in the portfolio (cash excluded)
        #
        # it would also be possible to add real values of prediction and buy price,
        # but in these case these values need to be normalized somehow. Therefore it might be better
        # to just use the change since the title had been bought for prediction and price

        # returns [ticker,prediction, close]
        current_values = self.universe.get_data_per(date, ['ticker', 'prediction', 'Close'])
        current_values.set_index('ticker', inplace=True)

        # returns index:ticker ['shares', 'buy_prediction', 'buy_price', 'buy_date']
        current_positions = self.portfolio.get_current_positions()

        merged = pd.merge(current_values, current_positions, how="left", left_index=True, right_index=True)

        # holding_days, normalised as 'years'
        merged['holding_days'] = date - merged.buy_date

        if current_positions.shape[0] > 0:
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

        # there are companies in the data which were not listed from the beginning
        # this entries will have an 'nan' as prediction (for instance DOW or CCC)
        merged.prediction.fillna(0, inplace=True)

        # only the following columns are needed as input for the agent. more over, we always have to use the same order
        merged = merged[['prediction', 'prediction_change', 'holding_days', 'price_change', 'portion']].sort_index()

        # convert it to a numpy array with float32 instead of float64, that should save some memory
        result_np = merged.to_numpy().astype(np.float32)

        # the data should be prepared in a way, that data for all companies for every day is delivered
        # in addition, no nan or null values should be present.
        assert result_np.shape[0] == len(self.universe.get_companies())
        assert np.count_nonzero(np.isnan(result_np)) == 0

        return result_np.reshape(-1)



