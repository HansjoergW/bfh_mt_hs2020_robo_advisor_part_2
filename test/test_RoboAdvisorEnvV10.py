from _01_environment.roboadvenv_v10 import RoboAdvisorEnvV10
from _01_environment.universe import InvestUniverse

import pandas as pd
import numpy as np

universe = InvestUniverse()


def test_init():
    env = RoboAdvisorEnvV10(universe)
    result = env.reset()
    assert result[0] == 0.0


def test_init_days():
    env = RoboAdvisorEnvV10(universe)

    assert env.current_evaluation_day == pd.to_datetime("2017-01-03")
    assert env.current_trading_day == pd.to_datetime("2017-01-04")


def test_advance():
    env = RoboAdvisorEnvV10(universe)
    env.step_counter += 1

    env._advance_time()
    assert env.current_evaluation_day == pd.to_datetime("2017-01-09")
    assert env.current_trading_day == pd.to_datetime("2017-01-10")

    # move to another date which is not a monday
    env.step_counter = 51
    env._advance_time()
    assert env.current_evaluation_day == pd.to_datetime("2017-12-26")
    assert env.current_trading_day == pd.to_datetime("2017-12-27")


def test_calculate_state():
    env = RoboAdvisorEnvV10(universe)
    env.reset()

    buy_date = pd.to_datetime("2017-01-03")
    buy_ticker = "AAPL"
    env.portfolio.add_buy_trade(buy_ticker, buy_date)

    buy2_ticker = "MSFT"
    env.portfolio.add_buy_trade(buy2_ticker, buy_date)

    calc_date = pd.to_datetime("2017-01-06")
    env._calculate_state(calc_date)
    print("")


def test_calculate_reward():
    env = RoboAdvisorEnvV10(universe)
    env.current_value_holder = np.arange(10)

    assert env._calculate_reward(0, 1) == 0.0
    assert env._calculate_reward(1, 1) == 1.0
    assert env._calculate_reward(2, 1) == 1.0

    assert env._calculate_reward(0, 2) == 0.0
    assert env._calculate_reward(1, 2) == 0.0
    assert env._calculate_reward(2, 2) == 1.0
    assert env._calculate_reward(3, 2) == 1.0

    env.current_value_holder = np.ones(10)
    assert env._calculate_reward(0, 1) == 0.0
    assert env._calculate_reward(1, 1) == 0.0
    assert env._calculate_reward(2, 1) == 0.0

    env.current_value_holder = np.array([0, 0, 0, 1, 0, 0, 1])
    assert env._calculate_reward(0, 1) == 0.0
    assert env._calculate_reward(1, 1) == 0.0
    assert env._calculate_reward(2, 1) == 0.0
    assert env._calculate_reward(3, 1) == 1.0
    assert env._calculate_reward(1, 2) == 0.0
    assert env._calculate_reward(2, 2) == 0.0
    assert env._calculate_reward(3, 2) == 0.5


def test_execute_actions():
    env = RoboAdvisorEnvV10(universe)
    env.reset()

    zero_list = [0] * len(universe.get_companies())

    env._execute_actions(zero_list)
    assert len(env.portfolio.trading_book) == 0
    assert len(env.portfolio.cash_book) == 1

    buy_list = zero_list.copy()
    buy_list[0:2] = [1,1]

    env._execute_actions(buy_list)
    assert len(env.portfolio.trading_book) == 2
    assert len(env.portfolio.cash_book) == 5

    # next, we only have cash to buy one
    # moreover, 'buy' suggestions include 2 tickers that were bought in the prior step
    env.portfolio.current_cash = 5000.0
    buy_list = zero_list.copy()
    buy_list[0:4] = [1, 1, 1, 1]
    # so we expect that only one candidate will be bought
    env._execute_actions(buy_list)

    # an addtional buy trade has to be added
    assert len(env.portfolio.trading_book) == 3
    assert len(env.portfolio.cash_book) == 7


    # next, we only have cash to buy one
    # moreover, 'buy' suggestions include 2 tickers that were bought in the prior step
    env.portfolio.current_cash = 10000.0
    buy_list = zero_list.copy()
    buy_list[0:30] = [1] * 30

    # so we expect that only two out of 30 candidate will be bought
    env._execute_actions(buy_list)

    # two addtional buy trades have to be added
    assert len(env.portfolio.trading_book) == 5
    assert len(env.portfolio.cash_book) == 11


    # next we set cash to zero and sell a few titles
    env._advance_time()
    env.portfolio.current_cash = 0.0

    # we know that we bought the first to titles and one out of third and fourth
    # so we 'sell' all the first four titles which should result in 3 sold titles
    # with the new money in the cashbook, we again try to buy stocks
    sell_buy_list = zero_list.copy()
    sell_buy_list[0:30] = [1] * 30
    sell_buy_list[0:4] = [2] * 4

    # so we expect that only two out of 30 candidate will be bought
    env._execute_actions(sell_buy_list)

    # two addtional buy trades have to be added
    assert len(env.portfolio.trading_book) > 8
    assert len(env.portfolio.cash_book) > 17


