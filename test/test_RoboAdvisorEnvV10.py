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

