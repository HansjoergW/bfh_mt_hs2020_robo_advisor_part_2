from _01_environment.roboadvenv_v10 import RoboAdvisorEnvV10
from _01_environment.universum import Universum

import pandas as pd

universe = Universum()


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