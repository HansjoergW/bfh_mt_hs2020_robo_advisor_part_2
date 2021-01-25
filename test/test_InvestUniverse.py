from _01_environment.universe import InvestUniverse

import pandas as pd


def test_init():
    universe = InvestUniverse()

    assert len(universe.get_companies()) > 100
    assert len(universe.get_trading_days()) > 100


def test_get_data():
    universe = InvestUniverse()
    a_date = pd.to_datetime('2017-01-04')
    ticker = 'AAPL'

    data_dict = universe.get_data(ticker, a_date)

    expected = {'Date', 'ticker', 'r_potential', 'prediction', 'Close', 'High', 'Low', 'Open', 'close_norm',
                'day_of_week', 'mid_price'}
    assert len(expected - set(data_dict.keys())) == 0


def test_find_trading_day_or_after():
    universe = InvestUniverse()
    a_date = pd.to_datetime('2017-01-01')

    result = universe.find_trading_day_or_after(a_date)
    assert result == pd.to_datetime("2017-01-03")

    a_date = pd.to_datetime('2017-01-04')
    result = universe.find_trading_day_or_after(a_date)
    assert result == pd.to_datetime("2017-01-04")


def test_find_trading_day_or_before():
    universe = InvestUniverse()
    a_date = pd.to_datetime('2017-01-08')

    result = universe.find_trading_day_or_before(a_date)
    assert result == pd.to_datetime("2017-01-06")

    a_date = pd.to_datetime('2017-01-04')
    result = universe.find_trading_day_or_before(a_date)
    assert result == pd.to_datetime("2017-01-04")