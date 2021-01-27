from _01_environment.universe import InvestUniverse

import pandas as pd

universe = InvestUniverse()


def test_init():
    assert len(universe.get_companies()) > 100
    assert len(universe.get_trading_days()) > 100


def test_get_data():
    a_date = pd.to_datetime('2017-01-04')
    ticker = 'AAPL'

    data_dict = universe.get_data(ticker, a_date)

    expected = {'Date', 'ticker', 'r_potential', 'prediction', 'Close', 'High', 'Low', 'Open', 'close_norm',
                'day_of_week', 'mid_price'}
    assert len(expected - set(data_dict.keys())) == 0


def test_find_trading_day_or_after():
    a_date = pd.to_datetime('2017-01-01')

    result = universe.find_trading_day_or_after(a_date)
    assert result == pd.to_datetime("2017-01-03")

    a_date = pd.to_datetime('2017-01-04')
    result = universe.find_trading_day_or_after(a_date)
    assert result == pd.to_datetime("2017-01-04")


def test_find_trading_day_or_before():
    a_date = pd.to_datetime('2017-01-08')

    result = universe.find_trading_day_or_before(a_date)
    assert result == pd.to_datetime("2017-01-06")

    a_date = pd.to_datetime('2017-01-04')
    result = universe.find_trading_day_or_before(a_date)
    assert result == pd.to_datetime("2017-01-04")


def test_get_prediction():
    a_date = pd.to_datetime('2017-01-04')
    trading_date = universe.find_trading_day_or_before(a_date)

    assert 400 < universe.get_data_per(trading_date, ['ticker', 'prediction', 'close']).shape[0]
