from _01_environment.universe import InvestUniverse
from _01_environment.portfolio_v10 import Portfolio, TradeType

import pandas as pd

universe = InvestUniverse()


def test_init():
    portfolio = Portfolio(universe, 10000)

    assert portfolio is not None

def test_add_trade():
    portfolio = Portfolio(universe, 10000)

    buy_date = pd.to_datetime("2017-01-03")
    buy_ticker = "AAPL"
    portfolio.add_buy_trade(buy_ticker, buy_date)

    assert portfolio.current_cash == 5000.0

    curpos_pd = portfolio.get_current_positions()
    assert curpos_pd.shape[0] == 1
    assert curpos_pd.loc[buy_ticker] > 0

    buy2_date = pd.to_datetime("2017-01-03")
    buy2_ticker = "MSFT"
    portfolio.add_buy_trade(buy2_ticker, buy2_date)

    assert portfolio.current_cash == 0.0

    curpos_pd = portfolio.get_current_positions()
    assert curpos_pd.shape[0] == 2
    assert curpos_pd.loc[buy_ticker] > 0
    assert curpos_pd.loc[buy2_ticker] > 0


