from _01_environment.universe import InvestUniverse
from _01_environment.portfolio_v10 import Portfolio, TradeType

import pandas as pd

universe = InvestUniverse()


def test_init():
    portfolio = Portfolio(universe, 10000)

    assert portfolio is not None


def test_buy_trade():
    portfolio = Portfolio(universe, 10000)

    buy_date = pd.to_datetime("2017-01-03")
    buy_ticker = "AAPL"
    portfolio.add_buy_trade(buy_ticker, buy_date)

    assert portfolio.current_cash == 5000.0

    curpos_pd = portfolio.get_positions()
    assert curpos_pd.shape[0] == 1
    assert curpos_pd.loc[buy_ticker] > 0

    buy2_date = pd.to_datetime("2017-01-03")
    buy2_ticker = "MSFT"
    portfolio.add_buy_trade(buy2_ticker, buy2_date)

    assert portfolio.current_cash == 0.0

    curpos_pd = portfolio.get_positions()

    assert curpos_pd.shape[0] == 2
    assert curpos_pd.loc[buy_ticker] > 0
    assert curpos_pd.loc[buy2_ticker] > 0

    evaluate_date = pd.to_datetime("2017-01-31")
    value = portfolio.get_evaluation(evaluate_date)

    assert int(round(value)) == 10341

    portfolio_flow = portfolio.get_portfolio_flow()
    assert int(round(portfolio_flow.loc[evaluate_date].total_current)) == 10341



def test_buy_and_sell_trade():
    portfolio = Portfolio(universe, 10000)

    buy_date = pd.to_datetime("2017-01-03")
    ticker = "AAPL"
    portfolio.add_buy_trade(ticker, buy_date)

    assert portfolio.current_cash == 5000.0

    sell_date = pd.to_datetime("2018-01-03")
    portfolio.add_sell_trade(ticker, sell_date)

    curpos_pd = portfolio.get_positions()
    assert curpos_pd.shape[0] == 0

    value = portfolio.get_evaluation(sell_date)
    assert int(round(value)) == 12517

    portfolio_flow = portfolio.get_portfolio_flow()
    assert int(round(portfolio_flow.loc[sell_date].total_current)) == 12517


