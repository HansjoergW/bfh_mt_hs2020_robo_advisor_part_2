from _01_environment.universe import InvestUniverse
from _01_environment.portfolio import Portfolio, TradeType

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
    assert curpos_pd.loc[buy_ticker].shares > 0

    buy2_date = pd.to_datetime("2017-01-03")
    buy2_ticker = "MSFT"
    portfolio.add_buy_trade(buy2_ticker, buy2_date)

    assert portfolio.current_cash == 0.0

    curpos_pd = portfolio.get_positions()

    assert curpos_pd.shape[0] == 2
    assert curpos_pd.loc[buy_ticker].shares > 0
    assert curpos_pd.loc[buy2_ticker].shares > 0

    evaluate_date = pd.to_datetime("2017-01-31")
    value = portfolio.get_evaluation(evaluate_date)
    value_cur = portfolio.get_current_evaluation(evaluate_date)

    assert int(round(value)) == 10341
    assert int(round(value_cur)) == 10341

    portfolio_flow = portfolio.get_total_flow()
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

    portfolio_flow = portfolio.get_total_flow()
    assert int(round(portfolio_flow.loc[sell_date].total_current)) == 12517


def test_on_empty():
    portfolio = Portfolio(universe, 10_000)

    positions = portfolio.get_positions()
    assert positions.shape[0] == 0

    evaluation = portfolio.get_evaluation()
    assert evaluation == 10_000

    cash_flow = portfolio.get_cash_flow()
    assert cash_flow.shape[0] == universe.get_trading_days().shape[0]
    assert (cash_flow.cash_current == 10000).all()

    portfolio_flow_per_title = portfolio.get_portfolio_flow_per_title()
    assert portfolio_flow_per_title.shape[0] == 0

    portfolio_flow = portfolio.get_total_flow()
    assert portfolio_flow.shape[0] == universe.get_trading_days().shape[0]
    assert (portfolio_flow.total_current == 10000).all()


def test_number_of_possible_buy_trades():
    portfolio = Portfolio(universe, 9_999)
    assert portfolio.number_of_possible_buy_trades_based_on_cash() == 1

    portfolio = Portfolio(universe, 2_500)
    assert portfolio.number_of_possible_buy_trades_based_on_cash() == 0

    portfolio = Portfolio(universe, 100_000)
    assert portfolio.number_of_possible_buy_trades_based_on_cash() == 20




