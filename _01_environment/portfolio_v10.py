# manages the portfolio

from _01_environment.universe import InvestUniverse
import pandas as pd
import numpy as np
from pandas import Timestamp
from typing import List, Dict, Union
from enum import Enum


class TradeType(Enum):
    BUY = 1
    SELL = 2


class Portfolio():

    def __init__(self, universe: InvestUniverse, cash: float, trading_cost: float = 40.0, buy_volume: float = 5000.0):
        self.universe = universe
        self.start_cash = float(cash)
        self.current_cash = float(cash)
        self.trading_cost = trading_cost
        self.buy_volume = buy_volume

        self.trading_book: List[Dict] = []
        self.cash_book: List[Dict] = []

        start_date = self.universe.get_trading_days()[0]
        self.cash_book.append({"date": start_date, "amount": cash, "what": "cash_start"})

    def add_buy_trade(self, ticker: str, date: Timestamp):
        """ adds a buy trade to the book. """
        data_dict = self.universe.get_data(ticker, date)

        shares = (self.buy_volume - self.trading_cost) / data_dict['mid_price']

        entry_dict = {
            'type': TradeType.BUY,
            'ticker': ticker,
            'date': date,
            'cost': self.trading_cost,
            'price': data_dict['mid_price'],
            'potential': data_dict['r_potential'],
            'prediction': data_dict['prediction'],
            'shares': shares,
            'amount': -shares * data_dict['mid_price']
        }

        self.trading_book.append(entry_dict)
        self._recalculate_cash(entry_dict)

    def add_sell_trade(self, ticker: str, date: Timestamp):
        """ adds a sell trade to the book. """
        data_dict = self.universe.get_data(ticker, date)

        # we always sell the whole position
        shares = self.get_positions().loc[ticker]

        entry_dict = {
            'type': TradeType.SELL,
            'ticker': ticker,
            'date': date,
            'cost': self.trading_cost,
            'price': data_dict['mid_price'],
            'potential': data_dict['r_potential'],
            'prediction': data_dict['prediction'],
            'shares': -shares,
            'amount': shares * data_dict['mid_price']
        }

        self.trading_book.append(entry_dict)
        self._recalculate_cash(entry_dict)

    def _recalculate_cash(self, trade: Dict):
        """ recalculates the current cash position after the trade. Also adds the appropriate entries in the cash book. """

        # since selling contains negative shares, we can use the same line
        # to calculate the change in cash for buy and for sell
        self.current_cash -= (trade['shares'] * trade['price']) + trade['cost']

        self.cash_book.append({
            "what": trade['ticker'],
            "date": trade['date'],
            "amount": -(trade['shares'] * trade['price'])
        })
        self.cash_book.append({
            "what": "cost",
            "date": trade['date'],
            "amount": -trade['cost']
        })

    def get_positions(self, date: Timestamp = pd.to_datetime("2100-01-01")) -> pd.DataFrame :
        """ returns the current positions at a specific date"""

        book_pd = pd.DataFrame(self.trading_book)
        book_pd = book_pd[book_pd.date <= date]
        positions = book_pd.groupby('ticker')['shares'].sum()
        return positions[positions > 0]

    def get_evaluation(self, date: Timestamp = pd.to_datetime("2100-01-01")) -> float:
        """
        returns the value at a specific date.
        get_evaluation is always at the end of day.
         if the provided date isn't a trading day, the last trading day before that date is taken.
        """

        trading_day = self.universe.find_trading_day_or_before(date)

        book_pd = pd.DataFrame(self.trading_book)
        book_pd = book_pd[book_pd.date <= trading_day]

        # value = start_cash - all transaction costs - all buys + all sells + current position value
        cost_all_transactions = book_pd.cost.sum().item()
        all_buys_and_sells = book_pd.amount.sum().item()

        positions = self.get_positions(trading_day)
        tickers = positions.index.to_list()

        current_value = 0.0
        if len(tickers):
            close_values = self.universe.get_close_for_per(tickers, trading_day)
            close_values.reset_index(drop=True, inplace=True)

            merged = pd.merge(positions, close_values, how="outer", on="ticker")

            assert merged.isna().sum().sum() == 0

            current_value = (merged.Close * merged.shares).sum().item()

        value = self.start_cash - cost_all_transactions + all_buys_and_sells + current_value

        return value

    def get_cash_flow(self) -> pd.DataFrame :
        """ returns the cash flow ober the whole period."""
        cash_book_pd = pd.DataFrame(self.cash_book)
        cash_book_day_pd = cash_book_pd[['date','amount']].groupby(['date']).sum()

        trading_days = pd.Series(self.universe.get_trading_days()).to_frame()
        trading_days.columns = ['date']
        merged = pd.merge(trading_days, cash_book_day_pd, how="outer", on="date")
        merged.replace(np.NaN, 0, inplace=True)
        merged['i_date'] = merged.date
        merged.set_index('i_date', inplace=True)
        merged.sort_index(inplace=True)
        merged['cash_current'] = merged.amount.cumsum()

        return merged

    def get_portfolio_flow_per_title(self) -> pd.DataFrame:
        """ returns the portfolio flow on ticker basis.
            so contains an entry for every ticker and date where stocks of that company were held.
        """
        book_pd = pd.DataFrame(self.trading_book)

        tickers = book_pd.ticker.unique().tolist()

        close_prices = self.universe.get_close_for_tickers(tickers)
        close_prices.rename(columns={'Date':'date'}, inplace=True)

        # merge the date from the trading book with the close prices
        merged = pd.merge(close_prices, book_pd[['date','ticker','shares']], how="left", on=['date','ticker'])
        merged.rename(columns={'shares':'shares_chg'}, inplace=True)
        merged.replace(np.NaN, 0, inplace=True)

        merged['i_date'] = merged.date
        merged.set_index('i_date', inplace=True)
        merged.sort_index(inplace=True)

        merged['shares_current'] = merged.groupby("ticker").shares_chg.cumsum()
        merged['value_current'] = merged.shares_current * merged.Close

        return merged[['ticker', 'shares_current','value_current']]

    def get_portfolio_flow(self) -> pd.DataFrame:
        """
        Summarizes the whole portfolio flow which consist of cash and stock positions for the whole period.
        """

        cash_flow_pd = self.get_cash_flow()

        portfolio_flow_per_title_pd = self.get_portfolio_flow_per_title()
        portfolio_flow_pd = portfolio_flow_per_title_pd['value_current'].groupby('i_date').sum().to_frame()
        portfolio_flow_pd.columns = ['value_current']

        merged = pd.merge(portfolio_flow_pd, cash_flow_pd[['cash_current']], how="outer", left_index=True, right_index=True)
        merged['total_current'] = merged.value_current + merged.cash_current
        return merged







        # und den Verlauf des Vermögens basierend auf den TradingDays
