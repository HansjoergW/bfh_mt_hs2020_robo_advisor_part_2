# manages the portfolio

from _01_environment.universe import InvestUniverse
import pandas as pd
from pandas import Timestamp
from typing import List, Dict
from enum import Enum


class TradeType(Enum):
    BUY = 1
    SELL = 2


class Portfolio():

    def __init__(self, universe: InvestUniverse, cash: float, trading_cost: float = 40, buy_volume: float = 5000):
        self.universe = universe
        self.current_cash = cash
        self.trading_cost = trading_cost
        self.buy_volume = buy_volume

        self.trading_book: List[Dict] = []

    def add_buy_trade(self, ticker: str, date: Timestamp):
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
        }

        self.trading_book.append(entry_dict)
        self._recalculate_cash(entry_dict)

    def add_sell_trade(self, ticker: str, date: Timestamp):
        data_dict = self.universe.get_data(ticker, date)

        # we always sell the whole position
        shares = self.get_current_positions().loc[ticker]

        entry_dict = {
            'type': TradeType.SELL,
            'ticker': ticker,
            'date': date,
            'cost': self.trading_cost,
            'price': data_dict['mid_price'],
            'potential': data_dict['r_potential'],
            'prediction': data_dict['prediction'],
            'shares': -shares,
        }

        self.trading_book.append(entry_dict)
        self._recalculate_cash(entry_dict)

    def _recalculate_cash(self, trade: Dict):
        # since selling contains negative shares, we can use the same line
        # to calculate the change in cash for buy and for sell
        self.current_cash -= (trade['shares'] * trade['price']) + trade['cost']

    def get_current_positions(self):
        book_pd = pd.DataFrame(self.trading_book)
        return book_pd.groupby('ticker')['shares'].sum()

    def get_get_evaluation_for(self, date: Timestamp):
        pass

        # next step total value for current day
    # und den Verlauf des Vermögens basierend auf den TradingDays
