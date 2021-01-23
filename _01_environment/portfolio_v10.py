# manages the portfolio

from _01_environment.universum import Universum
import pandas as pd
from pandas import Timestamp
from typing import List, Dict


class Portfolio():

    def __init__(self, universe:Universum, cash: float, trading_cost: float = 40, buy_volume: float = 5000):
        self.current_cash = cash
        self.trading_cost = trading_cost
        self.buy_volume = buy_volume

        self.trading_book:List[Dict] = []


    def add_trade(self, ticker:str, date:Timestamp):



        pass