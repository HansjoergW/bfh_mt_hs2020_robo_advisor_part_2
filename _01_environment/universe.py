import pandas as pd
from pandas import Timestamp
from typing import List, Dict

ROBO_ADVISOR_DATA_FILE = "D:/data_mt/09_training/robo_train_set.csv"

# Attention, there are companies which don't have data right from the beginning
#

class InvestUniverse():

    """
    represents the investment universe. loads all the data of all titles.
    offers several convenient methods to access the data.
    """

    def __init__(self):
        self.data = InvestUniverse._load_data()

        self.companies = self.data.ticker.unique().tolist()
        self.trading_days = pd.Series(self.data.index.unique().tolist())

    def get_companies(self) -> List[str]:
        """ returns a list with all the ticker symbols in the data"""
        return self.companies

    def get_trading_days(self) -> pd.Series:
        """ returns a pandas Series with all trading days"""
        return self.trading_days

    def get_data(self, ticker: str, date: Timestamp) -> Dict:
        """ returns the information of ticker on the provided date as dictionary."""
        return self.data[self.data.ticker == ticker].loc[date].to_dict()

    def find_trading_day_or_after(self, date: Timestamp):
        """ searches the next trading day if the provided date is not a trading day. """
        return self.trading_days[self.trading_days >= date].min()

    def find_trading_day_or_before(self, date: Timestamp):
        """ searches the last trading day before the provided date if the date is not a trading day."""
        return self.trading_days[self.trading_days <= date].max()

    def get_close_for_per(self, tickers: List[str], date: Timestamp) -> pd.DataFrame:
        """ get the close prices for all provided tickers for a specific date. """
        return self.data[self.data.ticker.isin(tickers)].loc[date][['ticker','Close']].copy()

    def get_close_for_tickers(self, tickers: List[str]) -> pd.DataFrame:
        """ gets the closing prices for all tickers for all trading days in the data. """
        return self.data[self.data.ticker.isin(tickers)][['Date','ticker','Close']].reset_index(drop=True)

    def get_predictions_per(self, date: Timestamp) -> pd.DataFrame:
        return self.data.loc[date][['ticker', 'prediction']].copy().reset_index(drop=True)


    @staticmethod
    def _load_data():
        df = pd.read_csv(ROBO_ADVISOR_DATA_FILE, sep=',', encoding='utf-8', header=0)
        df['Date'] = pd.to_datetime(df.Date)
        df['day_of_week'] = df.Date.dt.dayofweek
        df['mid_price'] = (df.High + df.Low) / 2

        df['i_date'] = df.Date
        df.set_index('i_date', inplace=True)
        df.sort_index(inplace=True)

        return df