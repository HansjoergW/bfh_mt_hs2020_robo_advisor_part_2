import pandas as pd
from pandas import Timestamp
from typing import List

ROBO_ADVISOR_DATA_FILE = "D:/data_mt/09_training/robo_train_set.csv"

# Attention, there are companies which don't have data right from the beginning
#

class Universum():

    def __init__(self):

        self.data = Universum._load_data()

        self.companies = self.data.ticker.unique().tolist()
        self.trading_days = self.data.index.unique().tolist()

    def get_companies(self) -> List[str]:
        return self.companies

    def get_trading_days(self) -> List[Timestamp]:
        return self.trading_days

    @staticmethod
    def _load_data():
        df = pd.read_csv(ROBO_ADVISOR_DATA_FILE, sep=',', encoding='utf-8', header=0)
        df['Date'] = pd.to_datetime(df.Date)
        df['day_of_week'] = df.Date.dt.dayofweek
        df['mid_price'] = (df.High + df.Low) / 2

        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        return df