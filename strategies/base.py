from abc import ABC, abstractmethod
import pandas as pd

class StrategyBase(ABC):
    def __init__(self, price_data: pd.DataFrame):
        """
        Initializes the strategy with price data.

        :param price_data: DataFrame containing historical price data
        """
        self.price_data = price_data

    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """
        Generates trading signals based on the provided price data.

        :return: DataFrame containing generated signals
        """
        pass

    @abstractmethod
    def run_backtest(self) -> pd.DataFrame:
        """
        Runs the backtest on the strategy.

        :return: DataFrame containing backtest results
        """
        pass

    @abstractmethod
    def get_metrics(self) -> dict:
        """
        Retrieves the performance metrics of the strategy.

        :return: Dictionary containing performance metrics (e.g., Sharpe ratio, drawdown)
        """
        pass
