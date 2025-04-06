import pandas as pd
import numpy as np
from typing import Dict
from .base import StrategyBase

class SMACrossover(StrategyBase):
    """
    A strategy based on the Simple Moving Average (SMA) crossover. The strategy generates buy signals
    when the short-term SMA crosses above the long-term SMA and the volatility is above a certain threshold.
    """
    def __init__(
            self,
            price_data: pd.DataFrame, 
            short_window: int = 20, 
            long_window: int = 50,
            volatility_threshold: float = 0.005
        ):
        """
        Initializes the SMACrossover strategy.

        :param price_data: DataFrame containing historical price data.
        :param short_window: The window size for the short-term SMA.
        :param long_window: The window size for the long-term SMA.
        :param volatility_threshold: The minimum volatility threshold required to trigger a signal.
        """
        super().__init__(price_data)
        self.short_window = short_window
        self.long_window = long_window
        self.volatility_threshold = volatility_threshold

    def generate_signals(self) -> pd.DataFrame:
        """
        Generates buy signals based on the crossover of the short-term and long-term SMAs, 
        along with a volatility filter. A signal is generated when the short-term SMA crosses above 
        the long-term SMA and the volatility is above the defined threshold.

        :return: DataFrame with calculated SMAs, volatility, signals, and positions.
        """
        df = self.price_data.copy()

        df['SMA_short'] = df['Close'].rolling(self.short_window).mean()
        df['SMA_long'] = df['Close'].rolling(self.long_window).mean()

        df['Volatility'] = df['Close'].pct_change().rolling(20).std()

        df['Signal'] = np.where(
            (df['SMA_short'] > df['SMA_long']) & (df['Volatility'] > self.volatility_threshold),
            1, 0
        )

        df['Position'] = df['Signal'].diff()

        return df

    def run_backtest(self) -> pd.DataFrame:
        """
        Runs the backtest based on the generated signals. It calculates returns, strategy performance, 
        and equity over time.

        :return: DataFrame with returns, strategy performance, and cumulative equity.
        """
        df = self.generate_signals()
        df['Return'] = df['Close'].pct_change()
        df['Strategy'] = df['Return'] * df['Position'].shift(1)
        df['Equity'] = (1 + df['Strategy']).cumprod()
        return df

    def get_metrics(self) -> Dict:
        """
        Calculates the key performance metrics of the strategy: Total Return, Sharpe Ratio, and 
        Maximum Drawdown.

        :return: A dictionary with 'Total Return', 'Sharpe Ratio', and 'Max Drawdown' metrics.
        """
        df = self.run_backtest()
        total_return = df['Equity'].iloc[-1] - 1
        
        if df['Strategy'].std() != 0:
            sharpe = df['Strategy'].mean() / df['Strategy'].std() * np.sqrt(252 * 24 * 60)
        else:
            sharpe = np.nan

        drawdown = (df['Equity'] / df['Equity'].cummax() - 1).min()
        return {
            "Total Return": total_return,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": drawdown
        }