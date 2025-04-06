import pandas as pd
import numpy as np
from typing import Dict
from .base import StrategyBase

class RSIBB(StrategyBase):
    """
    A strategy that combines the Relative Strength Index (RSI) and Bollinger Bands (BB) to generate signals.
    A buy signal is generated when RSI is below 30 (oversold condition) and the price is below the lower Bollinger Band.
    """
    def __init__(
            self, 
            price_data: pd.DataFrame, 
            rsi_period: int = 14, 
            bb_window: int = 20, 
            bb_std: float = 2.0
        ):
        """
        Initializes the RSIBB strategy.

        :param price_data: DataFrame containing historical price data.
        :param rsi_period: The period for calculating the RSI (default is 14).
        :param bb_window: The window size for calculating the Bollinger Bands (default is 20).
        :param bb_std: The number of standard deviations for the upper and lower Bollinger Bands (default is 2.0).
        """
        super().__init__(price_data)
        self.rsi_period = rsi_period
        self.bb_window = bb_window
        self.bb_std = bb_std

    def generate_signals(self) -> pd.DataFrame:
        """
        Generates buy signals based on the combination of RSI and Bollinger Bands. A signal is generated when:
        - RSI is below 30 (indicating an oversold condition).
        - The closing price is below the lower Bollinger Band.

        :return: DataFrame with calculated RSI, Bollinger Bands, signals, and positions.
        """
        df = self.price_data.copy()

        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(self.rsi_period).mean()
        avg_loss = loss.rolling(self.rsi_period).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df['BB_MA'] = df['Close'].rolling(self.bb_window).mean()
        df['BB_Upper'] = df['BB_MA'] + self.bb_std * df['Close'].rolling(self.bb_window).std()
        df['BB_Lower'] = df['BB_MA'] - self.bb_std * df['Close'].rolling(self.bb_window).std()

        df['Signal'] = np.where((df['RSI'] < 30) & (df['Close'] < df['BB_Lower']), 1, 0)
        df['Position'] = df['Signal'].diff()
        df = df.dropna(subset=['RSI', 'BB_Lower', 'BB_Upper', 'Signal'])

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
