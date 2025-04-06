import pandas as pd
import numpy as np
from typing import Dict
from .base import StrategyBase

class VWAPReversion(StrategyBase):
    """
    A strategy based on VWAP (Volume Weighted Average Price) reversion. It generates buy signals
    when the deviation between the VWAP and the closing price exceeds a given threshold.
    """
    def __init__(
            self, 
            price_data: pd.DataFrame, 
            threshold: float = 0.01, 
            vwap_window: int = 20):
        """
        Initializes the VWAP Reversion strategy.

        :param price_data: DataFrame containing historical price and volume data.
        :param threshold: The deviation threshold from VWAP to trigger a signal.
        :param vwap_window: The window size used to calculate VWAP.
        """
        super().__init__(price_data)
        self.threshold = threshold
        self.vwap_window = vwap_window  

    def generate_signals(self) -> pd.DataFrame:
        """
        Generates trading signals based on VWAP reversion strategy. A buy signal (1) is generated
        when the deviation between the VWAP and closing price exceeds the given threshold.

        :return: DataFrame with calculated VWAP, deviation, signals, and positions.
        """
        df = self.price_data.copy()

        df['CumVolume'] = df['Volume'].rolling(self.vwap_window).sum()
        df['CumPV'] = (df['Close'] * df['Volume']).rolling(self.vwap_window).sum()
        df['VWAP'] = df['CumPV'] / df['CumVolume']

        df['Deviation'] = (df['VWAP'] - df['Close']) / df['VWAP']
        df['Signal'] = np.where(df['Deviation'] > self.threshold, 1, 0)
        df['Position'] = df['Signal'].diff()

        return df

    def run_backtest(self) -> pd.DataFrame:
        """
        Runs a backtest based on the generated signals. It calculates the strategy's returns and 
        cumulative equity.

        :return: DataFrame with the strategy's performance, including returns and equity.
        """
        df = self.generate_signals()
        df['Return'] = df['Close'].pct_change()
        df['Strategy'] = df['Return'] * df['Position'].shift(1)
        df['Strategy'] = df['Strategy'].fillna(0) 
        df['Equity'] = (1 + df['Strategy']).cumprod()

        return df

    def get_metrics(self) -> Dict:
        """
        Calculates key performance metrics for the strategy: Total Return, Sharpe Ratio, and 
        Maximum Drawdown.

        :return: A dictionary with Total Return, Sharpe Ratio, and Maximum Drawdown metrics.
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