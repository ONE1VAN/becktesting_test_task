import pandas as pd
import numpy as np
from typing import Dict
from .base import StrategyBase
import vectorbt as vbt

class SMACrossover(StrategyBase):
    """
    A strategy based on the Simple Moving Average (SMA) crossover. The strategy generates buy signals
    when the short-term SMA crosses above the long-term SMA and the volatility is above a certain threshold.
    """
    def __init__(
            self,
            price_data: pd.DataFrame, 
            symbol: str = "ETHBTC",
            short_window: int = 50, 
            long_window: int = 200,
            volatility_window: int = 20,
            volatility_threshold: float = 0.005
        ):
        if 'symbol' in price_data.columns:
            price_data = price_data[price_data['symbol'] == symbol].copy()
        if price_data.empty:
            raise ValueError(f"No data found for symbol: {symbol}")
        """
        Initializes the SMACrossover strategy.

        :param price_data: DataFrame containing historical price data.
        :param symbol: The trading pair symbol (default is "ETHBTC").
        :param short_window: The window size for the short-term SMA.
        :param long_window: The window size for the long-term SMA.
        :param volatility_window: The window size for calculating volatility.
        :param volatility_threshold: The minimum volatility threshold required to trigger a signal.
        """
        super().__init__(price_data)
        self.short_window = short_window
        self.long_window = long_window
        self.volatility_window = volatility_window
        self.volatility_threshold = volatility_threshold
        self.symbol = symbol

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

        df['Volatility'] = df['Close'].pct_change().rolling(self.volatility_window).std()

        df['Signal'] = np.where(
            (df['SMA_short'] > df['SMA_long']) & (df['Volatility'] > self.volatility_threshold),
            1, 0
        )

        df['Position'] = df['Signal'].diff()

        return df

    def run_backtest(self) -> pd.DataFrame:
        """
         Run the backtest using vectorbt based on the strategy's generated signals.

        Parameters:
            cash (int): Initial capital to start the backtest with. Default is 10,000.

        Returns:
            vbt.Portfolio: A vectorbt Portfolio object containing performance metrics, trades, equity curve, and more.
        """
        df = self.generate_signals()
        
        portfolio = vbt.Portfolio.from_signals(
            df['Close'],  
            df['Signal'], 
            df['Signal'].shift(-1),  
            freq='1T',  
        )

        return portfolio
    
    def get_metrics(self, cash: int = 10_000) -> Dict[str, float]:
        pass