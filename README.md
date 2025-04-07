# 📈 Backtesting Test Task

This project is a modular cryptocurrency trading backtester built using VectorBT. It supports custom trading strategies, efficient data handling, and performance analysis with key metrics and visualizations.

## 🚀 Features

- Loads 1-minute historical data from Binance Data Vision
- Filters the top 100 most liquid BTC pairs for a given time range
- Saves all data to a compressed `.parquet` file for fast access
- Implements multiple trading strategies (e.g., SMA Crossover, RSI + Bollinger Bands, VWAP Reversion Intraday)
- Performs portfolio-level backtesting across all pairs simultaneously

## 🧠 Strategies

Implemented strategies inherit from a common `StrategyBase` and include:

- **SMA Crossover**  
  Buy signal when short-term SMA crosses above long-term SMA, with volatility filter.

- **RSI + Bollinger Bands (RSIBB)**  
  Combines RSI and Bollinger Bands to detect overbought/oversold levels.

- **VWAP Reversion Intraday** *(coming soon)*  
  Looks for price deviation from intraday VWAP for mean-reversion entries.



