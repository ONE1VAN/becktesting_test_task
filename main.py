from core.data_loader import DataLoader
from strategies.sma_cross import SMACrossover
import matplotlib.pyplot as plt

if __name__ == "__main__":
    loader = DataLoader(
        data_dir="data",
        interval='1m',
        date_start='2025-02-01',
        date_end='2025-02-28',
        top_n=2
    )

    loader.download_data()
    data_df = loader.load_parquet('btc_1m_20250201.parquet')
    
    data = SMACrossover(
        price_data=data_df,
        symbol='ETHBTC',
        short_window=50,
        long_window=200,
        volatility_threshold=0.005
    )

    strategy_results = data.run_backtest()

    plt.figure(figsize=(12, 8))
    plt.plot(strategy_results['Close'], label='ETHBTC Close Price')
    plt.plot(strategy_results['SMA50'], label='50-day SMA')
    plt.plot(strategy_results['SMA200'], label='200-day SMA')
    plt.plot(strategy_results[strategy_results['Position'] == 1].index, strategy_results['SMA50'][strategy_results['Position'] == 1], '^', markersize=10, color='g', label='Buy Signal')
    plt.plot(strategy_results[strategy_results['Position'] == -1].index, strategy_results['SMA50'][strategy_results['Position'] == -1], 'v', markersize=10, color='r', label='Sell Signal')
    plt.title('ETHBTC Simple Moving Average Crossover Strategy')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()