import vectorbt as vbt
import pandas as pd
from pathlib import Path
from binance.client import Client
import os
import time

class DataLoader:
    def __init__(
            self, 
            data_dir="data", 
            interval='1m', 
            date_start='2025-02-02', 
            date_end='2025-03-01',
            top_n=100
        ):
        self.client = Client()
        self.data_dir = Path(data_dir)
        self.interval = interval
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.date_start = date_start
        self.date_end = date_end
        self.top_n = top_n
        self.top_btc_pairs = []

    def get_top_liquid_pairs(self) -> list[str]:
        """Receive n number of most liquid pairs with BTC for last 24h."""
        tickers = self.client.get_ticker()  # 24-hour price change statistics
        btc_pairs = [t for t in tickers if t['symbol'].endswith('BTC')]
        btc_pairs_sorted = sorted(btc_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
        self.top_btc_pairs = [p['symbol'] for p in btc_pairs_sorted[:self.top_n]]
        print(f"Top {self.top_n} pairs: {self.top_btc_pairs[:5]}...")
        return self.top_btc_pairs

    def is_data_valid(self, path: Path) -> bool:
        """Check if a parquet file exists and contains valid OHLCV data."""
        try:
            if not path.exists():
                return False
            df = pd.read_parquet(path)
            required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
            if df.empty or not required_cols.issubset(df.columns):
                return False
            return True
        except Exception as e:
            print(f"Validation failed for {path.name}: {e}")
            return False

    def download_data(self, refresh: bool = False) -> None:
        """Download historical 1-minute OHLCV data for the top BTC pairs
        If refresh is True, delete all cached files and reload from scratch."""
        if not self.top_btc_pairs:
            print("Fetching top BTC pairs...")
            self.get_top_liquid_pairs()

        combined_path = self.data_dir / f"btc_{self.interval}_{self.date_start.replace('-', '')}.parquet"
        
        if refresh:
            print("Refreshing: removing all cached parquet files...")
            for file in self.data_dir.glob("*.parquet"):
                try:
                    file.unlink()
                    print(f"Deleted: {file.name}")
                except Exception as e:
                    print(f"Failed to delete {file.name}: {e}")

        all_dfs = []
        
        if not refresh and self.is_data_valid(combined_path):
            print(f"Combined data is valid and cached: {combined_path}")
        else:
            for i, symbol in enumerate(self.top_btc_pairs, 1):
                print(f"\n[{i}/{len(self.top_btc_pairs)}] Downloading {symbol}...")
                try:
                    df = vbt.BinanceData.download_symbol(
                        symbol=symbol,
                        client=self.client,
                        interval=self.interval,
                        start=self.date_start,
                        end=self.date_end,
                        delay=500,
                        show_progress=False
                    )
                    df['symbol'] = symbol
                    all_dfs.append(df)
                    print(f"Data for {symbol} downloaded successfully.")
                    time.sleep(0.5)  
                except Exception as e:
                    print(f"Error with {symbol}: {e}")
                    continue

            if all_dfs:
                combined_df = pd.concat(all_dfs)  
                combined_df.to_parquet(combined_path, compression='snappy')
                print(f"\nCombined dataset saved: {combined_path}")

    def load_parquet(self, filename: str) -> pd.DataFrame:
        """Loads data from a .parquet."""
        if not filename.endswith('.parquet'):
            raise ValueError("Filename must end with .parquet")
        if not (self.data_dir / filename).exists():
            raise FileNotFoundError(f"{filename} does not exist in {self.data_dir}")
        pd.set_option('display.max_rows', None)
        return pd.read_parquet(self.data_dir / filename)