from core.data_loader import DataLoader
from strategies.sma_cross import SMACrossover
import matplotlib.pyplot as plt

if __name__ == "__main__":
    loader = DataLoader(
        data_dir="data",
        interval='1m',
        date_start='2025-02-01',
        date_end='2025-02-28',
        top_n=100
    )

    loader.download_data()
    data_df = loader.load_parquet('btc_1m_20250201.parquet')
    print(data_df.head())

    