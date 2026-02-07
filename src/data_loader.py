from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import MetaTrader5 as mt5
from .config import (
    SYMBOL, DATA_DIR, DIRECTION_TIMEFRAME,
    TRAIN_YEARS, ENTRY_HISTORY_BARS
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DataProcessor:
    
    @staticmethod
    def clean_data(rates) -> pd.DataFrame:
        """
        Convert raw MT5 data to a clean DataFrame with proper column names and types.
        """
        if rates is None or len(rates) == 0:
            logging.error("No data to clean.")
            raise ValueError("Empty data received for cleaning.")
            return pd.DataFrame()  # Return empty DataFrame on error
        
        df = pd.DataFrame(rates)
        
        # convert MT5 'time' to pandas datetime
        df["time"] = pd.to_datetime(df["time"], unit="s")
    
        df = df.rename(columns={
            "time": "Datetime",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "tick_volume": "Volume"
        })

        #Drop unnecessary columns and handle missing values
        df.drop(columns=["spread","real_volume"], inplace=True)
        df.dropna(inplace=True)
        
        return df

class DataLoader:
    
    def __init__(self):
        self.symbol = SYMBOL
        self.data_dir = DATA_DIR
        self.timeframe = DIRECTION_TIMEFRAME
        self.processor = DataProcessor()
        
    
    def fetch_training_data(self, years: float = TRAIN_YEARS) -> pd.DataFrame:
        """
        Fetch raw DIRECTION_TIMEFRAME candles for SYMBOL over the last TRAIN_YEARS.
        Returns a pandas DataFrame with:
        ['time', 'open', 'high', 'low', 'close', 'volume']
        """
    
        end = datetime.now()
        start = end - timedelta(days=365 * years)
        logging.info(f"Fetching {self.symbol} data from {start.date()} to {end.date()} ...")

        rates = mt5.copy_rates_range(
            self.symbol,
            self.timeframe,
            start,
            end,
        )

        df = self.processor.clean_data(rates)
        if df.empty:
            raise RuntimeError(f"Failed to fetch historical data for {self.symbol}.")

        logging.info(f"Received {len(df)} bars of {self.symbol} data.") 
        return df
    
    def fetch_live_data(self, bars: int = ENTRY_HISTORY_BARS) -> pd.DataFrame:
        """
        Fetch the most recent ENTRY_HISTORY_BARS bars for SYMBOL at DIRECTION_TIMEFRAME.
        Returns a pandas DataFrame with:
        ['time', 'open', 'high', 'low', 'close', 'volume']
        """
        rates = mt5.copy_rates_from_pos(
            self.symbol,
            self.timeframe,
            0,
            bars
        )

        df = self.processor.clean_data(rates)
        if df.empty:
            raise RuntimeError(f"Failed to fetch live data for {self.symbol}.")

        logging.info(f"Received {len(df)} bars of live data for {self.symbol}.") 
        return df
    
    
    def save_to_csv(self, df: pd.DataFrame, suffix="raw", dir=DATA_DIR) -> Path:
        """
        Centralized method to save DataFrames to CSV with consistent naming and logging.
        """
        filename = f"{self.symbol}_{self.timeframe}_{suffix}.csv"
        out_path = dir / filename
        df.to_csv(out_path, index=False)
        logging.info(f"Data saved to: {out_path}")
        return out_path
    