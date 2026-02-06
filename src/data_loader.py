from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import MetaTrader5 as mt5
from .config import SYMBOL, DATA_DIR, DIRECTION_TIMEFRAME, TRAIN_YEARS, ENTRY_HISTORY_BARS
from .connection import MT5Connection

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

with MT5Connection() as conn:
    
    def __init__(self):
        self.symbol = SYMBOL
        self.data_dir = DATA_DIR
        self.timeframe = DIRECTION_TIMEFRAME
        self.train_years = TRAIN_YEARS
        self.entry_history_bars = ENTRY_HISTORY_BARS
        
    def get_training_date_range(self):
        """
        Compute (start, end) datetimes for the training range: last N years.
        """
        end = datetime.now()
        start = end - timedelta(days=365 * self.train_years)
        return start, end
    
    def fetch_raw_training_data(self) -> pd.DataFrame:
        """
        Fetch raw DIRECTION_TIMEFRAME candles for SYMBOL over the last TRAIN_YEARS.
        Returns a pandas DataFrame with:
        ['time', 'open', 'high', 'low', 'close', 'volume']
        """
    
        start, end = get_training_date_range()
        logging.info(f"Fetching {self.symbol} data from {start} to {end} ...")

        rates = mt5.copy_rates_range(
            self.symbol,
            self.timeframe,
            start,
            end,
        )

        if rates is None or len(rates) == 0:
            logging.error("No data returned by copy_rates_range: %s", mt5.last_error())
            raise RuntimeError("No historical data received from MT5.")

        df = pd.DataFrame(rates)
        
        # convert MT5 'time' (seconds since epoch) to pandas datetime
        df["time"] = pd.to_datetime(df["time"], unit="s")
    
        df = df.rename(columns={
            "time": "Datetime",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "tick_volume": "Volume"
        })

        df.drop(columns=["spread","real_volume"], inplace=True)
        df.dropna(inplace=True)
        
        logging.info(f"Received {len(df)} bars of {self.symbol} data.")
    
        return df
    

    def save_raw_training_data(self, df: pd.DataFrame) -> Path:
        """
        Save the raw data to the data/ folder as SYMBOL_TIMEFRAME_raw.csv.
        Returns the full path to the saved file.
        """
        filename = f"{self.symbol}_{self.timeframe}_raw.csv"
        out_path = DATA_DIR / filename
        df.to_csv(out_path, index=False)
        logging.info(f"Saved raw training data to: {out_path}")
        return out_path
    
    def fetch_recent_bars(self) -> pd.DataFrame:
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, self.entry_history_bars)
        if rates is None or len(rates) < self.entry_history_bars:
            raise RuntimeError("Not enough bars fetched")
            
        else:
            df = pd.DataFrame(rates)
            
            df["time"] = pd.to_datetime(df["time"], unit="s")
            
            df = df.rename(columns={
                "time": "Datetime",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "tick_volume": "Volume"
            })

            df.drop(columns=["spread","real_volume"], inplace=True)
        return df

