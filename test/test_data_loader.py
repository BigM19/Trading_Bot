import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import MetaTrader5 as mt5
from src.config import TEST_DATA_DIR
from src.data_loader import DataProcessor


def test_clean_data():
    fake_rates = np.array([
        (1770387705, 1.10, 1.12, 1.09, 1.11, 100, 1, 0),
    ], dtype=[('time', '<i8'), ('open', '<f8'), ('high', '<f8'), 
                ('low', '<f8'), ('close', '<f8'), ('tick_volume', '<i8'),
                ('spread', '<i4'), ('real_volume', '<i8')])
    
    df = DataProcessor.clean_data(fake_rates)
    
    assert "Datetime" in df.columns
    assert "Volume" in df.columns
    assert "spread" not in df.columns # Verify drop
    assert df.iloc[0]["Open"] == 1.10
    assert isinstance(df.iloc[0]["Datetime"], pd.Timestamp)

def test_clean_data_empty():
    empty_rates = np.array([], dtype=[('time', '<i8'), ('open', '<f8'), ('high', '<f8'), 
                                    ('low', '<f8'), ('close', '<f8'), ('tick_volume', '<i8'),
                                    ('spread', '<i4'), ('real_volume', '<i8')])
    with pytest.raises(ValueError, match="Empty data received for cleaning."):
        df = DataProcessor.clean_data(empty_rates)
    

@pytest.mark.live
def test_fetch_training_data(loader):
    df = loader.fetch_training_data(years=0.1) 
    assert not df.empty, f"Expected non-empty DataFrame, got {len(df)} rows."
    assert len(df) > 0
    assert "Datetime" in df.columns
    assert "Volume" in df.columns
    assert "spread" not in df.columns 
    

@pytest.mark.live
def test_fetch_live_data(loader):
    df = loader.fetch_live_data(bars=50)
    assert not df.empty, "Live data should not be empty"
    assert len(df) >= 30, f"Expected at least 30 bars of live data, got {len(df)}. Not enough bars create features."
    assert "Datetime" in df.columns
    assert "Volume" in df.columns
    assert "spread" not in df.columns
    
def test_fetch_live_data_insufficient_bars(loader):
    with pytest.raises(ValueError, match="Expected at least 30 bars of live data, got 10. Not enough bars create features."):
        df = loader.fetch_live_data(bars=10)    
    
def test_save_to_csv(mock_data_factory, loader):
    df = mock_data_factory(10).set_index("Datetime")
    out_path = loader.save_to_csv(df, suffix="test", dir=TEST_DATA_DIR)
    assert out_path.exists(), f"Expected file at {out_path} to exist."
    loaded_df = pd.read_csv(out_path, parse_dates=["Datetime"], index_col="Datetime")
    assert_frame_equal(df, loaded_df, check_dtype=False)
