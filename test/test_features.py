import pytest
import pandas as pd
import numpy as np
from src.features import FeatureEngineering
from pandas.testing import assert_frame_equal

def test_add_all_features_mock(expected_features):
    # Create a simple DataFrame with enough rows to avoid feature creation issues
    data = {
        "Datetime": pd.date_range(start="2024-01-01", periods=50, freq='h'),
        "Open": np.linspace(1.10, 1.20, 50),
        "High": np.linspace(1.11, 1.21, 50),
        "Low": np.linspace(1.09, 1.19, 50),
        "Close": np.linspace(1.105, 1.205, 50),
        "Volume": np.random.randint(100, 1000, 50)
    }
    df_raw = pd.DataFrame(data)
    
    # Process the DataFrame through the feature engineering function
    processed_df = FeatureEngineering.add_all_features(df_raw)
    
    #Check if every column is present in the processed DataFrame
    for col in expected_features:
        assert col in processed_df.columns, f"Expected column '{col}' not found in processed DataFrame."
    #Check if the index is set to Datetime
    assert isinstance(processed_df.index, pd.DatetimeIndex), "Expected index to be DatetimeIndex after processing."
    #Check if there are any NaN values in the processed DataFrame
    assert not processed_df.isnull().values.any(), "Processed DataFrame contains NaN values after feature engineering."
    
    
def test_add_all_features_insufficient_data():
    # Create a DataFrame with fewer than 30 rows to trigger the error
    data = {
        "Datetime": pd.date_range(start="2024-01-01", periods=20, freq='h'),
        "Open": np.linspace(1.10, 1.20, 20),
        "High": np.linspace(1.11, 1.21, 20),
        "Low": np.linspace(1.09, 1.19, 20),
        "Close": np.linspace(1.105, 1.205, 20),
        "Volume": np.random.randint(100, 1000, 20)
    }
    df = pd.DataFrame(data)
    
    with pytest.raises(ValueError, match="DataFrame has only 20 rows. Not enough data to create features reliably."):
        processed_df = FeatureEngineering.add_all_features(df)


@pytest.mark.live
def test_add_all_features_live(loader, expected_features):
    df = loader.fetch_live_data(bars=50)
    processed_df = FeatureEngineering.add_all_features(df)
    
    #Check if every column is present in the processed DataFrame
    for col in expected_features:
        assert col in processed_df.columns, f"Expected column '{col}' not found in processed DataFrame."
    #Check if the index is set to Datetime
    assert isinstance(processed_df.index, pd.DatetimeIndex), "Expected index to be DatetimeIndex after processing."
    #Check if there are any NaN values in the processed DataFrame
    assert not processed_df.isnull().values.any(), "Processed DataFrame contains NaN values after feature engineering."