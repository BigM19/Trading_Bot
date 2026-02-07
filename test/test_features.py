import pytest
import pandas as pd
import numpy as np
from src.features import FeatureEngineering
from pandas.testing import assert_frame_equal
from src.config import TEST_DATA_DIR

def test_add_all_features_mock(mock_data_factory, expected_features):
    # Create a simple DataFrame with enough rows to avoid feature creation issues
    df_raw = mock_data_factory(50)

    # Process the DataFrame through the feature engineering function
    processed_df = FeatureEngineering.add_all_features(df_raw)
    
    #Check if every column is present in the processed DataFrame
    for col in expected_features:
        assert col in processed_df.columns, f"Expected column '{col}' not found in processed DataFrame."
    #Check if the index is set to Datetime
    assert isinstance(processed_df.index, pd.DatetimeIndex), "Expected index to be DatetimeIndex after processing."
    #Check if there are any NaN values in the processed DataFrame
    assert not processed_df.isnull().values.any(), "Processed DataFrame contains NaN values after feature engineering."
    
    
def test_add_all_features_insufficient_data(mock_data_factory):
    # Create a DataFrame with fewer than 30 rows to trigger the error
    df = mock_data_factory(20)
    
    with pytest.raises(ValueError, match="DataFrame has only 20 rows. Not enough data to create features reliably."):
        df = FeatureEngineering.add_all_features(df)


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
    
    
def test_make_labels_mock(mock_data_factory):
    df_raw = mock_data_factory(50)
    df_features = FeatureEngineering.add_all_features(df_raw)
    df_labeled = FeatureEngineering.make_label(df_features)
    
    assert "Target" in df_labeled.columns, "Expected 'Target' column not found after labeling."
    assert set(df_labeled["Target"].unique()).issubset({0, 1}), "Target column contains values other than 0 and 1."
    
def test_split_labels_from_features(mock_data_factory):
    df_raw = mock_data_factory(50)
    df_features = FeatureEngineering.add_all_features(df_raw)
    df_labeled = FeatureEngineering.make_label(df_features)
    
    X, y = FeatureEngineering.split_labels_from_features(df_labeled)
    
    assert isinstance(X, pd.DataFrame), "Expected features (X) to be a DataFrame."
    assert isinstance(y, pd.Series), "Expected labels (y) to be a Series."
    assert set(y.unique()).issubset({0, 1}), "Labels contain values other than 0 and 1."
    

@pytest.mark.live
def test_make_labels_live(loader):
    df = loader.fetch_live_data(bars=50)
    df_features = FeatureEngineering.add_all_features(df)
    df_labeled = FeatureEngineering.make_label(df_features)
    
    assert "Target" in df_labeled.columns, "Expected 'Target' column not found after labeling."
    assert set(df_labeled["Target"].unique()).issubset({0, 1}), "Target column contains values other than 0 and 1."
    

@pytest.mark.live
def test_split_labels_from_features_live(loader):
    df = loader.fetch_live_data(bars=50)
    df_features = FeatureEngineering.add_all_features(df)
    df_labeled = FeatureEngineering.make_label(df_features)
    
    X, y = FeatureEngineering.split_labels_from_features(df_labeled)
    
    assert isinstance(X, pd.DataFrame), "Expected features (X) to be a DataFrame."
    assert isinstance(y, pd.Series), "Expected labels (y) to be a Series."
    assert set(y.unique()).issubset({0, 1}), "Labels contain values other than 0 and 1."
    

@pytest.mark.live
def test_save_to_csv_live(loader):
    df_raw = loader.fetch_live_data(bars=50)
    df_features = FeatureEngineering.add_all_features(df_raw)
    df_labeled = FeatureEngineering.make_label(df_features)
    
    out_path = loader.save_to_csv(df_labeled, suffix="features_test", dir=TEST_DATA_DIR)
    assert out_path.exists(), f"Expected file {out_path} to exist after saving."