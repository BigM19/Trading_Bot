import pytest
import pandas as pd
import numpy as np
from src.preprocessing import Preprocessor

def test_preprocessor_fit_transform_flow(mock_data_factory):
    """Verifies that fit_transform reduces dimensions and stores state."""
    # 1. Setup: 100 rows, 5 initial features
    df = mock_data_factory(rows=100).set_index("Datetime")
    # Add a non-stationary column (a simple trend)
    df["Trend_Col"] = np.linspace(1, 10, 100)
    
    preprocessor = Preprocessor(n_components=0.8)
    
    # 2. Act: Fit and Transform
    X_pca = preprocessor.fit_transform(df)
    
    # 3. Assertions
    assert preprocessor.is_fitted is True
    assert len(preprocessor.feature_cols) == len(df.columns)
    assert X_pca.shape[1] <= len(df.columns)  # PCA should reduce or keep dims
    assert isinstance(X_pca.index, pd.DatetimeIndex)
    assert not X_pca.isnull().values.any()

def test_preprocessor_stationarity_consistency(mock_data_factory):
    """Ensures non-stationary columns are identified and diffed."""
    df_train = mock_data_factory(rows=100).set_index("Datetime")
    # Force a non-stationary random walk
    df_train["Non_Stationary"] = np.cumsum(np.random.randn(100))
    
    preprocessor = Preprocessor()
    
    # Identify and diff
    df_diffed = preprocessor.find_and_diff_columns(df_train)
    
    # Check if 'Non_Stationary' was caught
    assert "Non_Stationary" in preprocessor.non_stat_cols
    assert len(df_diffed) == len(df_train) - 1 # First row lost to diffing

def test_transform_fails_if_not_fitted(mock_data_factory):
    """Ensures we cannot transform live data without a fitted state."""
    df = mock_data_factory(rows=50).set_index("Datetime")
    preprocessor = Preprocessor()
    
    with pytest.raises(RuntimeError, match="Preprocessor must be fitted before transform."):
        preprocessor.transform(df)

def test_transform_live_consistency(mock_data_factory):
    """Verifies live data transformation uses training parameters."""
    df_train = mock_data_factory(rows=100).set_index("Datetime")
    df_live = mock_data_factory(rows=10).set_index("Datetime")
    
    preprocessor = Preprocessor(n_components=2)
    preprocessor.fit_transform(df_train)
    
    # Transform live data
    X_live_pca = preprocessor.transform(df_live)
    
    # Assert: Live PCA must have same column count as Training PCA
    assert X_live_pca.shape[1] == 2