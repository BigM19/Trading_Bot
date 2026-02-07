import pytest
from src.connection import MT5Connection
from src.data_loader import DataLoader
from src.features import FeatureEngineering
import numpy as np
import pandas as pd

@pytest.fixture
def mock_data_factory():
    """
    Factory fixture to create mock DataFrames for testing feature engineering.
    Provides a simple interface to generate DataFrames with specified characteristics.
    """
    def _create_mock_data(rows=50):
        data = {
            "Datetime": pd.date_range(start="2024-01-01", periods=rows, freq='h'),
            "Open": np.linspace(1.10, 1.20, rows),
            "High": np.linspace(1.11, 1.21, rows),
            "Low": np.linspace(1.09, 1.19, rows),
            "Close": np.linspace(1.105, 1.205, rows),
            "Volume": np.random.randint(100, 1000, rows)
        }
        return pd.DataFrame(data)
    
    return _create_mock_data

@pytest.fixture(scope="session")
def connection():
    """
    Fixture to initialize the MT5 connection once per test session.
    Automatically shuts down the connection after tests are finished.
    """
    with MT5Connection() as conn:
        yield conn
        
@pytest.fixture()
def loader():
    """
    Fixture to provide a DataLoader instance for tests.
    """
    return DataLoader()

@pytest.fixture(scope="session")
def expected_features():
    """
    Provides the master list of features the model expects.
    This ensures all tests stay in sync with the FeatureEngineering class.
    """
    return FeatureEngineering.get_feature_columns()