import pytest
from src.connection import MT5Connection
from src.data_loader import DataLoader
from src.features import FeatureEngineering

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