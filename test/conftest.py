import pytest
from src.connection import MT5Connection
from src.data_loader import DataLoader

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