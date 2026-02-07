import pytest
import MetaTrader5 as mt5

@pytest.mark.live
def test_initialization(connection):
    """Verify that the terminal is initialized and account is logged in."""
    assert connection.is_valid_connection() is True
    assert mt5.account_info() is not None

@pytest.mark.live
def test_account_details(connection):
    """Ensure we are connected to the correct account from .env."""
    acc_info = mt5.account_info()
    from src.config import MT5_LOGIN
    assert acc_info.login == MT5_LOGIN

@pytest.mark.live
def test_trading_allowed_check(connection):
    """Check the status of algorithmic trading (expected to be a bool)."""
    allowed = connection.check_trading_allowed()
    assert isinstance(allowed, bool)

@pytest.mark.live
def test_symbol_access(connection):
    """Verify the symbol from config is visible in the terminal."""
    from src.config import SYMBOL
    symbol_info = mt5.symbol_info(SYMBOL)
    assert symbol_info is not None
    assert symbol_info.name == SYMBOL