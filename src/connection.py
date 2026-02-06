from .config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_TERMINAL_PATH, SYMBOL
import MetaTrader5 as mt5
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class MT5Connection:
    
    def __init__(self):
        self.login = MT5_LOGIN
        self.password = MT5_PASSWORD
        self.server = MT5_SERVER
        self.terminal_path = MT5_TERMINAL_PATH
        self.symbol = SYMBOL
        
    def __enter__(self):
        """Allows usage: with MT5Connection() as conn:"""
        if self.initialize_mt5():
            return self
        else:
            raise ConnectionError("Failed to initialize MT5 connection.")    
    
    def __exit__(self):
        """Ensures terminal shuts down cleanly when the block ends."""
        mt5.shutdown()
        logging.info("MT5 connection closed.")
    
    def initialize_mt5(self) -> bool:
        """
        Initialize MT5 terminal and log into the account using credentials from config.
        Returns True on success, False on failure.
        """
        # Shutdown any previous session (avoids weird IPC issues)
        mt5.shutdown()
        
        #Initialize terminal 
        if not mt5.initialize(path=self.terminal_path, login=self.login, password=self.password, server=self.server):
            logging.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        # Explicit login check (prevents 'Authorized: False' issues)
        if not mt5.login(self.login, self.password, self.server):
            logging.error(f"Login failed for account {self.login}: {mt5.last_error()}")
            return False
        
        info = mt5.account_info()
        if info is None:
            logging.error(f"Failed to retrieve account info: {mt5.last_error()}")
            return False
        
        logging.info(f"Logged in as {info.login} on server {info.server}")
        return True

            
            

