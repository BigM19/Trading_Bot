from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_TERMINAL_PATH
import MetaTrader5 as mt5
import logging


class MT5Connection:
    
    def __init__(self):
        self.connected = False
        self.account_info = None
        self.login = MT5_LOGIN
        self.password = MT5_PASSWORD
        self.server = MT5_SERVER
        self.terminal_path = MT5_TERMINAL_PATH
        
    def initialize_mt5(self) -> bool:
        """
        Initialize MT5 terminal and log into the account using credentials from config.
        Returns True on success, False on failure.
        """
        # Shutdown any previous session (avoids weird IPC issues)
        mt5.shutdown()
        
        ok = mt5.initialize(
            path=self.terminal_path,
            login=self.login,
            password=self.password,
            server=self.server,
        )
        
        logging.info(f"MT5 initialize: {ok}, error: {mt5.last_error()}")
        self.connected = True if ok else False
        return self.connected
    
    def fetch_account_info(self):
        if not self.connected:
            logging.error("MT5 not initialized. Call initialize_mt5() first.")
            return None
        
        info = mt5.account_info()
        if info is None:
            logging.error(f"Failed to retrieve account info: {mt5.last_error()}")
            return None
        
        self.account_info = info
        logging.info(f"Logged in as {info.login} on server {info.server}")
            return info
            


