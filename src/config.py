#Imports
# ==========
import os
from pathlib import Path
from dotenv import load_dotenv
import MetaTrader5 as mt5

# ==========
# PATHS
# ==========
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

for d in (DATA_DIR, MODEL_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ==========
# Load environment variables from .env file
# ==========
dotenv_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=dotenv_path, override=True)  

# ==========
# MT5 CONNECTION SETTINGS
# ==========
MT5_LOGIN = int(os.getenv("MT5_LOGIN", 0))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
MT5_TERMINAL_PATH = os.getenv("MT5_TERMINAL_PATH", "")

# Map string timeframes to MT5 constants 
TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1
}

SELECTED_TIMEFRAME = os.getenv("TIMEFRAME", "H1")

DIRECTION_TIMEFRAME = TIMEFRAMES[SELECTED_TIMEFRAME]


# ==========
# TRAINING SETUP
# ==========

BASE_TIMEFRAME_MINUTES = 60  # H1 is 60 minutes
BASE_TRAIN_YEARS = 5  # Train on 5 years of H1 data

TIMEFRAME_MINUTES_MAP = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440
}

# Get the minutes for the selected timeframe
current_minutes = TIMEFRAME_MINUTES_MAP[SELECTED_TIMEFRAME]

# 2. Calculate the 'Timeframe Multiplier' relative to H1 (60 mins)
tf_multiplier = current_minutes / BASE_TIMEFRAME_MINUTES

# How much history to fetch
TRAIN_YEARS = round(BASE_TRAIN_YEARS * tf_multiplier, 2)
TRAIN_YEARS = max(TRAIN_YEARS, 0.2) # Ensure a minimum floor 

#How many bars to fetch when entering a trade
ENTRY_HISTORY_BARS = 50    


# ==========
# MODEL FILES (XGBoost)
# ==========

# The Pre-processing/Transformation objects
FEATURES_PATH = MODEL_DIR / f"xgb_direction_{SELECTED_TIMEFRAME}_features.pkl"
NON_STATIONARY_PATH = MODEL_DIR / f"non_stationary_cols_{SELECTED_TIMEFRAME}.pkl"
PCA_PATH = MODEL_DIR / f"pca_{SELECTED_TIMEFRAME}.pkl"

# Metadata about the training and final model
TRAIN_INFO_PATH = MODEL_DIR / f"train_info_{SELECTED_TIMEFRAME}.json"

# The actual XGBoost Model
MODEL_PATH = MODEL_DIR / f"xgb_direction_{SELECTED_TIMEFRAME}.json"


# ==========
# TRADING SETUP
# ==========

SYMBOL = os.getenv("SYMBOL", "EURUSD")
RISK_PER_TRADE = 0.01        # 1% of balance per trade
RISK_REWARD_RATIO = 2.0      # TP = 2x SL distance

# How much of the ATR to use for SL/TP calculation. 0.2 means 20% of ATR distance.
ATR_MULTIPLER = 0.2  

# Magic number to identify trades opened by this bot
MAGIC_NUMBER = 202602

# Max spread in points to allow for opening a trade
MAX_SPREAD_POINTS = 3

# Max number of open trades at the same time 
MAX_OPEN_TRADES = 1


# ==========
# LOG TRADES
# ==========

COLS = [
    "ticket", "symbol", "magic", "timeframe",
    "open_time_utc", "close_time_utc",
    "direction", "prob",
    "risk_amount",
    "volume",
    "entry_price", "exit_price",
    "sl_price", "tp_price",
    "spread_points_entry",
    "balance_before",
    "profit",
    "reason_close",
]