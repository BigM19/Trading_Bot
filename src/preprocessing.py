import pandas as pd
from statsmodels.tsa.stattools import adfuller
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

