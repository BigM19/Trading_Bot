import pandas as pd
import numpy as np
import ta
import logging 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineering:
    
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Takes a raw DataFrame with columns ['time', 'open', 'high', 'low', 'close', 'volume'] 
        and adds technical indicators."""
        
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df = df.set_index("Datetime")
        else:
            df.index = pd.to_datetime(df.index)
        
        # --- Basic features ---
        #Returns
        df["Returns"] = (df["Close"] / df["Close"].shift(1)) - 1

        #Range
        df["Range"] = (df["High"] / df["Low"]) - 1

        #Day of the Week
        df.index = pd.to_datetime(df.index)
        df["DOW"] = df.index.dayofweek
        
        logging.info("Basic features added: Returns, Range, DOW")
        logging.ingfo(f"Shape of DataFrame after basic features: {df.shape}")
        
        # --- Momentum ---
        #ROC (Rate of Change)
        df["ROC"] = ta.momentum.ROCIndicator(df["Close"], window=12).roc()

        #RSI (Relative Strength Index)
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

        #Stoch (Stochastic Oscillator)
        df["STOCH"] = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=14).stoch()

        #VROC (Volume Rate of Change)
        df["VROC"] = (df["Volume"] - df["Volume"].shift(14)) / df["Volume"].shift(14) * 100
        
        logging.info("Momentum indicators added: ROC, RSI, STOCH, VROC")
        logging.info(f"Shape of DataFrame after momentum features: {df.shape}")
        
        # --- Volume ---
        #CMF (Chaikin Money Flow)
        df["CMF"] = ta.volume.ChaikinMoneyFlowIndicator(df["High"], df["Low"], df["Close"], df["Volume"], window=20).chaikin_money_flow()

        #MFI (Money Flow Index)
        df["MFI"] = ta.volume.MFIIndicator(df["High"], df["Low"], df["Close"], df["Volume"], window=14).money_flow_index()

        #OBV (On-Balance Volume)
        df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()

        #VWAP (Volume Weighted Average Price)
        df["VWAP"] = ta.volume.VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"], window=14).volume_weighted_average_price()

        logging.info("Volume indicators added: CMF, MFI, OBV, VWAP")
        logging.info(f"Shape of DataFrame after volume features: {df.shape}")

        # --- Volatility ---
        #ATR (Average True Range)
        df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()

        #Bollinger Bands
        indicator_bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
        df["BB_upper"] = indicator_bb.bollinger_hband()
        df["BB_middle"] = indicator_bb.bollinger_mavg()
        df["BB_lower"] = indicator_bb.bollinger_lband()

        #Donchian Channel
        indicator_donchian = ta.volatility.DonchianChannel(df["High"], df["Low"], df["Close"], window=20)
        df["Donchian_Upper"] = indicator_donchian.donchian_channel_hband()
        df["Donchian_Lower"] = indicator_donchian.donchian_channel_lband()
        df["Donchian_Middle"] = indicator_donchian.donchian_channel_mband()

        logging.info("Volatility indicators added: ATR, Bollinger Bands, Donchian Channel")
        logging.info(f"Shape of DataFrame after volatility features: {df.shape}")
        
        # --- Trend ---
        #ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
        df["ADX"] = adx.adx()
        df['DMP'] = adx.adx_pos()   # +DI
        df['DMN'] = adx.adx_neg()   # -DI

        #CCI (Commodity Channel Index)
        df["CCI"] = ta.trend.CCIIndicator(df["High"], df["Low"], df["Close"], window=20).cci()

        #EMA (8, 20)
        df["MA_8"] = ta.trend.ema_indicator(df["Close"], window=8)
        df["MA_20"] = ta.trend.ema_indicator(df["Close"], window=20)

        #Ichimoku Cloud
        ichimoku = ta.trend.IchimokuIndicator(df["High"], df["Low"])
        df["Ichi_A"] = ichimoku.ichimoku_a()
        df["Ichi_B"] = ichimoku.ichimoku_b()
        df["Ichi_base"] = ichimoku.ichimoku_base_line()

        #MACD (Moving Average Convergence/Divergence)
        df["MACD_Line"] = ta.trend.MACD(df["Close"]).macd()
        df['Signal_Line'] = ta.trend.MACD(df['Close']).macd_signal()
        
        logging.info("Trend indicators added: ADX, CCI, EMA, Ichimoku, MACD")
        logging.info(f"Shape of DataFrame after trend features: {df.shape}")
        
        return df
    
    def add_signals(df: pd.DataFrame) -> pd.DataFrame:
        """Adds trading signals based on the indicators."""
        
        #MA Signal
        df.loc[df["MA_8"] > df["MA_20"], "Signal_MA"] = 1
        df.loc[df["MA_8"] <= df["MA_20"], "Signal_MA"] = 0

        #Price Above / Below MA
        df['Signal_Price_Above_MA'] = (df['Close'] > df['MA_20']).astype(int)

        #MACD Signal
        df.loc[df["MACD_Line"] > df["Signal_Line"], "Signal_MACD"] = 1
        df.loc[df["MACD_Line"] <= df["Signal_Line"], "Signal_MACD"] = 0

        #RSI Signal
        df["Signal_RSI"] = 0 #nothing
        df.loc[df["RSI"] <= 30, "Signal_RSI"] = 1 #buy
        df.loc[df["RSI"] >= 70, "Signal_RSI"] = 2 #sell

        #Bollinger Band Signal
        df['Signal_BB'] = 0
        df.loc[df['Close'] <= df['BB_lower'], "Signal_BB"] = 1
        df.loc[df['Close'] >= df['BB_upper'], "Signal_BB"] = 2

        #ATR Signal
        df['Signal_ATR'] = (df['ATR'] >= df['ATR'].rolling(14).mean()).astype(int)

        #On-Balance Volume Signal
        df['Signal_OBV'] = (df['OBV'] >= df['OBV'].rolling(14).mean()).astype(int)

        #MFI Signal
        df['Signal_MFI'] = 0
        df.loc[df["MFI"] <= 20 , "Signal_MFI"] = 1
        df.loc[df['MFI'] >= 80, "Signal_MFI"] = 2

        ### VROC Signal
        df['Signal_VROC'] = (df['VROC'] >= df['VROC'].rolling(14).mean()).astype(int)

        ###ADX Signal
        df['Signal_ADX'] = 0
        trend_strong = df['ADX'] > 20
        df.loc[(df['DMP'] > df['DMN']) & trend_strong, 'Signal_ADX'] = 1 #BUY
        df.loc[(df['DMN'] >= df['DMP']) & trend_strong, 'Signal_ADX'] = 2 #SELL

        #CCI Signal
        df.loc[(df["CCI"] <= -100) | ((df["CCI"] > 0) & (df["CCI"] < 100)), "Signal_CCI"] = 1
        df.loc[(df["CCI"] >= 100) | ((df["CCI"] <= 0) & (df["CCI"] > -100)), "Signal_CCI"] = 0
        
        logging.info("Trading signals added based on indicators.")
        logging.info(f"Shape of DataFrame after adding signals: {df.shape}")
        
        return df
    
    def get_feature_columns() -> list[str]:
        """
        Single source of truth. PCA and XGBoost depend on this exact order.
        """
        return [
            "Open","High","Low","Close","Volume",
            "Returns", "Range", "DOW",
            "ROC", "RSI", "STOCH", "VROC",
            "CMF", "MFI", "OBV", "VWAP",
            "ATR", "BB_upper", "BB_middle", "BB_lower",
            "Donchian_Upper", "Donchian_Lower", "Donchian_Middle",
            "ADX", "DMP", "DMN", "CCI", "MA_8", "MA_20",
            "Ichi_A", "Ichi_B", "Ichi_base",
            "MACD_Line", "Signal_Line",
            "Signal_MA", "Signal_Price_Above_MA", "Signal_MACD",
            "Signal_RSI", "Signal_BB", "Signal_ATR", "Signal_OBV",
            "Signal_MFI", "Signal_VROC", "Signal_ADX", "Signal_CCI",
        ]
        
    def make_label(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """
        Create target column.
        If next day close is higher Target=1, if lowwer Target=0.
        """
        df = df.copy()

        df["Target"] = 0
        df.loc[df["Close"].shift(-1) > df["Close"], "Target"] = 1

        return df

    def split_label_from_features(df: pd.DataFrame):
        X = df.iloc[:,:-1]
        y = df["Target"]
        
        return X, y
    

