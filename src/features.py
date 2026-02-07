import pandas as pd
import numpy as np
import ta
import logging 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineering:
    
    @staticmethod
    def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardized class for adding technical indicators and signals.
        Ensures identical processing for training and live inference.
        """
        df = df.copy()
        
        # --- Ensure DatetimeIndex ---
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df = df.set_index("Datetime")
        
        # --- Add features ---
        df = FeatureEngineering._add_basic_features(df)
        df = FeatureEngineering._add_momentum_indicators(df)
        df = FeatureEngineering._add_volume_indicators(df)
        df = FeatureEngineering._add_volatility_indicators(df)
        df = FeatureEngineering._add_trend_indicators(df)
        df = FeatureEngineering._add_signals(df)
        
        # --- Final cleanup ---
        df.dropna(inplace=True)
        return df
        
    @staticmethod
    def _add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
        df["Returns"] = df["Close"].pct_change()
        df["Range"] = (df["High"] / df["Low"]) - 1
        df["DOW"] = df.index.dayofweek
        return df
    
    @staticmethod
    def _add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df["ROC"] = ta.momentum.ROCIndicator(df["Close"], window=12).roc()
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
        df["STOCH"] = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window=14).stoch()
        df["VROC"] = df["Volume"].pct_change(periods=14) * 100
        return df
    
    @staticmethod
    def _add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df["CMF"] = ta.volume.ChaikinMoneyFlowIndicator(df["High"], df["Low"], df["Close"], df["Volume"], window=20).chaikin_money_flow()
        df["MFI"] = ta.volume.MFIIndicator(df["High"], df["Low"], df["Close"], df["Volume"], window=14).money_flow_index()
        df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
        df["VWAP"] = ta.volume.VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"], window=14).volume_weighted_average_price()
        return df
    
    @staticmethod
    def _add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
        indicator_bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
        df["BB_upper"] = indicator_bb.bollinger_hband()
        df["BB_middle"] = indicator_bb.bollinger_mavg()
        df["BB_lower"] = indicator_bb.bollinger_lband()
        indicator_donchian = ta.volatility.DonchianChannel(df["High"], df["Low"], df["Close"], window=20)
        df["Donchian_Upper"] = indicator_donchian.donchian_channel_hband()
        df["Donchian_Lower"] = indicator_donchian.donchian_channel_lband()
        df["Donchian_Middle"] = indicator_donchian.donchian_channel_mband()
        return df
    
    @staticmethod
    def _add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
        adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
        df["ADX"] = adx.adx()
        df['DMP'] = adx.adx_pos()   # +DI
        df['DMN'] = adx.adx_neg()   # -DI
        df["CCI"] = ta.trend.CCIIndicator(df["High"], df["Low"], df["Close"], window=20).cci()
        df["MA_8"] = ta.trend.ema_indicator(df["Close"], window=8)
        df["MA_20"] = ta.trend.ema_indicator(df["Close"], window=20)
        ichimoku = ta.trend.IchimokuIndicator(df["High"], df["Low"])
        df["Ichi_A"] = ichimoku.ichimoku_a()
        df["Ichi_B"] = ichimoku.ichimoku_b()
        df["Ichi_base"] = ichimoku.ichimoku_base_line()
        df["MACD_Line"] = ta.trend.MACD(df["Close"]).macd()
        df['Signal_Line'] = ta.trend.MACD(df['Close']).macd_signal()
        return df

    @staticmethod
    def _add_signals(df: pd.DataFrame) -> pd.DataFrame:
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
        
        return df

    @staticmethod
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
        
    @staticmethod
    def make_label(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """
        Create target column.
        If next day close is higher Target=1, if lowwer Target=0.
        """
        df = df.copy()
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        return df




