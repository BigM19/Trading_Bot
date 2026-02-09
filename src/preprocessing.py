from importlib.resources import path
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from .config import MODEL_DIR
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class Preprocessor:
    def __init__(self, n_components=0.8):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components, random_state=69)
        self.non_stat_cols = []
        self.feature_cols = []
        self.is_fitted = False
        
    def find_and_diff_columns(self, df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
        """
        Detects non-stationarity columns and applies first-differencing.
        Stored non_stat_cols ensures live data is diffed exactly like training data.
        """
        if not self.is_fitted:
            # Only identify non-stationary columns during the training 'fit' phase
            cols_to_test = (c for c in df.columns if not (c == "DOW" or c.startswith("Signal_")))
            for col in cols_to_test:
                series = df[col].dropna().values
                if len (series) > 30 and adfuller(series)[1] > alpha:
                    self.non_stat_cols.append(col)
        
        df_stat = df.copy()
        if self.non_stat_cols:
            df_stat[self.non_stat_cols] = df_stat[self.non_stat_cols].diff()
        
        return df_stat.dropna()
        
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:  
        """Fits the pipeline on training data."""
        X_stat = self.find_and_diff_columns(X)
        self.feature_cols = X_stat.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_stat)
        X_pca = self.pca.fit_transform(X_scaled)
        
        self.is_fitted = True
        return self._to_pca_df(X_pca, X_stat.index)  
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms validation/live data using fitted state."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")
            
        X_stat = self.find_and_diff_columns(X)
        X_scaled = self.scaler.transform(X_stat[self.feature_cols])
        X_pca = self.pca.transform(X_scaled)
        return self._to_pca_df(X_pca, X_stat.index)

    def _to_pca_df(self, data, index):
        cols = [f"PC{i+1}" for i in range(data.shape[1])]
        return pd.DataFrame(data, columns=cols, index=index)