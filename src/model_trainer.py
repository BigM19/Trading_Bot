import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, ParameterSampler
from sklearn.metrics import average_precision_score, roc_auc_score
from src.preprocessing import Preprocessor
from config import MODEL_DIR, MODEL_PATH

class ModelTrainer:
    def __init__(self, n_splits=5, n_iter=50):
        self.n_splits = n_splits
        self.n_iter = n_iter
        self.tscv = TimeSeriesSplit(n_splits=self.n_splits)
        self.base_params = {
            'objective': 'binary:logistic',
            'booster': 'gbtree',
            'eval_metric': 'aucpr',
            'random_state': 69,
            'tree_method': 'hist',
            'device': 'cuda'
        }
        
    def cross_validation(self,X, y, params):
        """Runs CV by fitting Preprocessor and Model independently per fold."""
        fold_scores = []
        best_iters = []

        for train_idx, test_idx in self.tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = XGBClassifier(**self.base_params, **params

