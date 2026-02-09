import mlflow
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, ParameterSampler
from sklearn.metrics import average_precision_score
from .preprocessing import Preprocessor
from collections import Counter
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


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
    
    def compute_scale_pos_weight(self, y) -> float:
        cnt = Counter(y)
        neg, pos = cnt.get(0, 0), cnt.get(1, 0)
        return (neg / max(1, pos)) if pos > 0 else 1.0

    def cross_validate(self, X, y, params):
        """Runs CV by fitting Preprocessor and Model independently per fold."""
        fold_scores = []

        for train_idx, test_idx in self.tscv.split(X):
            X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
            
            preprocessor = Preprocessor()
            X_train_pca = preprocessor.fit_transform(X_train_raw)
            X_test_pca = preprocessor.transform(X_test_raw)
            
            y_train = y.loc[X_train_pca.index].values.ravel()
            y_test = y.loc[X_test_pca.index].values.ravel()
            
            spw = self.compute_scale_pos_weight(y_train)
            
            # Train model on PCA-transformed data
            model = XGBClassifier(**self.base_params, **params, n_estimators=5000, early_stopping_rounds=50, scale_pos_weight=spw)
            model.fit(
                X_train_pca, y_train,
                eval_set=[(X_test_pca, y_test)],
                verbose=False
            )
            
            score = average_precision_score(y_test, model.predict_proba(X_test_pca)[:, 1])
            fold_scores.append(score)
            
        return float(np.mean(fold_scores))

    def run_experiment(self, X, y, param_grid):
        """Logs every hyperparameter combination as a child run in MLflow."""
        sampler = ParameterSampler(param_grid, n_iter=self.n_iter, random_state=69)
        best_score = -np.inf
        best_params = None
        
        for i, params in enumerate(sampler):
            # Log each hyperparameter combination as a child run
            with mlflow.start_run(run_name=f"XGB_CV_{i}", nested=True):
                mean_aucpr = self.cross_validate(X, y, params)
                
                mlflow.log_params(params)
                mlflow.log_metric("mean_aucpr", mean_aucpr)
                
                if mean_aucpr > best_score:
                    best_score = mean_aucpr
                    best_params = params
                    
            logging.info(f"Completed run {i+1}/{self.n_iter} with AUPR: {mean_aucpr:.4f}")
        return best_params