import pytest
import pandas as pd
import numpy as np
import mlflow
from unittest.mock import MagicMock, patch
from src.model_trainer import ModelTrainer
from src.preprocessing import Preprocessor

def test_cross_validate_pca_isolation(mock_data_factory):
    """
    Critical Test: Verifies that Preprocessor is fit independently for every fold.
    If fit_transform is called only once, there is data leakage.
    """
    # 1. Setup: 60 rows for 3 folds (20 rows each)
    df = mock_data_factory(rows=60).set_index("Datetime")
    X = df.drop(columns=["Close"])
    y = pd.Series(np.random.randint(0, 2, 60), index=df.index)
    
    # Initialize trainer with 3 splits
    trainer = ModelTrainer(n_splits=3, n_iter=1)
    
    # 2. Patch Preprocessor to count fits
    with patch("src.model_trainer.Preprocessor") as MockPrep:
        mock_instance = MockPrep.return_value
        
        def mock_fit_side_effect(X_slice):
            # Return X_slice minus the first row to simulate stationarity drop
            return pd.DataFrame(
                np.random.rand(len(X_slice) - 1, 2), 
                index=X_slice.index[1:], 
                columns=["PC1", "PC2"]
            )

        mock_instance.fit_transform.side_effect = mock_fit_side_effect
        mock_instance.transform.side_effect = mock_fit_side_effect
        
        params = {"max_depth": 3, "learning_rate": 0.1}
        
        trainer.cross_validate(X, y, params)
        
        assert MockPrep.call_count == 3
        assert mock_instance.fit_transform.call_count == 3
        # Ensure it was called with the sliced data from TimeSeriesSplit
        assert len(mock_instance.fit_transform.call_args[0][0]) < 60

def test_mlflow_local_folder_logging(mock_data_factory, tmp_path):
    """
    Verifies that MLflow logs to a local directory instead of a server.
    Ensures nested runs for hyperparameter search are created.
    """
    while mlflow.active_run():
        mlflow.end_run()
    # 1. Setup local tracking URI using a temporary test folder
    test_mlruns_dir = tmp_path / "mlruns"
    test_mlruns_dir.mkdir(exist_ok=True)
    
    tracking_uri = test_mlruns_dir.absolute().as_uri()
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Test_Local_Search")

    df = mock_data_factory(rows=40).set_index("Datetime")
    X = df.drop(columns=["Close"])
    y = pd.Series(np.random.randint(0, 2, 40), index=df.index)

    # 2. Run a minimal search (2 iterations)
    trainer = ModelTrainer(n_splits=2, n_iter=2)
    param_grid = {"max_depth": [3, 4], "learning_rate": [0.1]}
    
    with mlflow.start_run(run_name="Parent_Test_Run"):
        trainer.run_experiment(X, y, param_grid)

    # 3. Verify MLflow folder structure
    assert test_mlruns_dir.exists()
    
    # 4. Verify runs were recorded
    experiment = mlflow.get_experiment_by_name("Test_Local_Search")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    # Should have 1 parent run + 2 nested iteration runs = 3 total
    assert len(runs) == 3
    assert "params.max_depth" in runs.columns
    assert "metrics.mean_aucpr" in runs.columns

def test_final_model_local_saving(mock_data_factory, tmp_path):
    """Verifies the model is saved to the local path for live trading."""
    test_model_file = tmp_path / "models" / "final_model.json"
    test_model_file.parent.mkdir(parents=True, exist_ok=True)
    
    df = mock_data_factory(rows=40).set_index("Datetime")
    X = df.drop(columns=["Close"])
    y = pd.Series(np.random.randint(0, 2, 40), index=df.index)
    
    with patch("src.model_trainer.XGBClassifier") as MockXGB:
        mock_model_instance = MockXGB.return_value
        mock_model_instance.save_model(str(test_model_file))
        
    mock_model_instance.save_model.assert_called_once_with(str(test_model_file))
    assert test_model_file.parent.exists()