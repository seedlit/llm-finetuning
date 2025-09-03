"""
MLflow utilities for experiment tracking.
"""
import os
from typing import Any, Dict, Optional

import mlflow

from configs.config import MLFLOW_CONFIG
from src.utils.helpers import setup_logging

logger = setup_logging()


class MLflowTracker:
    """MLflow experiment tracking utilities."""
    
    def __init__(self, experiment_name: Optional[str] = None):
        """Initialize MLflow tracker."""
        self.experiment_name = experiment_name or MLFLOW_CONFIG["experiment_name"]
        
        # Set tracking URI
        mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
        
        # Create experiment if it doesn't exist
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created MLflow experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {self.experiment_name}")
            
            mlflow.set_experiment(self.experiment_name)
            
        except Exception as e:
            logger.error(f"Failed to setup MLflow experiment: {e}")
            raise
    
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run."""
        if run_name is None:
            run_name = f"{MLFLOW_CONFIG['run_name_prefix']}_{self._generate_run_suffix()}"
        
        run = mlflow.start_run(run_name=run_name)
        logger.info(f"Started MLflow run: {run_name}")
        return run
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        try:
            mlflow.log_params(params)
            logger.debug(f"Logged {len(params)} parameters to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow."""
        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
            logger.debug(f"Logged {len(metrics)} metrics to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")
    
    def log_artifacts(self, artifact_path: str, artifact_name: Optional[str] = None) -> None:
        """Log artifacts to MLflow."""
        try:
            if artifact_name:
                mlflow.log_artifact(artifact_path, artifact_name)
            else:
                mlflow.log_artifacts(artifact_path)
            logger.debug(f"Logged artifacts from {artifact_path}")
        except Exception as e:
            logger.warning(f"Failed to log artifacts: {e}")
    
    def log_model(self, model_path: str, artifact_path: str = "model") -> None:
        """Log model to MLflow."""
        try:
            mlflow.log_artifacts(model_path, artifact_path)
            logger.info(f"Logged model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to log model: {e}")
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        try:
            mlflow.end_run()
            logger.info("Ended MLflow run")
        except Exception as e:
            logger.warning(f"Failed to end MLflow run: {e}")
    
    def _generate_run_suffix(self) -> str:
        """Generate a unique suffix for run names."""
        import datetime
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def log_training_results(results: Dict[str, Any], model_path: str) -> None:
    """Convenience function to log training results."""
    tracker = MLflowTracker()
    
    with tracker.start_run():
        # Log parameters
        if "training_params" in results:
            tracker.log_params(results["training_params"])
        
        # Log metrics
        if "training_stats" in results:
            stats = results["training_stats"]
            metrics = {
                "final_loss": stats.get("eval_loss", 0),
                "train_samples": stats.get("num_train_samples", 0),
                "eval_samples": stats.get("num_eval_samples", 0),
                "training_time": stats.get("training_time", 0),
            }
            tracker.log_metrics(metrics)
        
        # Log model
        if os.path.exists(model_path):
            tracker.log_model(model_path)
        
        # Log metadata
        metadata_path = os.path.join(model_path, "training_metadata.json")
        if os.path.exists(metadata_path):
            tracker.log_artifacts(metadata_path, "metadata")