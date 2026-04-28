"""
Gestion du tracking MLflow et métriques
"""
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from .config import config
import logging

logger = logging.getLogger(__name__)

class MLflowTracker:
    """Gestionnaire centralisé pour MLflow tracking"""
    
    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or config.experiment_name
        self.client = MlflowClient()
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Configure MLflow"""
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=f"{config.mlflow_tracking_uri}/artifacts/{self.experiment_name}"
                )
                logger.info(f"Expérience créée: {self.experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Expérience existante: {self.experiment_name} (ID: {experiment_id})")
        except Exception as e:
            logger.error(f"Erreur configuration MLflow: {e}")
            experiment_id = "0"
        
        mlflow.set_experiment(experiment_id=experiment_id)
        return experiment_id
    
    def log_parameters(self, params: dict):
        """Log des paramètres"""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: dict):
        """Log des métriques"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
    
    def log_model(self, model, artifact_path: str = "model", 
                  registered_model_name: str = None, **kwargs):
        """Log et enregistrement du modèle"""
        model_name = registered_model_name or config.model_name
        
        return mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=model_name,
            **kwargs
        )
    
    def set_tags(self, tags: dict):
        """Définir des tags pour le run"""
        for key, value in tags.items():
            mlflow.set_tag(key, value)
    
    def get_best_model(self, metric_name: str = "test_accuracy", stage: str = "Production"):
        """Obtenir le meilleur modèle basé sur une métrique"""
        versions = self.client.get_latest_versions(config.model_name, stages=[stage])
        
        if not versions:
            return None
            
        best_version = None
        best_score = -1
        
        for version in versions:
            run = self.client.get_run(version.run_id)
            score = run.data.metrics.get(metric_name, 0)
            
            if score > best_score:
                best_score = score
                best_version = version
        
        return best_version, best_score

class ModelRegistry:
    """Gestionnaire du registry de modèles"""
    
    def __init__(self):
        self.client = MlflowClient()
    
    def promote_to_production(self, model_name: str = None, 
                            min_accuracy: float = None) -> bool:
        """Promotion automatique vers Production"""
        model_name = model_name or config.model_name
        min_accuracy = min_accuracy or config.min_accuracy_production
        
        # Obtenir la dernière version
        latest_versions = self.client.get_latest_versions(
            model_name, 
            stages=["Staging", "None"]
        )
        
        if not latest_versions:
            logger.warning("Aucune version du modèle trouvée")
            return False
        
        latest_version = latest_versions[0]
        
        # Vérifier les métriques
        run = self.client.get_run(latest_version.run_id)
        test_accuracy = run.data.metrics.get('test_accuracy', 0)
        
        logger.info(f"Version {latest_version.version}: Test Accuracy = {test_accuracy:.4f}")
        
        if test_accuracy >= min_accuracy:
            # Promouvoir vers Production
            self.client.transition_model_version_stage(
                name=model_name,
                version=latest_version.version,
                stage="Production",
                archive_existing_versions=True
            )
            
            logger.info(f"Modèle version {latest_version.version} promu vers Production!")
            return True
        else:
            logger.warning(f"Accuracy {test_accuracy:.4f} < {min_accuracy:.4f}, promotion refusée")
            return False
    
    def get_production_model_uri(self, model_name: str = None) -> str:
        """Obtenir l'URI du modèle en Production"""
        model_name = model_name or config.model_name
        return f"models:/{model_name}/Production"