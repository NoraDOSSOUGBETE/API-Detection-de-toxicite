"""
Configuration centralisée pour le pipeline MLOps
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv('.env.local')

@dataclass
class MLOpsConfig:
    # Google Cloud
    project_id: str = field(default_factory=lambda: os.getenv('PROJECT_ID', 'default'))
    region: str = field(default_factory=lambda: os.getenv('REGION', 'europe-west1'))

    # Cloud Storage
    bucket_models: str = field(default_factory=lambda: f"mlops-models-{os.getenv('PROJECT_ID', 'default')}")
    bucket_data: str = field(default_factory=lambda: f"mlops-data-{os.getenv('PROJECT_ID', 'default')}")

    # MLflow
    mlflow_tracking_uri: str = field(default_factory=lambda: f"gs://mlops-models-{os.getenv('PROJECT_ID', 'default')}/mlflow")
    experiment_name: str = "toxic-detection-svm"

    # Vertex AI
    vertex_pipeline_root: str = field(default_factory=lambda: f"gs://mlops-models-{os.getenv('PROJECT_ID', 'default')}/pipeline-root")

    # Modèle
    model_name: str = "toxic-detection-svm"
    min_accuracy_production: float = 0.85
    
    # Entraînement
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5

    # Données
    data_sources: List[str] = field(default_factory=lambda: [
        "data/train_toxic_10k.csv",
        "data/dataset_cleaned_and_anonymized_10k.csv"
    ])
    
    # Sécurité
    enable_model_validation: bool = True
    enable_data_validation: bool = True
    
    # Monitoring
    enable_prometheus_metrics: bool = True
    metrics_port: int = 8080

# Configuration par défaut
config = MLOpsConfig()