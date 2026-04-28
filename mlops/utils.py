"""
Utilitaires partagés pour le pipeline MLOps
"""
import pandas as pd
from google.cloud import storage
import io
import tempfile
import pickle
import os
from .config import config
import logging

logger = logging.getLogger(__name__)

class CloudStorageManager:
    """Gestionnaire pour Cloud Storage"""
    
    def __init__(self, project_id: str = None):
        self.project_id = project_id or config.project_id
        self.client = storage.Client(project=self.project_id)
    
    def load_dataframe(self, bucket_name: str, blob_path: str) -> pd.DataFrame:
        """Charger un DataFrame depuis Cloud Storage"""
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        content = blob.download_as_text()
        return pd.read_csv(io.StringIO(content))
    
    def save_dataframe(self, df: pd.DataFrame, bucket_name: str, blob_path: str):
        """Sauvegarder un DataFrame vers Cloud Storage"""
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        csv_content = df.to_csv(index=False)
        blob.upload_from_string(csv_content)
        
        logger.info(f"DataFrame sauvegardé: gs://{bucket_name}/{blob_path}")
    
    def load_model(self, bucket_name: str, model_path: str):
        """Charger un modèle depuis Cloud Storage"""
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(model_path)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            blob.download_to_filename(temp_file.name)
            
            with open(temp_file.name, 'rb') as f:
                model = pickle.load(f)
            
            os.unlink(temp_file.name)
            return model
    
    def save_model(self, model, bucket_name: str, model_path: str) -> str:
        """Sauvegarder un modèle vers Cloud Storage"""
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(model_path)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            pickle.dump(model, temp_file)
            temp_file.flush()
            
            blob.upload_from_filename(temp_file.name)
            os.unlink(temp_file.name)
        
        gcs_path = f"gs://{bucket_name}/{model_path}"
        logger.info(f"Modèle sauvegardé: {gcs_path}")
        return gcs_path

class DataLoader:
    """Chargeur de données avec fallback automatique"""
    
    def __init__(self, storage_manager: CloudStorageManager = None):
        self.storage = storage_manager or CloudStorageManager()
    
    def load_training_data(self, bucket_name: str = None) -> pd.DataFrame:
        """Charger les données d'entraînement avec fallback"""
        bucket_name = bucket_name or config.bucket_data
        
        for source in config.data_sources:
            try:
                df = self.storage.load_dataframe(bucket_name, source)
                logger.info(f"Données chargées depuis gs://{bucket_name}/{source}")
                
                # Normaliser les colonnes
                df = self._normalize_columns(df)
                return df
                
            except Exception as e:
                logger.warning(f"Impossible de charger {source}: {e}")
                continue
        
        raise ValueError("Aucune source de données trouvée dans Cloud Storage")
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliser les noms de colonnes"""
        if 'comment_text' in df.columns and 'toxic' in df.columns:
            return df
        
        # Mapping automatique des colonnes
        column_mapping = {}
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['text', 'comment', 'message']):
                column_mapping[col] = 'comment_text'
            elif any(keyword in col.lower() for keyword in ['toxic', 'label', 'target']):
                column_mapping[col] = 'toxic'
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
            logger.info(f"Colonnes renommées: {column_mapping}")
        
        return df

def validate_data(df: pd.DataFrame) -> bool:
    """Valider les données d'entraînement"""
    required_columns = ['comment_text', 'toxic']
    
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Colonne manquante: {col}")
            return False
    
    # Vérifier les valeurs nulles
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        logger.warning(f"Valeurs nulles détectées: {null_counts.to_dict()}")
    
    # Vérifier la distribution des classes
    if 'toxic' in df.columns:
        toxic_ratio = df['toxic'].mean()
        logger.info(f"Distribution toxic/non-toxic: {toxic_ratio:.3f}/{1-toxic_ratio:.3f}")
        
        if toxic_ratio < 0.01 or toxic_ratio > 0.99:
            logger.warning("Distribution des classes déséquilibrée")
    
    return True