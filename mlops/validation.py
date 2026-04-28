"""
Validation des données et des modèles pour garantir la qualité
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validation des données d'entraînement"""
    
    def __init__(self, min_samples: int = 1000, min_toxic_ratio: float = 0.05, max_toxic_ratio: float = 0.95):
        self.min_samples = min_samples
        self.min_toxic_ratio = min_toxic_ratio
        self.max_toxic_ratio = max_toxic_ratio
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Valide un DataFrame de données
        
        Returns:
            (is_valid, errors)
        """
        errors = []
        
        # 1. Vérifier les colonnes requises
        required_columns = ['comment_text', 'toxic']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Colonnes manquantes: {missing_columns}")
        
        # 2. Vérifier le nombre de samples
        if len(df) < self.min_samples:
            errors.append(f"Pas assez de données: {len(df)} < {self.min_samples}")
        
        # 3. Vérifier les valeurs manquantes
        if df['comment_text'].isnull().any():
            null_count = df['comment_text'].isnull().sum()
            errors.append(f"{null_count} valeurs manquantes dans comment_text")
        
        if df['toxic'].isnull().any():
            null_count = df['toxic'].isnull().sum()
            errors.append(f"{null_count} valeurs manquantes dans toxic")
        
        # 4. Vérifier le type des labels
        if not df['toxic'].dtype in [np.int64, np.int32, int, bool]:
            errors.append(f"Type de label incorrect: {df['toxic'].dtype}")
        
        # 5. Vérifier les valeurs des labels
        unique_labels = df['toxic'].unique()
        if not set(unique_labels).issubset({0, 1}):
            errors.append(f"Valeurs de labels invalides: {unique_labels}")
        
        # 6. Vérifier la distribution des classes
        toxic_ratio = df['toxic'].mean()
        if toxic_ratio < self.min_toxic_ratio:
            errors.append(f"Trop peu de cas toxiques: {toxic_ratio:.2%} < {self.min_toxic_ratio:.2%}")
        
        if toxic_ratio > self.max_toxic_ratio:
            errors.append(f"Trop de cas toxiques: {toxic_ratio:.2%} > {self.max_toxic_ratio:.2%}")
        
        # 7. Vérifier la longueur des textes
        text_lengths = df['comment_text'].astype(str).str.len()
        if (text_lengths == 0).any():
            empty_count = (text_lengths == 0).sum()
            errors.append(f"{empty_count} textes vides trouvés")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info(f" Validation des données réussie: {len(df)} échantillons")
        else:
            logger.error(f" Validation des données échouée: {len(errors)} erreurs")
            for error in errors:
                logger.error(f"  - {error}")
        
        return is_valid, errors


class ModelValidator:
    """Validation des performances du modèle"""
    
    def __init__(
        self,
        min_accuracy: float = 0.85,
        min_precision: float = 0.80,
        min_recall: float = 0.70,
        min_f1: float = 0.75
    ):
        self.min_accuracy = min_accuracy
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.min_f1 = min_f1
    
    def validate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[bool, Dict[str, float], List[str]]:
        """
        Valide les métriques du modèle
        
        Returns:
            (is_valid, metrics, errors)
        """
        errors = []
        
        # Calculer les métriques
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Vérifier les seuils
        if metrics['accuracy'] < self.min_accuracy:
            errors.append(f"Accuracy trop faible: {metrics['accuracy']:.4f} < {self.min_accuracy}")
        
        if metrics['precision'] < self.min_precision:
            errors.append(f"Precision trop faible: {metrics['precision']:.4f} < {self.min_precision}")
        
        if metrics['recall'] < self.min_recall:
            errors.append(f"Recall trop faible: {metrics['recall']:.4f} < {self.min_recall}")
        
        if metrics['f1'] < self.min_f1:
            errors.append(f"F1-Score trop faible: {metrics['f1']:.4f} < {self.min_f1}")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info(f" Validation du modèle réussie:")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")
        else:
            logger.error(f" Validation du modèle échouée: {len(errors)} erreurs")
            for error in errors:
                logger.error(f"  - {error}")
        
        return is_valid, metrics, errors
    
    def compare_with_baseline(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        min_improvement: float = 0.00
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Compare les métriques avec une baseline
        
        Returns:
            (is_better, improvements)
        """
        improvements = {}
        
        for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
            if metric_name in current_metrics and metric_name in baseline_metrics:
                improvement = current_metrics[metric_name] - baseline_metrics[metric_name]
                improvements[metric_name] = improvement
        
        # Le modèle est meilleur si au moins une métrique s'améliore sans régression majeure
        has_improvement = any(imp > min_improvement for imp in improvements.values())
        no_regression = all(imp > -0.05 for imp in improvements.values())  # Max 5% de régression
        
        is_better = has_improvement and no_regression
        
        if is_better:
            logger.info(f" Nouveau modèle meilleur que la baseline")
            for metric_name, improvement in improvements.items():
                logger.info(f"  {metric_name}: {improvement:+.4f}")
        else:
            logger.warning(f"  Nouveau modèle pas meilleur que la baseline")
        
        return is_better, improvements


class DriftDetector:
    """Détection de drift dans les données"""
    
    def detect_distribution_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        threshold: float = 0.1
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Détecte un drift dans la distribution des classes
        
        Returns:
            (has_drift, drift_metrics)
        """
        ref_toxic_ratio = reference_data['toxic'].mean()
        curr_toxic_ratio = current_data['toxic'].mean()
        
        drift = abs(curr_toxic_ratio - ref_toxic_ratio)
        has_drift = drift > threshold
        
        drift_metrics = {
            'reference_toxic_ratio': ref_toxic_ratio,
            'current_toxic_ratio': curr_toxic_ratio,
            'drift': drift,
            'has_drift': has_drift
        }
        
        if has_drift:
            logger.warning(f"  Drift détecté dans la distribution des classes: {drift:.4f}")
        else:
            logger.info(f" Pas de drift détecté: {drift:.4f} < {threshold}")
        
        return has_drift, drift_metrics


# Fonctions utilitaires
def validate_training_data(df: pd.DataFrame) -> bool:
    """Valide les données d'entraînement"""
    validator = DataValidator()
    is_valid, errors = validator.validate_dataframe(df)
    return is_valid


def validate_model_performance(y_true: np.ndarray, y_pred: np.ndarray) -> bool:
    """Valide les performances du modèle"""
    validator = ModelValidator()
    is_valid, metrics, errors = validator.validate_metrics(y_true, y_pred)
    return is_valid
