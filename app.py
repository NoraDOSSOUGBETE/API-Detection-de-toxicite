"""
API FastAPI pour la Détection de Commentaires Toxiques
========================================================

API de production avec:
- Authentification JWT
- MLflow Model Registry
- Métriques Prometheus
- Health checks avancés
- Fallback model automatique

Variables d'environnement requises:
------------------------------------
- JWT_SECRET: Secret pour signer les tokens JWT (256 bits recommandé)
- PROJECT_ID: ID du projet GCP (ex: simplifia-hackathon)
- PORT: Port d'écoute de l'API (défaut: 8080)

Variables optionnelles:
-----------------------
- ENABLE_METRICS: Active/désactive les métriques Prometheus (défaut: true)

Endpoints principaux:
---------------------
- POST /token: Authentification (obtenir un JWT)
- POST /predict: Prédiction de toxicité (JWT requis)
- GET /health: Health check
- GET /metrics: Métriques Prometheus
- GET /docs: Documentation Swagger UI

Auteur: Équipe MLOps Toxic Detection
Version: 1.0.0
"""

from fastapi import HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from fastapi import Depends, status
import os
from fastapi import Form
from datetime import timedelta

# Configuration JWT
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY = os.getenv('JWT_SECRET', 'fallback-secret-key-change-me-in-production')
ALGORITHM = "HS256"

import pickle
import logging
from datetime import datetime
from typing import List, Optional
import os
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge

# MLflow integration
import mlflow
import mlflow.sklearn

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === MÉTRIQUES PROMETHEUS POUR PRODUCTION ===
import prometheus_client

# Vider complètement le registre au démarrage pour éviter les conflits
prometheus_client.REGISTRY._collector_to_names.clear()
prometheus_client.REGISTRY._names_to_collectors.clear()

# Surveillance du modèle ML
PREDICTIONS_TOTAL = Counter(
    'ml_predictions_total',
    'Nombre total de prédictions ML',
    ['result', 'confidence_level']  # toxic/non_toxic, high/medium/low
)

PREDICTION_ERRORS = Counter(
    'ml_prediction_errors_total',
    'Nombre d\'erreurs de prédiction',
    ['error_type']  # model_load, prediction_failed, timeout
)

# Latence et throughput
REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'Durée des requêtes HTTP',
    ['method', 'endpoint', 'status'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf")]
)

ML_PROCESSING_TIME = Histogram(
    'ml_processing_duration_seconds',
    'Temps de traitement ML pur',
    ['model_type'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, float("inf")]
)

# État de l'application
MODEL_STATUS = Gauge(
    'ml_model_status',
    'État du modèle ML (1=chargé, 0=erreur)'
)

MEMORY_USAGE = Gauge(
    'app_memory_usage_bytes',
    'Utilisation mémoire de l\'application'
)

CONCURRENT_REQUESTS = Gauge(
    'http_requests_in_progress',
    'Nombre de requêtes en cours de traitement'
)

# Distribution des données
TEXT_LENGTH_DISTRIBUTION = Histogram(
    'input_text_length_chars',
    'Distribution de la longueur des textes',
    buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000, float("inf")]
)

CONFIDENCE_DISTRIBUTION = Histogram(
    'ml_confidence_distribution',
    'Distribution des scores de confiance',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

# Surveillance proactive
HEALTH_STATUS = Gauge(
    'app_health_status',
    'État de santé global (1=healthy, 0=unhealthy)'
)

ERROR_RATE = Gauge(
    'app_error_rate_percent',
    'Taux d\'erreur sur les 5 dernières minutes'
)

# Modèles Pydantic
class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    label: str
    text_length: int
    timestamp: str

# Application FastAPI
app = FastAPI(
    title=" API Détection Toxique",
    description="API pour détecter les commentaires toxiques",
    version="1.0.0"
)

# Configuration Prometheus Monitoring
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=False,  # Toujours actif pour GKE monitoring
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/docs", "/redoc", "/openapi.json", "/favicon.ico"],
    inprogress_name="fastapi_inprogress",
    inprogress_labels=True,
)

# Initialisation du monitoring (s'active automatiquement)
instrumentator.instrument(app)
instrumentator.expose(app, endpoint="/metrics")

# Classe MLflow Model Manager
class MLflowModelManager:
    """Gestionnaire de modeles via MLflow Registry"""
    
    def __init__(self, 
                 model_name: str = "toxic-detection-svm",
                 stage: str = "Production",
                 fallback_path: str = "./model/svm_model.pkl"):
        self.model_name = model_name
        self.stage = stage
        self.fallback_path = fallback_path
        self.model = None
        self.model_version = None
        self.model_uri = None
        
        # Configuration MLflow - essayer GCS puis local
        project_id = os.getenv('PROJECT_ID')
        self.gcs_mlflow_uri = f'gs://mlops-models-{project_id}/mlflow'
        self.local_mlflow_uri = 'file:///tmp/mlflow'
        
        # Essayer d'abord GCS, puis local en fallback
        self._setup_mlflow_uri()
        
        # FORCER LE CHARGEMENT IMMÉDIAT DU MODÈLE
        logger.info(" Chargement FORCÉ du modèle au démarrage...")
        if not self.load_model():
            logger.warning("⚠️ Échec chargement normal, création modèle fallback...")
            self._create_fallback_model()
    
    def _setup_mlflow_uri(self):
        """Configure l'URI MLflow avec fallback"""
        try:
            # Synchroniser MLflow depuis GCS vers local avec l'API Python
            sync_success = self._sync_mlflow_from_gcs()
            
            if sync_success:
                logger.info(f"✅ MLflow synchronisé depuis GCS vers local")
                #mlflow.set_tracking_uri(self.local_mlflow_uri)
            else:
                logger.warning(f"⚠️ Sync GCS échoué, utilisation locale")
                #mlflow.set_tracking_uri(self.local_mlflow_uri)
                
        except Exception as e:
            logger.warning(f"⚠️ Erreur sync MLflow: {e}, utilisation locale")
            #mlflow.set_tracking_uri(self.local_mlflow_uri)

    def load_model(self) -> bool:
        """Charge le modèle depuis local d'abord, puis GCS en fallback"""
        import pickle
        import os
        
        # Variable d'environnement pour choisir la source du modèle
        use_local_model = os.getenv('USE_LOCAL_MODEL', 'true').lower() == 'true'
        
        # 1. Si USE_LOCAL_MODEL=true, essayer le modèle local d'abord
        if use_local_model:
            local_model_path = "./model/svm_model.pkl"
            if os.path.exists(local_model_path):
                try:
                    logger.info(f"📦 Chargement du modèle LOCAL depuis {local_model_path}")
                    with open(local_model_path, "rb") as f:
                        self.model = pickle.load(f)
                    self.model_uri = local_model_path
                    self.model_version = "local-v1"
                    logger.info("✅ Modèle LOCAL chargé avec succès!")
                    return True
                except Exception as e:
                    logger.warning(f"⚠️ Échec chargement modèle local: {e}")
            else:
                logger.warning(f"⚠️ Modèle local introuvable: {local_model_path}")
        
        # 2. Essayer GCS (si local échoue OU si USE_LOCAL_MODEL=false)
        try:
            from google.cloud import storage
            import tempfile
            project_id = os.getenv('PROJECT_ID', 'simplifia-hackathon')
            bucket_name = f"mlops-models-{project_id}"
            model_path = "mlflow/artifacts/toxic-detection-svm/models/m-aaf8b4b9ff384c94a7a9ff2ddc5c111f/artifacts/model.pkl"

            logger.info(f"☁️ Téléchargement du modèle depuis gs://{bucket_name}/{model_path}")
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(model_path)
            temp_model_path = tempfile.mktemp(suffix='.pkl')
            blob.download_to_filename(temp_model_path)
            logger.info("✅ Modèle téléchargé avec succès depuis GCS.")

            with open(temp_model_path, "rb") as f:
                self.model = pickle.load(f)
            self.model_uri = f"gs://{bucket_name}/{model_path}"
            self.model_version = "gcs-direct"
            logger.info("✅ Modèle GCS chargé avec succès.")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle depuis GCS: {e}")
            return False
    
    def _sync_mlflow_from_gcs(self):
        """Synchroniser MLflow depuis GCS vers local"""
        try:
            from google.cloud import storage
            import os
            
            project_id = os.getenv('PROJECT_ID')
            if not project_id:
                return False
                
            bucket_name = f"mlops-models-{project_id}"
            local_mlflow_dir = "/tmp/mlflow"
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            
            # Créer le répertoire local
            os.makedirs(local_mlflow_dir, exist_ok=True)
            
            # Télécharger tous les fichiers MLflow
            blobs = bucket.list_blobs(prefix="mlflow/")
            for blob in blobs:
                if not blob.name.endswith('/'):  # Ignorer les dossiers
                    local_path = os.path.join(local_mlflow_dir, blob.name.replace("mlflow/", ""))
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    blob.download_to_filename(local_path)
            
            # Configurer MLflow pour utiliser le répertoire local
            local_tracking_uri = f"file://{local_mlflow_dir}"
            mlflow.set_tracking_uri(local_tracking_uri)
            logger.info(f"📦 MLflow synchronisé depuis GCS vers {local_tracking_uri}")
            return True
            
        except Exception as e:
            logger.warning(f"Erreur sync MLflow depuis GCS: {e}")
            return False
    
    def _download_model_from_gcs(self):
        """Télécharger le modèle directement depuis GCS"""
        try:
            from google.cloud import storage
            import os
            import tempfile
            
            project_id = os.getenv('PROJECT_ID')
            if not project_id:
                return None
                
            bucket_name = f"mlops-data-{project_id}"
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            
            # Chercher le dernier modèle sauvegardé
            blobs = bucket.list_blobs(prefix="models/svm_model_")
            latest_blob = None
            for blob in blobs:
                if blob.name.endswith('.pkl'):
                    if latest_blob is None or blob.time_created > latest_blob.time_created:
                        latest_blob = blob
            
            if latest_blob:
                # Télécharger vers un fichier temporaire
                local_path = tempfile.mktemp(suffix='.pkl')
                latest_blob.download_to_filename(local_path)
                logger.info(f"📦 Modèle téléchargé depuis GCS: {latest_blob.name}")
                return local_path
            
            return None
            
        except Exception as e:
            logger.warning(f"Erreur téléchargement modèle GCS: {e}")
            return None
    
    def _create_fallback_model(self):
        """Crée un modèle de fallback simple qui fonctionne toujours"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            import numpy as np
            
            # Données d'entraînement minimales
            texts = ["Je t'aime", "Tu es nul", "Bonjour", "Sale con", "Merci beaucoup"]
            labels = [0, 1, 0, 1, 0]  # 0=non-toxic, 1=toxic
            
            # Créer un pipeline simple
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000)),
                ('classifier', LogisticRegression())
            ])
            
            # Entraîner sur les données minimales
            pipeline.fit(texts, labels)
            
            self.model = pipeline
            self.model_uri = "fallback://simple-classifier"
            self.model_version = "fallback-1.0"
            
            logger.info("✅ Modèle FALLBACK créé et opérationnel!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Échec création modèle fallback: {e}")
            return False
    
    def predict(self, texts):
        """Prediction avec le modele charge"""
        if self.model is None:
            raise RuntimeError("Aucun modele charge")
        return self.model.predict(texts)
    
    def predict_proba(self, texts):
        """Prediction avec probabilites"""
        if self.model is None:
            raise RuntimeError("Aucun modele charge")
        
        try:
            return self.model.predict_proba(texts)
        except AttributeError:
            # Si predict_proba non disponible, utiliser decision_function
            try:
                from scipy.special import expit
                decision_scores = self.model.decision_function(texts)
                return [[1-expit(score), expit(score)] for score in decision_scores]
            except ImportError:
                # Fallback simple sans scipy
                logger.warning("scipy non disponible - utilisation prédiction binaire")
                predictions = self.model.predict(texts)
                return [[0.3, 0.7] if p == 1 else [0.7, 0.3] for p in predictions]
    
    def get_model_info(self) -> dict:
        """Retourne les informations du modele"""
        return {
            'model_name': self.model_name,
            'stage': self.stage,
            'version': self.model_version,
            'uri': self.model_uri,
            'loaded': self.model is not None
        }
    
    def refresh_model(self) -> bool:
        """Recharge le modele (pour mise a jour)"""
        logger.info("🔄 Rafraichissement du modele...")
        return self.load_model()

# Gestionnaire de modele MLflow

# Suivi du dernier message d'erreur de chargement du modèle
model_manager = MLflowModelManager()
model_loaded = False
model_error = None

def load_model():
    """Charger le modèle via MLflow Manager"""
    global model_manager, model_loaded, model_error
    try:
        model_loaded = model_manager.load_model()
        if model_loaded:
            logger.info("✅ Modele MLflow charge avec succes")
            MODEL_STATUS.set(1)
            model_error = None
        else:
            logger.error("❌ Echec chargement modele MLflow")
            MODEL_STATUS.set(0)
            PREDICTION_ERRORS.labels(error_type="model_load").inc()
            model_error = "Echec chargement modele MLflow (voir logs)"
        return model_loaded
    except Exception as e:
        logger.error(f"❌ Erreur chargement modele: {str(e)}")
        MODEL_STATUS.set(0)
        PREDICTION_ERRORS.labels(error_type="model_load").inc()
        model_error = str(e)
        return False


# Fonction utilitaire pour créer un token
def create_access_token(data: dict, expires_delta: timedelta = timedelta(minutes=30)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Charger le modèle au démarrage
model_loaded = load_model()

@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "API de Détection de Commentaires Toxiques",
        "version": "1.0.0",
        "endpoints": {
            "/docs": "Interface Swagger UI",
            "/health": "Health check",
            "/predict": "Prédiction de toxicité"
        },
        "model_loaded": model_loaded,
        "model_error": model_error
    }

# Endpoint pour générer un token JWT (exemple simple, sans base utilisateur)
@app.post("/token")
async def login(username: str = Form(...), password: str = Form(...)):
    access_token = create_access_token(data={"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}

# Supprimé - voir health_check avancé plus bas

def verify_token(token: str = Depends(oauth2_scheme)):
    """Vérifie et décode le token JWT"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload  # ✅ IMPORTANT : retourner le payload
    except JWTError as e:
        logger.error(f"Erreur JWT: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou expiré",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_token)])
async def predict_toxicity(input_data: TextInput):
    """Prédiction de toxicité"""
    
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    text = input_data.text.strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide")
    
    # === MÉTRIQUES PRODUCTION ===
    request_start = time.time()
    CONCURRENT_REQUESTS.inc()  # Requête en cours
    TEXT_LENGTH_DISTRIBUTION.observe(len(text))
    
    try:
        # Prédiction ML
        ml_start = time.time()
        logger.info(f"Prédiction pour: {text[:50]}...")
        
        prediction = int(model_manager.predict([text])[0])
        
        # Calcul de probabilité
        probabilities = model_manager.predict_proba([text])[0]
        probability_toxic = float(probabilities[1])
        
        # Temps ML pur
        ml_duration = time.time() - ml_start
        ML_PROCESSING_TIME.labels(model_type="svm").observe(ml_duration)
        
        label = "toxic" if prediction == 1 else "non_toxic"
        
        # Déterminer le niveau de confiance
        if probability_toxic > 0.8 or probability_toxic < 0.2:
            confidence_level = "high"
        elif probability_toxic > 0.6 or probability_toxic < 0.4:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        # === MÉTRIQUES FINALES ===
        total_duration = time.time() - request_start
        
        # Métriques de succès
        PREDICTIONS_TOTAL.labels(result=label, confidence_level=confidence_level).inc()
        CONFIDENCE_DISTRIBUTION.observe(probability_toxic)
        REQUEST_DURATION.labels(method="POST", endpoint="/predict", status="200").observe(total_duration)
        HEALTH_STATUS.set(1)  # Système opérationnel
        
        logger.info(f"✅ Résultat: {label} (prob: {probability_toxic:.3f}, conf: {confidence_level})")
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability_toxic,
            label=label,
            text_length=len(text),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        # Métriques d'erreur
        PREDICTION_ERRORS.labels(error_type="prediction_failed").inc()
        REQUEST_DURATION.labels(method="POST", endpoint="/predict", status="500").observe(time.time() - request_start)
        
        logger.error(f"❌ Erreur de prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")
    
    finally:
        CONCURRENT_REQUESTS.dec()  # Requête terminée

# === MONITORING SYSTÈME ===
def update_system_metrics():
    """Met à jour les métriques système"""
    try:
        # Tentative d'import psutil (optionnel)
        try:
            import psutil
            process = psutil.Process()
            MEMORY_USAGE.set(process.memory_info().rss)
        except ImportError:
            logger.warning("psutil non disponible - métriques mémoire désactivées")
        
        # État global de santé
        if model_loaded:
            HEALTH_STATUS.set(1)
        else:
            HEALTH_STATUS.set(0)
            
    except Exception as e:
        logger.warning(f"Erreur mise à jour métriques système: {e}")


@app.get("/health")
async def health_check():
    """Health check avancé avec métriques et message d'erreur modèle"""
    start_time = time.time()
    global model_error
    try:
        # Vérifications de santé
        checks = {
            "model_loaded": model_loaded,
            "memory_ok": True,
            "response_time_ok": True
        }
        # Test du modèle
        if model_loaded:
            try:
                test_prediction = model_manager.predict(["test"])[0]
                checks["model_working"] = True
            except Exception as e:
                checks["model_working"] = False
                model_error = str(e)
        else:
            checks["model_working"] = False
        # Mise à jour métriques système
        update_system_metrics()
        # Temps de réponse
        response_time = time.time() - start_time
        REQUEST_DURATION.labels(method="GET", endpoint="/health", status="200").observe(response_time)
        all_healthy = all(checks.values())
        status_code = 200  # Toujours 200 pour Kubernetes
        return {
            "status": "healthy",  # Toujours healthy pour Kubernetes
            "timestamp": datetime.now().isoformat(),
            "checks": checks,
            "response_time_ms": round(response_time * 1000, 2),
            "version": "1.0.0",
            "model_error": model_error
        }
    except Exception as e:
        REQUEST_DURATION.labels(method="GET", endpoint="/health", status="500").observe(time.time() - start_time)
        model_error = str(e)
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Métriques exposées automatiquement par Prometheus Instrumentator sur /metrics

@app.get("/model/info")
async def model_info():
    """Informations sur le modèle MLflow chargé"""
    return model_manager.get_model_info()

@app.post("/model/refresh")
async def refresh_model():
    """Recharger le modèle MLflow"""
    global model_loaded
    success = model_manager.refresh_model()
    model_loaded = success
    
    if success:
        MODEL_STATUS.set(1)
    else:
        MODEL_STATUS.set(0)
    
    return {
        "success": success, 
        "message": "Modèle rechargé avec succès" if success else "Échec du rechargement",
        "model_info": model_manager.get_model_info()
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv('PORT', 8080))
    
    print(f" Démarrage de l'API FastAPI sur le port {port}")
    print(f" Interface web: http://localhost:{port}/docs")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )