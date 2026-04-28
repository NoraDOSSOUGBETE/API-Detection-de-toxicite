# API Toxic Detection - MLOps Production

> Système de détection de commentaires toxiques avec MLOps complet : CI/CD automatisé, monitoring en production, conformité RGPD, et retraining automatique sur Vertex AI.

## Vue d'Ensemble

Ce projet implémente une API de détection de toxicité dans les commentaires avec une infrastructure MLOps complète sur Google Cloud Platform. L'API utilise un modèle SVM entraîné sur le dataset Toxic Comment Classification de Hugging Face, avec anonymisation RGPD des données personnelles.

**API en Production :** http://34.22.130.34 | **Documentation Interactive :** http://34.22.130.34/docs

## Démarrage Rapide

```bash
# 1. Clone et installation
git clone https://github.com/Khadija0203/API-Digital-Social-Score.git
cd API-Digital-Social-Score
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Lancer l'API en local
python app.py

# 3. Tester l'API
curl http://localhost:8080/health
```

**Accès à l'API locale :** http://localhost:8080/docs

## Fonctionnalités Principales

- **Pipeline CI/CD Automatisé** : Déploiement continu sur GKE via Cloud Build
- **Retraining Automatique** : Pipeline Vertex AI déclenché par Pub/Sub sur upload GCS
- **MLflow Tracking** : Suivi des expériences et Model Registry
- **Kubernetes (GKE)** : Déploiement avec 3 replicas et auto-scaling
- **Monitoring** : Cloud Monitoring et Cloud Logging intégrés
- **Sécurité** : Authentification JWT, IAM, et Service Accounts
- **Conformité RGPD** : Anonymisation des données avec spaCy NER
- **Load Testing** : Tests de charge avec Locust et métriques de performance

## Architecture

```
Code Push (GitHub)
    ↓
Cloud Build (CI/CD automatique)
    ├── Tests unitaires
    ├── Build Docker
    └── Deploy GKE
         ↓
    API Production (3 replicas)

Nouvelles données (GCS)
    ↓
Pub/Sub notification
    ↓
Vertex AI Pipeline
    ├── Anonymisation RGPD
    ├── Entraînement SVM
    └── MLflow Registry
         ↓
    Nouveau modèle disponible
```

## Utilisation de l'API

### Authentification

```bash
# Obtenir un token JWT
curl -X POST http://34.22.130.34/token \
  -d "username=admin&password=admin"
```

### Prédiction

```bash
# Analyser un commentaire
curl -X POST http://34.22.130.34/predict \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test comment"}'

# Réponse
{
  "text": "This is a test comment",
  "is_toxic": false,
  "score": 15
}
```

### Endpoints Disponibles

- `POST /token` - Authentification JWT
- `POST /predict` - Prédiction de toxicité
- `GET /health` - Health check
- `GET /metrics` - Métriques Prometheus
- `GET /docs` - Documentation interactive Swagger

## Documentation Détaillée

La documentation complète est disponible dans le dossier [`docs/`](./docs/) :

- **[Architecture MLOps](./docs/MLOPS_ARCHITECTURE.md)** - Architecture complète, composants, flux de données
- **[Guide de Déploiement](./docs/DEPLOYMENT.md)** - Déploiement sur GCP et GKE pas à pas
- **[Réentraînement Automatique](./docs/RETRAINING.md)** - Configuration du pipeline de retraining
- **[Monitoring & Métriques](./docs/MONITORING.md)** - Justification des 11 métriques MLOps, dashboards Grafana
- **[Accès Grafana](./GRAFANA_ACCESS.md)** - Guide d'accès et import des dashboards

## Structure du Projet

```
API-Digital-Social-Score/
├── app.py                           # API FastAPI
├── auth.py                          # Authentification JWT
├── cloudbuild.yaml                  # Pipeline CI/CD
├── cloudbuild-retrain.yaml          # Pipeline retraining
├── Dockerfile                       # Image Docker
├── vertex.py                        # Pipeline Vertex AI
├── locustfile.py                    # Tests de charge
│
├── mlops/                           # Pipeline MLOps
│   ├── training.py                  # Entraînement + MLflow
│   ├── validation.py                # Validation données/modèle
│   ├── config.py                    # Configuration
│   └── utils.py                     # Utilitaires
│
├── model/                           # Modèles ML
│   ├── SVM.py                       # Modèle SVM
│   └── BERT.py                      # Modèle BERT (optionnel)
│
├── grafana/                         # Dashboards Grafana
│   └── dashboards/
│       ├── dashboard-business.json       # Dashboard Business & Utilisation
│       ├── dashboard-performance.json    # Dashboard Performance & Model Health
│       └── dashboard-infrastructure.json # Dashboard Infrastructure & Ressources
│
├── prometheus/                      # Configuration Prometheus
│   └── prometheus.yml
│
├── k8s/                             # Manifests Kubernetes
│   ├── deployment-mlops.yaml        # Déploiement API
│   ├── prometheus-deployment.yaml   # Déploiement Prometheus
│   └── grafana-deployment.yaml      # Déploiement Grafana
│
├── data/                            # Datasets
│   ├── train_toxic_10k.csv
│   └── test_toxic_10k.csv
│
├── docs/                            # Documentation
│   ├── MLOPS_ARCHITECTURE.md
│   ├── DEPLOYMENT.md
│   └── RETRAINING.md
│
└── Tests/                           # Tests unitaires
    └── test_unitaires.py
```

## Développement

### Tests

```bash
# Tests unitaires
pytest Tests/ -v

# Tests de charge
locust -f locustfile.py --host=http://localhost:8080

# Validation données/modèle
python mlops/validation.py
```

### Build et Run Local

```bash
# Avec Docker
docker build -t toxic-detection-api .
docker run -p 8080:8080 toxic-detection-api

# Sans Docker
python app.py
```

### MLflow UI (Local)

```bash
# Lancer MLflow UI
mlflow ui

# Accès : http://localhost:5000
```

## Déploiement

### Déploiement Automatique (Production)

```bash
# Push sur main déclenche automatiquement le déploiement
git add .
git commit -m "feat: nouvelle fonctionnalité"
git push origin main
```

### Déploiement Manuel

```bash
# Via Cloud Build
gcloud builds submit --config=cloudbuild.yaml .

# Via kubectl
kubectl apply -f k8s/deployment-mlops.yaml
```

Pour un guide complet du déploiement, voir [docs/DEPLOYMENT.md](./docs/DEPLOYMENT.md).

## Monitoring

### Architecture Hybride Prometheus + Cloud Monitoring

L'API expose **11 métriques MLOps pertinentes** pour détecter le drift, surveiller la performance et optimiser les ressources :

**Métriques Business** :

- `ml_predictions_total` - Nombre de prédictions par classe (toxic/non_toxic) et niveau de confiance
- `ml_confidence_distribution` - Distribution des scores de confiance (détection drift)

**Métriques Performance** :

- `ml_processing_duration_seconds` - Latence d'inférence (P50/P95/P99)
- `ml_prediction_errors_total` - Erreurs ML par type (model_load, timeout, etc.)
- `http_request_duration_seconds` - Latence end-to-end API

**Métriques Infrastructure** :

- `app_memory_usage_bytes` - Consommation mémoire (détection memory leak)
- `http_requests_in_progress` - Requêtes simultanées (détection saturation)
- `container_cpu_usage_seconds` - Utilisation CPU des pods

### Dashboards Grafana

**Accès Grafana** : http://146.148.127.36:3000 (admin/admin123)

**3 Dashboards Professionnels** organisés par thématique :

#### 1 **Business & Utilisation**

- Total Predictions (stat avec tendance)
- Taux de Toxicité (gauge avec seuils colorés : vert <30%, jaune 30-50%, rouge >50%)
- Model Status (HEALTHY/DOWN)
- Volume Predictions/Heure (bar gauge)
- Predictions par Classe - Toxic vs Non-Toxic (timeseries)

#### 2 **Performance & Model Health**

- Latence Inference ML - P50/P95/P99 (timeseries)
- Latence HTTP End-to-End - P50/P95/P99 (timeseries)
- Taux d'Erreur ML (stat avec alertes : vert <1%, jaune 1-5%, rouge >5%)
- Requêtes en Cours (stat)
- Latence P95 détaillée en milliseconds
- **Distribution de Confiance (heatmap)** - Détection proactive du drift

#### 3 **Infrastructure & Ressources**

- Memory Usage MB (timeseries avec seuils)
- CPU Usage par Pod (timeseries multi-pods)
- Mémoire Actuelle (stat : vert <1GB, jaune 1-1.5GB, rouge >1.5GB)
- CPU Moyen (stat : vert <70%, jaune 70-90%, rouge >90%)
- Nombre de Pods & Pods Actifs
- Corrélation Mémoire vs CPU (timeseries)

**Justification Pédagogique MLOps** :

- **Détection proactive du drift** via heatmap de confiance
- **SLA de latence** (< 100ms P95 pour inférence ML, < 500ms pour HTTP)
- **Optimisation coûts GKE** via monitoring ressources détaillé
- **Traçabilité complète** du texte brut à la prédiction avec erreurs
- **Séparation des préoccupations** : Business / Performance / Infrastructure

**Documentation complète** : [docs/MONITORING.md](./docs/MONITORING.md)

### Commandes Kubernetes

```bash
# Pods status
kubectl get pods -l app=mlops-toxic-detection-api

# Logs en temps réel
kubectl logs -f deployment/mlops-toxic-detection-api

# Métriques Prometheus
curl http://34.22.130.34/metrics

# Port-forward Grafana (si LoadBalancer indisponible)
kubectl port-forward svc/grafana 3000:3000
```

### Cloud Platform

- **Cloud Console** : https://console.cloud.google.com
- **Cloud Monitoring** : Métriques infrastructure GKE (CPU, mémoire, réseau)
- **Cloud Logging** : Logs centralisés et recherche
- **MLflow** : Tracking des expériences
- **Prometheus** : Métriques applicatives custom (ML, API)
- **Grafana** : Dashboards unifiés (Prometheus + Cloud Monitoring)

## Sécurité et Conformité

### Authentification

L'API utilise JWT (JSON Web Tokens) pour l'authentification. Chaque requête vers `/predict` nécessite un token valide dans le header Authorization.

### Conformité RGPD

- **Anonymisation préalable** : Détection et masquage des noms, emails, téléphones avec spaCy NER
- **Minimisation des données** : Seules les données strictement nécessaires sont collectées
- **Pas de stockage** : Aucune donnée personnelle n'est conservée après traitement
- **Registre de traitement** : Documentation complète disponible

### Permissions IAM

Le service account Cloud Build dispose uniquement des permissions minimales requises :

- `roles/storage.admin` - Cloud Storage
- `roles/container.developer` - GKE
- `roles/aiplatform.user` - Vertex AI

## Technologies

- **Backend** : FastAPI 0.104, Python 3.11
- **ML** : scikit-learn, spaCy, MLflow
- **Cloud** : Google Cloud Platform (GKE, Cloud Build, Vertex AI, Cloud Storage)
- **Container** : Docker, Kubernetes
- **Monitoring** : Cloud Monitoring, Cloud Logging, Prometheus
- **CI/CD** : Cloud Build, GitHub
