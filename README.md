# 🛡️ Digital Social Score — API de Détection de Toxicité - MLOps Production

> Système de détection de commentaires toxiques avec MLOps complet : CI/CD automatisé, monitoring en production, conformité RGPD, et retraining automatique sur Vertex AI.

## Vue d'Ensemble

Ce projet implémente une API de détection de toxicité dans les commentaires avec une infrastructure MLOps complète sur Google Cloud Platform. L'API utilise un modèle SVM entraîné sur le "dataset Toxic Comment Classification" de Hugging Face, avec anonymisation RGPD des données personnelles.




<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-GKE-326CE5?logo=kubernetes&logoColor=white)
![GCP](https://img.shields.io/badge/Google_Cloud-Platform-4285F4?logo=googlecloud&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.8-0194E2?logo=mlflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikitlearn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Système MLOps de bout en bout pour la détection de commentaires toxiques**  
Déploiement continu · Retraining automatique · Monitoring production · Conformité RGPD · IA Act

[🚀 API en production](http://34.22.130.34) · [📖 Swagger Docs](http://34.22.130.34/docs) · [📊 Grafana](http://146.148.127.36:3000)

</div>

---


### Ce que ce projet accomplit concrètement

| Dimension | Ce qui est mis en place |
|---|---|
| **Modèle ML** | SVM + TF-IDF entraîné sur 10 000 commentaires (Jigsaw/Kaggle), avec fallback automatique |
| **API** | FastAPI avec auth JWT, documentation Swagger auto-générée, endpoints Prometheus |
| **CI/CD** | Pipeline Cloud Build déclenché à chaque push sur `main` : tests → entraînement → build Docker → déploiement GKE |
| **Retraining** | Pipeline Vertex AI déclenché automatiquement via Pub/Sub quand de nouvelles données arrivent sur GCS |
| **Monitoring** | 11 métriques MLOps custom exposées via Prometheus, 3 dashboards Grafana thématisés |
| **RGPD** | Anonymisation des données personnelles (NER spaCy) avant tout traitement |
| **IA Act** | Classification du modèle à « Risque Limité » documentée dans le `model_card.json` |
| **Résilience** | 3 replicas Kubernetes, auto-scaling, health checks, fallback model |

---

## 🏗️ Architecture Complète

```
┌────────────────────────────────────────────────────────────────────┐
│                         DÉVELOPPEMENT                              │
│  Git Push (main) ──► Cloud Build Trigger                          │
│                           │                                        │
│              ┌────────────▼────────────┐                          │
│              │   CI/CD Pipeline        │                          │
│              │  1. Tests unitaires     │                          │
│              │  2. Entraînement MLflow │                          │
│              │  3. Build Docker        │                          │
│              │  4. Push GCR            │                          │
│              │  5. Deploy GKE          │                          │
│              └────────────┬────────────┘                          │
└───────────────────────────┼────────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────────┐
│                    PRODUCTION (GKE)                                │
│                                                                    │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐                       │
│   │  Pod 1   │  │  Pod 2   │  │  Pod 3   │  ← 3 replicas         │
│   │ FastAPI  │  │ FastAPI  │  │ FastAPI  │    LoadBalancer        │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘                       │
│        └─────────────┼─────────────┘                              │
│                      │                                             │
│         ┌────────────▼────────────┐                               │
│         │     Prometheus          │                               │
│         │   (métriques custom)    │                               │
│         └────────────┬────────────┘                               │
│                      │                                             │
│         ┌────────────▼────────────┐                               │
│         │       Grafana           │                               │
│         │   (3 dashboards)        │                               │
│         └─────────────────────────┘                               │
└────────────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────────────┐
│                   RETRAINING AUTOMATIQUE                           │
│                                                                    │
│   Nouvelles données (GCS)                                         │
│         │                                                          │
│   Pub/Sub notification                                             │
│         │                                                          │
│   Cloud Function Trigger                                           │
│         │                                                          │
│   ┌─────▼──────────────────────────────────┐                      │
│   │       Vertex AI Pipeline               │                      │
│   │  1. Chargement des données             │                      │
│   │  2. Anonymisation RGPD (spaCy NER)     │                      │
│   │  3. Entraînement SVM + validation      │                      │
│   │  4. Enregistrement MLflow Registry     │                      │
│   │  5. Déploiement si métriques OK        │                      │
│   └─────────────────────────────────────────┘                      │
└────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Fonctionnalités Détaillées

### 🤖 Modèle Machine Learning

- **Algorithme** : `LinearSVC` avec pipeline `TfidfVectorizer → SVM`
- **Dataset** : Jigsaw Toxic Comment Classification (Hugging Face), 10 000 commentaires équilibrés
- **Métriques** : Accuracy, Precision, Recall, F1-Score trackées dans MLflow à chaque run
- **Fallback** : En cas d'indisponibilité du modèle MLflow, chargement automatique depuis un pickle local
- **Model Card** : Classification IA Act « Risque Limité » documentée (`model_card.json`)

### 🔄 Pipeline CI/CD (Cloud Build)

Déclenché automatiquement à chaque push sur `main` :

1. **Setup Cloud Storage** — Création des buckets GCS si absents
2. **Tests unitaires** — `pytest` avec couverture
3. **Entraînement MLOps** — Entraînement SVM avec tracking MLflow complet
4. **Build Docker** — Image optimisée multi-stage
5. **Push GCR** — Versionnage de l'image dans Google Container Registry
6. **Deploy GKE** — Rolling update sans downtime sur le cluster Kubernetes

### 🔁 Retraining Automatique (Vertex AI)

Cycle de vie du modèle entièrement automatisé :

- **Déclencheur** : Upload d'un nouveau CSV dans le bucket GCS → notification Pub/Sub → Cloud Function → Pipeline Vertex AI
- **Pipeline KFP** (Kubeflow Pipelines) en 4 composants :
  - `prepare_data_op` — Chargement + anonymisation RGPD
  - `train_model_op` — Entraînement SVM + validation
  - `register_model_op` — Enregistrement dans MLflow Model Registry
  - `deploy_model_op` — Déploiement si les métriques dépassent les seuils
- **Gestion des versions** : Chaque modèle est versionné, taggé et auditable

### 📊 Monitoring Production (11 métriques MLOps)

#### Métriques Business
| Métrique | Description |
|---|---|
| `ml_predictions_total` | Nombre de prédictions par classe (`toxic`/`non_toxic`) et niveau de confiance |
| `ml_confidence_distribution` | Distribution des scores de confiance (détection de drift) |

#### Métriques Performance
| Métrique | Description |
|---|---|
| `ml_processing_duration_seconds` | Latence d'inférence ML pure (P50/P95/P99) |
| `ml_prediction_errors_total` | Erreurs par type (`model_load`, `timeout`, etc.) |
| `http_request_duration_seconds` | Latence end-to-end (P50/P95/P99) |

#### Métriques Infrastructure
| Métrique | Description |
|---|---|
| `app_memory_usage_bytes` | Consommation mémoire par pod (détection de memory leak) |
| `http_requests_in_progress` | Requêtes simultanées (détection de saturation) |
| `container_cpu_usage_seconds` | CPU des pods Kubernetes |

**SLAs configurés** :
- Latence ML P95 < 100ms
- Latence HTTP P95 < 500ms
- Taux d'erreur < 1%

### 📈 Dashboards Grafana (3 thématiques)

**Dashboard 1 — Business & Utilisation**
- Total des prédictions (stat avec tendance)
- Taux de toxicité (gauge colorée : vert <30%, jaune 30–50%, rouge >50%)
- Statut du modèle (HEALTHY / DOWN)
- Volume prédictions/heure (bar gauge)
- Séries temporelles toxic vs non-toxic

**Dashboard 2 — Performance & Model Health**
- Latence inférence ML (P50/P95/P99)
- Latence HTTP end-to-end
- Taux d'erreur ML avec alertes
- Heatmap de distribution de confiance (détection proactive du drift)

**Dashboard 3 — Infrastructure & Ressources**
- Memory usage par pod (MB)
- CPU par pod (multi-series)
- Nombre de pods actifs
- Corrélation Mémoire × CPU

### 🔒 Sécurité & Conformité

**Authentification JWT**
- Tokens HS256 signés, durée de vie configurable
- Endpoint `/token` via OAuth2 Password Flow
- Chaque requête `/predict` requiert un Bearer token valide

**Conformité RGPD**
- Anonymisation préalable des PII via spaCy NER (noms propres, emails, numéros de téléphone)
- Aucun stockage des textes après traitement
- Minimisation des données collectées
- Logs anonymisés

**Permissions IAM minimales** (principe du moindre privilège)
- `roles/storage.admin` — Cloud Storage uniquement
- `roles/container.developer` — GKE uniquement
- `roles/aiplatform.user` — Vertex AI uniquement

---

## 🚀 Démarrage Rapide

### Prérequis

- Python 3.11+
- Docker (optionnel, pour le déploiement conteneurisé)
- Compte GCP avec les APIs activées (pour le déploiement cloud)

### Installation locale

```bash
# 1. Cloner le dépôt
git clone https://github.com/Khadija0203/API-Digital-Social-Score.git
cd API-Digital-Social-Score

# 2. Installer les dépendances
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos valeurs

# 4. Lancer l'API
python app.py
```

L'API est accessible sur [http://localhost:8080/docs](http://localhost:8080/docs)

### Avec Docker

```bash
# Build
docker build -t digital-social-score-api .

# Run
docker run -p 8080:8080 \
  -e JWT_SECRET=votre-secret \
  -e PROJECT_ID=votre-projet-gcp \
  digital-social-score-api
```

### Avec Docker Compose (API + Prometheus + Grafana)

```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

---

## 📡 Utilisation de l'API

### 1. Authentification

```bash
curl -X POST http://34.22.130.34/token \
  -d "username=admin&password=admin"
```

Réponse :
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### 2. Prédiction de toxicité

```bash
curl -X POST http://34.22.130.34/predict \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test comment"}'
```

Réponse :
```json
{
  "text": "This is a test comment",
  "is_toxic": false,
  "score": 15
}
```

### 3. Health Check

```bash
curl http://34.22.130.34/health
```

### 4. Métriques Prometheus

```bash
curl http://34.22.130.34/metrics
```

### Endpoints

| Méthode | Endpoint | Auth | Description |
|---|---|---|---|
| `POST` | `/token` | ❌ | Authentification, obtention du JWT |
| `POST` | `/predict` | ✅ JWT | Prédiction de toxicité sur un texte |
| `GET` | `/health` | ❌ | Health check de l'API et du modèle |
| `GET` | `/metrics` | ❌ | Métriques Prometheus (format texte) |
| `GET` | `/docs` | ❌ | Documentation interactive Swagger UI |

---

## 🧪 Tests

### Tests unitaires

```bash
pytest Tests/ -v
```

### Tests de charge (Locust)

```bash
# Interface web
locust -f locustfile.py --host=http://localhost:8080

# Mode headless (500 utilisateurs, spawn 10/sec)
locust -f locustfile.py --host=http://localhost:8080 \
  --headless -u 500 -r 10 --run-time 60s
```

Le `locustfile.py` simule des utilisateurs réalistes qui :
1. S'authentifient via `/token`
2. Envoient des requêtes de prédiction variées
3. Testent les health checks

### Validation du modèle

```bash
python mlops/validation.py
```

Le `DataValidator` vérifie :
- Présence des colonnes requises
- Volume minimum de données (≥ 1 000 samples)
- Absence de valeurs manquantes
- Cohérence des labels (0/1 uniquement)
- Distribution des classes (ratio toxic entre 5% et 95%)

---

## ☁️ Déploiement Cloud

### Déploiement automatique (Production)

Tout push sur `main` déclenche le pipeline complet :

```bash
git add .
git commit -m "feat: nouvelle fonctionnalité"
git push origin main
# → Cloud Build démarre automatiquement
```

### Déploiement manuel via Cloud Build

```bash
gcloud builds submit --config=cloudbuild.yaml .
```

### Déploiement direct Kubernetes

```bash
kubectl apply -f k8s/deployment-mlops.yaml
kubectl apply -f k8s/prometheus-deployment.yaml
kubectl apply -f k8s/grafana-deployment.yaml
```

### Commandes utiles Kubernetes

```bash
# Statut des pods
kubectl get pods -l app=mlops-toxic-detection-api

# Logs en temps réel
kubectl logs -f deployment/mlops-toxic-detection-api

# Scaling manuel
kubectl scale deployment mlops-toxic-detection-api --replicas=5

# Port-forward Grafana (si LoadBalancer indisponible)
kubectl port-forward svc/grafana 3000:3000
```

### Configurer le Retraining Automatique

```bash
# Déployer le trigger Pub/Sub + Cloud Function
bash setup-triggers.sh

# Déclencher manuellement un retraining
python cloud-function-trigger.py
```

---

## 🗂️ Structure du Projet

```
API-Digital-Social-Score/
│
├── app.py                          # API FastAPI + métriques Prometheus
├── vertex.py                       # Pipeline Vertex AI (KFP)
├── cloud-function-trigger.py       # Cloud Function déclencheur retraining
├── locustfile.py                   # Tests de charge
├── setup-triggers.sh               # Setup Pub/Sub + triggers GCS
├── model_card.json                 # Fiche modèle (IA Act, RGPD)
├── Dockerfile                      # Image Docker optimisée
├── requirements.txt                # Dépendances Python
│
├── mlops/                          # Cœur du pipeline MLOps
│   ├── training.py                 # Entraînement + MLflow tracking
│   ├── validation.py               # Validation données & modèle
│   ├── tracking.py                 # Intégration MLflow avancée
│   ├── config.py                   # Configuration centralisée
│   └── utils.py                    # Utilitaires partagés
│
├── model/                          # Implémentations des modèles
│   ├── SVM.py                      # Modèle principal (LinearSVC + TF-IDF)
│   └── BERT.py                     # Modèle alternatif (expérimental)
│
├── k8s/                            # Manifests Kubernetes
│   ├── deployment-mlops.yaml       # Déploiement API (3 replicas)
│   ├── prometheus-deployment.yaml  # Déploiement Prometheus
│   └── grafana-deployment.yaml     # Déploiement Grafana
│
├── grafana/
│   ├── dashboards/
│   │   ├── dashboard-business.json      # Business & Utilisation
│   │   ├── dashboard-performance.json   # Performance & Model Health
│   │   └── dashboard-infrastructure.json # Infrastructure & Ressources
│   └── datasources/
│       └── prometheus.yml
│
├── prometheus/
│   └── prometheus.yml              # Configuration scraping
│
├── attack/
│   └── attack_dos.py              # Script test de résistance DoS
│
├── docs/
│   ├── MLOPS_ARCHITECTURE.md      # Architecture détaillée
│   ├── DEPLOYMENT.md              # Guide de déploiement complet
│   ├── MONITORING.md              # Justification des 11 métriques
│   └── RETRAINING.md              # Configuration du retraining
│
├── .gitlab.yaml                   # Pipeline GitLab CI (alternatif)
├── cloudbuild.yaml                # Pipeline CI/CD principal (Cloud Build)
├── cloudbuild-retrain.yaml        # Pipeline retraining Cloud Build
├── docker-compose.monitoring.yml  # Stack monitoring locale
└── GRAFANA_ACCESS.md              # Guide accès & import dashboards
```

---

## ⚙️ Variables d'Environnement

| Variable | Obligatoire | Description | Défaut |
|---|---|---|---|
| `JWT_SECRET` | ✅ | Secret de signature des tokens JWT (256 bits min) | `fallback-secret-key` |
| `PROJECT_ID` | ✅ | ID du projet Google Cloud | — |
| `PORT` | ❌ | Port d'écoute de l'API | `8080` |
| `REGION` | ❌ | Région GCP pour Vertex AI | `europe-west1` |
| `ENABLE_METRICS` | ❌ | Active/désactive les métriques Prometheus | `true` |

Voir `.env.example` pour un template complet.

---

## 🛠️ Stack Technique

| Catégorie | Technologie | Version |
|---|---|---|
| **API** | FastAPI | 0.104.1 |
| **Serveur** | Uvicorn | 0.24.0 |
| **ML** | scikit-learn | ≥ 1.3.0 |
| **Tracking** | MLflow | ≥ 2.8.0 |
| **NLP (RGPD)** | spaCy | — |
| **Auth** | python-jose (JWT) | ≥ 3.3.0 |
| **Monitoring** | Prometheus Client | 0.19.0 |
| **Load Test** | Locust | — |
| **Cloud** | Google Cloud Platform | — |
| **Orchestration** | Kubernetes (GKE) | — |
| **Pipeline ML** | Vertex AI + KFP | ≥ 2.0.0 |
| **CI/CD** | Cloud Build | — |
| **Registry** | Google Container Registry | — |
| **Dashboards** | Grafana | — |

---

## 🌐 Accès aux Environnements

| Service | URL | Identifiants |
|---|---|---|
| **API Production** | http://34.22.130.34 | — |
| **Swagger UI** | http://34.22.130.34/docs | — |
| **Grafana** | http://146.148.127.36:3000 | `admin` / `admin123` |
| **Métriques** | http://34.22.130.34/metrics | — |
| **MLflow (local)** | http://localhost:5000 | `mlflow ui` |

---

## 📚 Documentation

| Document | Contenu |
|---|---|
| [MLOPS_ARCHITECTURE.md](./docs/MLOPS_ARCHITECTURE.md) | Architecture complète, flux de données, composants GCP |
| [DEPLOYMENT.md](./docs/DEPLOYMENT.md) | Guide pas-à-pas pour déployer sur GCP et GKE |
| [RETRAINING.md](./docs/RETRAINING.md) | Configuration du pipeline de retraining automatique |
| [MONITORING.md](./docs/MONITORING.md) | Justification des 11 métriques MLOps, dashboards Grafana |
| [GRAFANA_ACCESS.md](./GRAFANA_ACCESS.md) | Guide d'accès et import des dashboards |


---

<div align="center">

Projet académique fait en binôme  
Projet MLOps — Google Cloud Platform · FastAPI · Kubernetes

</div>
