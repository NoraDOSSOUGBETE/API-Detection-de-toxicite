# Architecture MLOps

## Vue d'ensemble

Ce projet implémente une architecture MLOps complète pour la détection de commentaires toxiques, incluant :

- Pipeline CI/CD automatisé avec Cloud Build
- Retraining automatique sur Vertex AI
- Tracking des expériences avec MLflow
- Déploiement sur Google Kubernetes Engine (GKE)
- Monitoring hybride (Prometheus + Cloud Monitoring) avec Grafana

Pour le monitoring détaillé, voir [MONITORING.md](MONITORING.md)

## Architecture Globale

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DÉVELOPPEMENT                                 │
├─────────────────────────────────────────────────────────────────────┤
│  Code Push (GitHub)                                                 │
│       ↓                                                             │
│  Cloud Build Trigger (main branch)                                  │
│       ↓                                                             │
│  ┌──────────────────────────────────────────────────────┐          │
│  │ Cloud Build Pipeline (cloudbuild.yaml)               │          │
│  │  Step 1: Setup Cloud Storage                         │          │
│  │  Step 2: Tests Unitaires (pytest)                    │          │
│  │  Step 3: Training MLflow + Model Registry            │          │
│  │  Step 4: Build Docker Image                          │          │
│  │  Step 5: Push to GCR                                 │          │
│  │  Step 6: Deploy to GKE                               │          │
│  └──────────────────────────────────────────────────────┘          │
│       ↓                                                             │
│  GKE Cluster (mlops-toxic-detection-cluster)                       │
│  - 3 Pods (LoadBalancer)                                           │
│  - Auto-scaling                                                     │
│  - Health checks                                                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Composants Principaux

### 1. Pipeline de Déploiement (cloudbuild.yaml)

Fichier de configuration Cloud Build qui orchestre le déploiement complet :

**Étapes du pipeline :**

1. **Configuration Cloud Storage** : Création des buckets pour les modèles et données
2. **Tests** : Exécution des tests unitaires (pytest)
3. **Entraînement MLOps** : Entraînement du modèle avec MLflow tracking
4. **Build Docker** : Construction de l'image de l'API
5. **Push GCR** : Envoi de l'image vers Google Container Registry
6. **Déploiement GKE** : Déploiement sur le cluster Kubernetes

**Déclenchement :** Push sur la branche `main`

### 2. Cluster Kubernetes (GKE)

**Configuration :**

- **Nom** : `mlops-toxic-detection-cluster`
- **Zone** : `europe-west1-b`
- **Type de machine** : `e2-standard-2`
- **Nombre de nœuds** : 2 (autoscale 1-4)
- **Namespace** : `default`

**Workloads :**

- **API Deployment** : 3 replicas pour haute disponibilité
- **Service LoadBalancer** : Exposition publique de l'API
- **Service Account** : Authentification GCP avec permissions minimales

### 3. MLflow Tracking

**Configuration :**

- **Backend Store** : SQLite local ou Cloud SQL (production)
- **Artifact Store** : Google Cloud Storage (`gs://mlops-models-{PROJECT_ID}/mlflow`)
- **Tracking URI** : GCS pour persistance

**Métriques trackées :**

- Accuracy, Precision, Recall, F1-Score
- Temps d'entraînement
- Taille du modèle
- Distribution des données
- Matrice de confusion

### 4. Monitoring et Observabilité

**Cloud Monitoring :**

- Métriques Kubernetes (CPU, mémoire, réseau)
- Métriques applicatives (requêtes/sec, latence, erreurs)
- Alertes configurées (P95 > 500ms, erreurs > 5%)

**Cloud Logging :**

- Logs centralisés des pods Kubernetes
- Logs Cloud Build
- Logs Vertex AI
- Recherche et filtrage avancés

## Flux de Données

### Données d'Entraînement

```
Données brutes (Hugging Face)
    ↓
Anonymisation RGPD (spaCy NER)
    ↓
Nettoyage (ponctuation, emojis, casse)
    ↓
Upload GCS (gs://mlops-data-{PROJECT_ID}/data/)
    ↓
Training Pipeline
    ↓
Modèle + Métriques → MLflow Registry
    ↓
Promotion "Production"
```

### Flux de Prédiction

```
Client HTTP Request
    ↓
LoadBalancer (IP externe)
    ↓
Service Kubernetes
    ↓
Pod API (1 parmi 3 replicas)
    ↓
Authentication JWT
    ↓
Validation Pydantic
    ↓
Modèle SVM (chargé en mémoire)
    ↓
Prédiction + Score
    ↓
Response JSON
```

## Sécurité et Conformité

### Authentification et Autorisation

- **JWT** : Tokens signés avec secret cryptographique (HS256)
- **Service Account GCP** : Permissions minimales (Least Privilege)
- **IAM Roles** : Attribution granulaire par ressource
- **Network Policies** : Isolation des pods Kubernetes

### Conformité RGPD

- **Anonymisation préalable** : spaCy NER pour détecter et masquer les données personnelles
- **Pas de stockage** : Aucune donnée personnelle n'est conservée
- **Registre de traitement**
- **Privacy by Design** : Anonymisation dès la collecte

### Audit et Traçabilité

- **Cloud Logging** : Tous les accès sont loggés
- **MLflow Tracking** : Historique complet des entraînements
- **Git** : Versioning du code source
- **Container Registry** : Versioning des images Docker

## Performance et Scalabilité

### Auto-scaling

**Horizontal Pod Autoscaler (HPA) :**

```yaml
minReplicas: 2
maxReplicas: 10
targetCPUUtilizationPercentage: 70
```

**Node Auto-scaling :**

- Min nodes: 1
- Max nodes: 4
- Scale-up déclenchée à 80% CPU/Memory

### Rollback Rapide

```bash
# Rollback déploiement Kubernetes
kubectl rollout undo deployment/mlops-toxic-detection-api

# Rollback modèle MLflow
# Charger une version précédente depuis MLflow Registry

# Rollback code
git revert COMMIT_HASH
git push origin main  # Déclenche redéploiement automatique
```
