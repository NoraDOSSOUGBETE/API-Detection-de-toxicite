# Guide de Réentraînement Automatique

## Vue d'ensemble

Le système de réentraînement automatique permet de mettre à jour le modèle de détection toxique lorsque de nouvelles données sont disponibles, sans intervention manuelle.

## Architecture du Pipeline

```
Nouvelles données → Cloud Storage (gs://.../data/)
    ↓
Notification Pub/Sub (gcs-data-changes topic)
    ↓
Cloud Build Trigger (retrain-on-data-upload)
    ↓
┌──────────────────────────────────────────────────────┐
│ Vertex AI Pipeline (cloudbuild-retrain.yaml)         │
│  Step 1: Install dependencies                        │
│  Step 2: Download spaCy model                        │
│  Step 3: Run Vertex AI training                      │
│  Step 4: Verify new model in GCS                     │
│  Step 5: MLflow tracking + Model Registry            │
└──────────────────────────────────────────────────────┘
    ↓
Nouveau modèle disponible → gs://.../models/svm_model_vertex.pkl
    ↓
Redéploiement manuel ou automatique de l'API
```

## Configuration Initiale

### 1. Créer le Topic Pub/Sub

```bash
gcloud pubsub topics create gcs-data-changes \
    --project=simplifia-hackathon
```

### 2. Configurer les Notifications GCS

```bash
gsutil notification create -t gcs-data-changes \
    -f json \
    -e OBJECT_FINALIZE \
    -p data/ \
    gs://mlops-models-simplifia-hackathon
```

Cette commande crée une notification qui envoie un message Pub/Sub chaque fois qu'un fichier est créé dans le dossier `data/`.

### 3. Créer le Trigger Cloud Build

**Via la console Cloud Build :**

1. Aller sur https://console.cloud.google.com/cloud-build/triggers
2. Cliquer sur "CREATE TRIGGER"
3. Configuration :
   - **Name** : `retrain-on-data-upload`
   - **Event** : Pub/Sub message
   - **Topic** : `gcs-data-changes`
   - **Source** : Khadija0203/API-Digital-Social-Score (branch: Adji)
   - **Build configuration** : Cloud Build configuration file
   - **Location** : `cloudbuild-retrain.yaml`

## Fichiers Impliqués

### cloudbuild-retrain.yaml

Pipeline Cloud Build pour le réentraînement :

```yaml
steps:
  # Étape 1: Installer les dépendances
  - name: "python:3.11"
    entrypoint: "pip"
    args:
      - "install"
      - "--no-cache-dir"
      - "google-cloud-aiplatform"
      - "kfp"
      - "mlflow"
      - "pandas"
      - "scikit-learn"
      - "datasets"
      - "spacy"
      - "google-cloud-storage"

  # Étape 2: Télécharger le modèle spaCy
  - name: "python:3.11"
    entrypoint: "python"
    args: ["-m", "spacy", "download", "en_core_web_sm"]

  # Étape 3: Lancer le pipeline Vertex AI
  - name: "python:3.11"
    entrypoint: "python"
    args: ["vertex.py"]
    env:
      - "PROJECT_ID=${PROJECT_ID}"
      - "REGION=europe-west1"

options:
  machineType: "E2_HIGHCPU_8"
  logging: CLOUD_LOGGING_ONLY

timeout: "3600s"
```

### vertex.py

Pipeline Vertex AI avec 3 composants principaux :

1. **prepare_data_op** : Anonymisation RGPD et nettoyage des données
2. **train_model_op** : Entraînement du modèle SVM
3. **evaluate_model_op** : Évaluation et sauvegarde dans MLflow

## Déclenchement du Réentraînement

### Méthode 1 : Upload de Nouvelles Données (Automatique)

```bash
# Uploader de nouvelles données
gsutil cp nouvelles_donnees.csv gs://mlops-models-simplifia-hackathon/data/

# Le pipeline se déclenche automatiquement via Pub/Sub
# Suivre l'exécution :
gcloud builds list --ongoing
```

### Méthode 2 : Déclenchement Manuel

```bash
# Depuis la console Cloud Build
# https://console.cloud.google.com/cloud-build/triggers
# Cliquer sur "RUN" pour le trigger "retrain-on-data-upload"
```

### Méthode 3 : Via gcloud submit

```bash
# Soumettre directement le build
gcloud builds submit --config=cloudbuild-retrain.yaml .
```

## Monitoring du Réentraînement

### Suivre le Build en Cours

```bash
# Lister les builds
gcloud builds list --limit=5

# Suivre les logs d'un build spécifique
gcloud builds log BUILD_ID --stream
```

### Vérifier le Nouveau Modèle

```bash
# Lister les modèles dans GCS
gsutil ls -lh gs://mlops-models-simplifia-hackathon/models/

# Vérifier MLflow
gsutil ls -r gs://mlops-models-simplifia-hackathon/mlflow/
```

### Console Web

- **Cloud Build** : https://console.cloud.google.com/cloud-build/builds
- **Vertex AI** : https://console.cloud.google.com/vertex-ai/pipelines
- **Cloud Storage** : https://console.cloud.google.com/storage

## Métriques et Validation

### Critères de Qualité

Le nouveau modèle doit satisfaire :

- **Accuracy ≥ 85%**
- **F1-Score ≥ 80%**
- **Amélioration ≥ 2%** par rapport au modèle actuel

### Métriques Trackées dans MLflow

```python
{
    "accuracy": 0.87,
    "precision_toxic": 0.85,
    "recall_toxic": 0.82,
    "f1_toxic": 0.83,
    "training_time_seconds": 45.2,
    "data_size": 10000,
    "model_size_mb": 12.4
}
```

## Redéploiement de l'API

### Option 1 : Redéploiement Automatique

Configurer un trigger qui redéploie l'API après un réentraînement réussi :

```yaml
# Dans cloudbuild-retrain.yaml, ajouter une dernière étape :
- name: "gcr.io/cloud-builders/gcloud"
  args:
    - "builds"
    - "submit"
    - "--config=cloudbuild.yaml"
    - "."
```

### Option 2 : Redéploiement Manuel

```bash
# Déclencher le déploiement de l'API
gcloud builds submit --config=cloudbuild.yaml .

# OU via GitHub (push sur main)
git commit --allow-empty -m "deploy: nouveau modèle"
git push origin main
```

### Option 3 : Restart des Pods Kubernetes

```bash
# Les nouveaux pods chargeront automatiquement le dernier modèle depuis GCS
kubectl rollout restart deployment/mlops-toxic-detection-api

# Vérifier le rollout
kubectl rollout status deployment/mlops-toxic-detection-api
```

## Troubleshooting

### Le Trigger ne se Déclenche Pas

**Vérifier la configuration Pub/Sub :**

```bash
# Vérifier le topic
gcloud pubsub topics list

# Vérifier les notifications GCS
gsutil notification list gs://mlops-models-simplifia-hackathon

# Tester manuellement
gcloud pubsub topics publish gcs-data-changes --message="test"
```

### Build Échoue à l'Étape 1 (Dependencies)

**Erreur courante :** Timeout ou conflit de versions

**Solution :**

```yaml
# Augmenter le timeout et spécifier les versions exactes
- name: "python:3.11"
  entrypoint: "pip"
  args:
    - "install"
    - "google-cloud-aiplatform==1.35.0"
    - "kfp==2.0.0"
    - "mlflow==2.8.0"
```

### Build Échoue à l'Étape 3 (Vertex AI)

**Erreur courante :** Permissions insuffisantes

**Solution :**

```bash
# Donner les permissions Vertex AI au service account Cloud Build
gcloud projects add-iam-policy-binding simplifia-hackathon \
    --member="serviceAccount:PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

### Le Nouveau Modèle n'est pas Chargé

**Vérifier le chemin dans l'API :**

```python
# app.py
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "gs://mlops-models-simplifia-hackathon/models/svm_model_vertex.pkl"
)
```

**Redémarrer les pods pour forcer le rechargement :**

```bash
kubectl delete pods -l app=mlops-toxic-detection-api
```

## Optimisations

### Réduire le Temps de Build

1. **Utiliser des images Docker pré-construites** avec les dépendances
2. **Cacher les artefacts** entre builds
3. **Paralléliser les étapes** indépendantes

### Réduire les Coûts

1. **Utiliser des machines moins puissantes** pour l'entraînement (si données < 100k)
2. **Limiter la fréquence** des réentraînements (une fois par semaine max)
3. **Supprimer les anciens modèles** automatiquement (GCS lifecycle policy)

```bash
# Lifecycle policy pour supprimer les modèles > 30 jours
gsutil lifecycle set lifecycle.json gs://mlops-models-simplifia-hackathon
```
