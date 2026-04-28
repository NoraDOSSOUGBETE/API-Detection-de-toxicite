# Guide de Déploiement

Ce guide couvre le déploiement complet de l'API de détection toxique sur Google Kubernetes Engine (GKE).

## Prérequis

- Compte Google Cloud Platform actif
- gcloud CLI installé et configuré
- kubectl installé
- Docker installé (pour build local)
- Git

## Étape 1 : Configuration GCP

### 1.1 Créer un Projet GCP

```bash
# Créer un nouveau projet
gcloud projects create simplifia-hackathon --name="API Toxic Detection"

# Définir comme projet par défaut
gcloud config set project simplifia-hackathon

# Activer la facturation
gcloud beta billing projects link simplifia-hackathon \
    --billing-account=BILLING_ACCOUNT_ID
```

### 1.2 Activer les APIs Nécessaires

```bash
# Activer toutes les APIs en une commande
gcloud services enable \
    container.googleapis.com \
    cloudbuild.googleapis.com \
    storage-api.googleapis.com \
    aiplatform.googleapis.com \
    logging.googleapis.com \
    monitoring.googleapis.com \
    secretmanager.googleapis.com
```

### 1.3 Créer les Buckets Cloud Storage

```bash
# Bucket pour les modèles
gsutil mb -p simplifia-hackathon -l europe-west1 \
    gs://mlops-models-simplifia-hackathon

# Bucket pour les données
gsutil mb -p simplifia-hackathon -l europe-west1 \
    gs://mlops-data-simplifia-hackathon

# Configurer le versioning
gsutil versioning set on gs://mlops-models-simplifia-hackathon
```

## Étape 2 : Création du Cluster GKE

### 2.1 Créer le Cluster

```bash
gcloud container clusters create mlops-toxic-detection-cluster \
    --zone=europe-west1-b \
    --machine-type=e2-standard-2 \
    --num-nodes=2 \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=4 \
    --enable-autorepair \
    --enable-autoupgrade \
    --enable-cloud-logging \
    --enable-cloud-monitoring \
    --addons=HttpLoadBalancing
```

### 2.2 Obtenir les Credentials

```bash
gcloud container clusters get-credentials mlops-toxic-detection-cluster \
    --zone=europe-west1-b \
    --project=simplifia-hackathon
```

### 2.3 Vérifier la Connexion

```bash
kubectl cluster-info
kubectl get nodes
```

## Étape 3 : Configuration des Secrets

### 3.1 Créer le Secret JWT

```bash
# Générer une clé secrète
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Stocker dans Secret Manager
echo -n "VOTRE_CLE_SECRETE" | gcloud secrets create JWT_SECRET \
    --data-file=- \
    --replication-policy="automatic"
```

### 3.2 Créer le Secret Kubernetes

```bash
kubectl create secret generic api-secrets \
    --from-literal=JWT_SECRET="VOTRE_CLE_SECRETE" \
    --from-literal=JWT_ALGORITHM="HS256"
```

## Étape 4 : Upload du Modèle Initial

### 4.1 Entraîner le modèle Localement

```bash
# Installer les dépendances
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Entraîner le modèle
python model/SVM.py
```

### 4.2 Upload vers GCS

```bash
# Upload du modèle
gsutil cp model/svm_pipeline.pkl \
    gs://mlops-models-simplifia-hackathon/models/svm_model.pkl

# Upload des données d'entraînement
gsutil cp data/train_toxic_10k.csv \
    gs://mlops-data-simplifia-hackathon/data/
```

## Étape 5 : Déploiement via Cloud Build

### 5.1 Connecter GitHub à Cloud Build

### 5.2 Créer le Trigger de Déploiement

```bash
gcloud builds triggers create github \
    --name="deploy-api-main" \
    --repo-name="API-Digital-Social-Score" \
    --repo-owner="Khadija0203" \
    --branch-pattern="^main$" \
    --build-config="cloudbuild.yaml"
```

### 5.3 Déclencher le Premier Déploiement

```bash
# Option 1 : Via push GitHub
git push origin main

# Option 2 : Déclenchement manuel
gcloud builds triggers run deploy-api-main --branch=main

# Option 3 : Submit direct
gcloud builds submit --config=cloudbuild.yaml .
```

## Étape 6 : Vérification du Déploiement

### 6.1 Vérifier les Pods

```bash
# Lister les pods
kubectl get pods -l app=mlops-toxic-detection-api

# Vérifier les logs
kubectl logs -l app=mlops-toxic-detection-api --tail=50

# Vérifier le statut
kubectl describe deployment mlops-toxic-detection-api
```

### 6.2 Obtenir l'IP Externe

```bash
# Attendre que l'IP externe soit attribuée (peut prendre 2-3 minutes)
kubectl get service mlops-toxic-detection-service

# Récupérer uniquement l'IP
kubectl get service mlops-toxic-detection-service \
    -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

### 6.3 Tester l'API

```bash
# Remplacer EXTERNAL_IP par l'IP obtenue ci-dessus
export API_URL=http://EXTERNAL_IP

# Health check
curl $API_URL/health

# Obtenir un token
curl -X POST $API_URL/token \
    -d "username=admin&password=admin"

# Faire une prédiction
curl -X POST $API_URL/predict \
    -H "Authorization: Bearer YOUR_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"text": "This is a test comment"}'
```

## Étape 7 : Configuration du Monitoring

### 7.1 Créer un Dashboard Cloud Monitoring

1. Aller sur https://console.cloud.google.com/monitoring
2. Créer un nouveau dashboard
3. Ajouter des graphiques pour :
   - CPU utilization des pods
   - Memory utilization des pods
   - Request rate
   - Response latency (P50, P95, P99)
   - Error rate

### 7.2 Configurer les Alertes

```bash
# Créer une alerte pour latence élevée
gcloud alpha monitoring policies create \
    --notification-channels=CHANNEL_ID \
    --display-name="API High Latency" \
    --condition-display-name="P95 latency > 500ms" \
    --condition-threshold-value=0.5 \
    --condition-threshold-duration=300s
```

## Étape 8 : Configuration du Retraining Automatique

Voir [RETRAINING.md](./RETRAINING.md) pour la configuration complète.

**Résumé rapide :**

```bash
# 1. Créer le topic Pub/Sub
gcloud pubsub topics create gcs-data-changes

# 2. Configurer la notification GCS
gsutil notification create -t gcs-data-changes \
    -f json -e OBJECT_FINALIZE -p data/ \
    gs://mlops-models-simplifia-hackathon

# 3. Créer le trigger (via console ou gcloud)
# Voir RETRAINING.md pour les détails
```

## Étape 9 : Configuration du DNS (Optionnel)

### 9.1 Réserver une IP Statique

```bash
gcloud compute addresses create toxic-api-ip \
    --region=europe-west1

# Obtenir l'IP
gcloud compute addresses describe toxic-api-ip \
    --region=europe-west1 \
    --format="value(address)"
```

### 9.2 Configurer le Service avec IP Statique

```yaml
# Dans k8s/deployment-mlops.yaml
apiVersion: v1
kind: Service
metadata:
  name: mlops-toxic-detection-service
spec:
  type: LoadBalancer
  loadBalancerIP: "VOTRE_IP_STATIQUE"
  # ...
```

### 9.3 Configurer le DNS

```bash
# Ajouter un enregistrement A dans votre DNS
api.votredomaine.com -> VOTRE_IP_STATIQUE
```

## Maintenance

### Mise à Jour de l'API

```bash
# Méthode 1 : Via Git (recommandé)
git add .
git commit -m "feat: nouvelle fonctionnalité"
git push origin main  # Déclenche Cloud Build automatiquement

# Méthode 2 : Build et deploy manuel
gcloud builds submit --config=cloudbuild.yaml .
```

### Mise à Jour du Modèle

```bash
# 1. Entraîner un nouveau modèle
python model/SVM.py

# 2. Upload vers GCS
gsutil cp model/svm_pipeline.pkl \
    gs://mlops-models-simplifia-hackathon/models/svm_model.pkl

# 3. Redémarrer les pods
kubectl rollout restart deployment/mlops-toxic-detection-api
```

### Scaling Manuel

```bash
# Augmenter le nombre de replicas
kubectl scale deployment mlops-toxic-detection-api --replicas=5

# Vérifier
kubectl get pods -l app=mlops-toxic-detection-api
```

### Consulter les Logs

```bash
# Logs en temps réel
kubectl logs -f deployment/mlops-toxic-detection-api

# Logs Cloud Logging
gcloud logging read "resource.type=k8s_pod" --limit=50
```

## Désactivation/Suppression

### Supprimer le Déploiement (Garder le Cluster)

```bash
kubectl delete deployment mlops-toxic-detection-api
kubectl delete service mlops-toxic-detection-service
```

### Supprimer le Cluster Complet

```bash
gcloud container clusters delete mlops-toxic-detection-cluster \
    --zone=europe-west1-b
```

### Supprimer les Ressources GCP

```bash
# Supprimer les buckets
gsutil rm -r gs://mlops-models-simplifia-hackathon
gsutil rm -r gs://mlops-data-simplifia-hackathon

# Supprimer les secrets
gcloud secrets delete JWT_SECRET

# Désactiver les APIs (optionnel)
gcloud services disable container.googleapis.com
```

## Troubleshooting

### Pods en CrashLoopBackOff

```bash
# Voir les logs du pod
kubectl logs POD_NAME

# Décrire le pod pour voir les events
kubectl describe pod POD_NAME

# Causes communes :
# - Modèle non trouvé dans GCS
# - Secret JWT manquant
# - Dépendance manquante
```

### LoadBalancer en Pending

```bash
# Vérifier le service
kubectl describe service mlops-toxic-detection-service

```

### Build Cloud Build Échoue

```bash
# Voir les logs
gcloud builds log BUILD_ID
```


