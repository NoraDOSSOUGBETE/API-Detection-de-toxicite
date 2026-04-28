# Documentation

Cette documentation complète couvre tous les aspects du projet API Toxic Detection avec MLOps.

## Table des Matières

### 1. [Architecture MLOps](./MLOPS_ARCHITECTURE.md)

Documentation complète de l'architecture MLOps incluant :

- Vue d'ensemble de l'architecture
- Composants principaux (Cloud Build, GKE, MLflow, Monitoring)
- Flux de données (entraînement et prédiction)
- Sécurité et conformité RGPD
- Performance et scalabilité
- Disaster recovery
- Coûts estimés

### 2. [Guide de Déploiement](./DEPLOYMENT.md)

Guide pas à pas pour déployer l'API sur GCP :

- Configuration initiale GCP
- Création du cluster GKE
- Configuration des secrets
- Upload du modèle initial
- Déploiement via Cloud Build
- Vérification et tests
- Configuration du monitoring
- Maintenance et mises à jour

### 3. [Réentraînement Automatique](./RETRAINING.md)

Documentation du pipeline de réentraînement automatique :

- Architecture du pipeline Vertex AI
- Configuration Pub/Sub et Cloud Build
- Déclenchement automatique et manuel
- Monitoring des builds
- Métriques et validation
- Redéploiement de l'API
- Troubleshooting

### 4. [Monitoring MLOps](./MONITORING.md)

Documentation du système de monitoring hybride :

- Architecture Prometheus + Cloud Monitoring + Grafana
- Métriques MLOps pertinentes (business, performance, drift)
- Dashboards Grafana configurés
- Alerting et SLA
- Justification des choix techniques

## Guide de Lecture Recommandé

### Pour Commencer

Si vous débutez avec ce projet, suivez cet ordre :

1. Lire le [README principal](../README.md) pour une vue d'ensemble
2. Consulter [Architecture MLOps](./MLOPS_ARCHITECTURE.md) pour comprendre le système
3. Suivre le [Guide de Déploiement](./DEPLOYMENT.md) pour déployer l'API
4. Configurer le [Réentraînement Automatique](./RETRAINING.md) pour l'automatisation complète

## Ressources Additionnelles

### Fichiers du Projet

- **main.ipynb** : Notebook Jupyter avec le workflow complet d'analyse, anonymisation, entraînement et tests
- **cloudbuild.yaml** : Pipeline CI/CD pour le déploiement de l'API
- **cloudbuild-retrain.yaml** : Pipeline pour le réentraînement automatique
- **vertex.py** : Pipeline Vertex AI avec composants KFP
