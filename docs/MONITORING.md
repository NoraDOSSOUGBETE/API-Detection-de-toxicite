# Monitoring MLOps - Architecture Hybride

## Architecture Deployee

```
┌─────────────────────────────────────────────────────────────┐
│                    GKE Cluster Production                   │
│                                                             │
│  ┌────────────────┐      ┌──────────────┐                   │
│  │ API Pods (x3)  │─────>│  Prometheus  │────┐              │
│  │  /metrics      │      │  (collecte)  │    │              │
│  └────────────────┘      └──────────────┘    │              │
│                                              │              │
│                          ┌──────────────┐    │              │
│                          │   Grafana    │<───┤              │
│                          │ LoadBalancer │    │              │
│                          └──────────────┘    │              │
│                                 │            │              │
│                                 └────────────┴──────────┐   │
│                                                          │   │
└──────────────────────────────────────────────────────────│───┘
                                                           │
                                                           v
                                              ┌────────────────────┐
                                              │ Cloud Monitoring   │
                                              │ (Infrastructure)   │
                                              └────────────────────┘
```

## Metriques MLOps Pertinentes

### 1. Metriques Business (Prometheus)

**ml_predictions_total{result="toxic|non_toxic", confidence_level="high|medium|low"}**

- **Justification**: Mesure directe de la valeur metier - detection de contenu toxique
- **Usage**: Calculer le taux de toxicite, detecter les pics anormaux
- **Alerte**: Si taux toxique > 30% pendant 10min → drift potentiel

**ml_confidence_distribution**

- **Justification**: Detection de data drift - changement de distribution des predictions
- **Usage**: Comparer la distribution actuelle vs baseline d'entrainement
- **Alerte**: Si 80% des predictions ont confiance < 0.6 → retraining requis

### 2. Metriques Performance Modele (Prometheus)

**ml_processing_duration_seconds**

- **Justification**: SLA de latence inference (<100ms P95)
- **Usage**: Monitorer degradation performance modele
- **Alerte**: P95 > 200ms → probleme performance

**ml_prediction_errors_total{error_type="model_load|prediction_failed|timeout"}**

- **Justification**: Fiabilite du systeme ML
- **Usage**: Detecter erreurs chargement modele, timeout
- **Alerte**: Taux erreur > 1% → incident critique

### 3. Metriques Applicatives (Prometheus)

**http_request_duration_seconds**

- **Justification**: Performance end-to-end API
- **Usage**: Latence totale incluant preprocessing, inference, postprocessing
- **Alerte**: P99 > 500ms → degradation service

**http_requests_in_progress**

- **Justification**: Saturation des pods
- **Usage**: Detecter besoin d'autoscaling
- **Alerte**: > 50 requetes simultanées par pod → scale up

**app_memory_usage_bytes**

- **Justification**: Detection de memory leak du modele
- **Usage**: Monitorer consommation memoire (modele BERT = ~500MB)
- **Alerte**: Memoire > 1.5GB → memory leak potentiel

### 4. Metriques Infrastructure (Cloud Monitoring)

**kubernetes.io/container/cpu/core_usage_time**

- **Justification**: Optimisation des ressources GKE
- **Usage**: Ajuster CPU requests/limits
- **Dashboard**: Corrélation CPU vs latence inference

**kubernetes.io/container/memory/used_bytes**

- **Justification**: Detection OOMKilled
- **Usage**: Prevenir crashes par manque de memoire
- **Alerte**: Memoire > 90% → risque OOM

**kubernetes.io/pod/network/received_bytes_count**

- **Justification**: Detection d'attaques DDoS
- **Usage**: Monitorer trafic reseau entrant
- **Alerte**: Traffic > 10x baseline → attaque potentielle

## Dashboards Grafana

### Dashboard 1: MLOps Overview

- **Panel 1**: Predictions par classe (toxic vs non-toxic) - Rate 5min
- **Panel 2**: Latence inference P50/P95/P99
- **Panel 3**: Heatmap distribution de confiance (detection drift)
- **Panel 4**: Taux d'erreurs ML par type
- **Panel 5**: CPU/Memoire pods (Cloud Monitoring)

### Dashboard 2: Business KPIs

- **Taux de toxicite global**: % predictions toxic sur 24h
- **Volume de predictions**: Total predictions/heure
- **Top langages toxiques**: Si metadata disponible
- **Temps de reponse moyen**: P50 latence end-to-end

### Dashboard 3: Model Health

- **Model drift score**: Ecart distribution confiance vs baseline
- **Prediction confidence trend**: Evolution moyenne sur 7 jours
- **Error rate trend**: Tendance taux erreur
- **Retraining triggers**: Historique retraining automatiques

## Acces Grafana

**URL Production**: http://146.148.127.36:3000

**Credentials**:

- Username: `admin`
- Password: `admin123`

**Datasources configurees**:

1. **Prometheus** (default): http://prometheus:9090
2. **Cloud Monitoring**: Authentification GCE automatique

## Alerting (Future)

### Alertes Critiques (PagerDuty)

- Taux erreur ML > 5% pendant 5min
- P95 latence > 500ms pendant 10min
- Model status = 0 (modele non charge)

### Alertes Warning (Email)

- Taux toxicite > 40% pendant 30min (drift suspect)
- Memoire pods > 80% pendant 15min
- Confidence moyenne < 0.65 pendant 1h

### Alertes Info (Slack)

- Nouveau retraining demarre
- Nouveau modele deploye
- Degradation latence P95 > 200ms

## Justification choix techniques

**Pourquoi Prometheus + Cloud Monitoring ?**

- **Prometheus**: Metriques applicatives custom (predictions, confiance, latence modele)
- **Cloud Monitoring**: Metriques infrastructure GKE (CPU, memoire, reseau)
- **Grafana**: Unification des deux sources dans dashboards coherents

**Pourquoi ces metriques ?**

- **Business**: Alignement avec objectif metier (detection toxicite)
- **Performance**: SLA latence < 100ms P95
- **Drift**: Detection proactive degradation modele
- **Infrastructure**: Optimisation couts GKE et fiabilite

**Conformite pedagogique**:

- Métriques pertinentes pour MLOps (pas juste CPU/RAM)
- Monitoring proactif (detection drift avant degradation)
- Traçabilite complete (du texte brut a la prediction)
- Integration production-ready (Prometheus + GKE)
