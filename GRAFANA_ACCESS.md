# Instructions pour acceder au dashboard Grafana

## Etapes :

1. **Ouvrir Grafana** : http://146.148.127.36:3000

   - Login: `admin`
   - Password: `admin123`

2. **Acceder au dashboard** :

   - Option 1: Cliquer sur le menu hamburger (☰) en haut a gauche
   - Aller dans "Dashboards"
   - Chercher "MLOps - Toxic Detection Monitoring"

   - Option 2: URL directe (apres connexion) :
     http://146.148.127.36:3000/d/mlops-toxic-detection/mlops-toxic-detection-monitoring

3. **Si le dashboard est vide** :

   - Executer `.\test_metrics.ps1` pour generer des predictions
   - Attendre 15 secondes (scrape interval de Prometheus)
   - Rafraichir le dashboard (F5)

4. **Verifier les metriques Prometheus** :

   - Prometheus UI: http://localhost:9090 (si port-forward actif)
   - Chercher: `ml_predictions_total`
   - Executer la requete pour voir les valeurs

5. **Panels du dashboard** :

   - Predictions par Classe (Toxic vs Non-Toxic)
   - Latence Inference ML (P50/P95/P99)
   - Taux de Toxicite (gauge)
   - Predictions Totales (stat)
   - Taux d'Erreurs ML
   - Requetes Concurrentes
   - Utilisation Memoire Application

6. **Troubleshooting** :
   - Si "No Data" : Verifier que Prometheus scrape les pods (`kubectl logs -l app=prometheus`)
   - Si erreur datasource : Verifier connexion Prometheus dans Grafana Settings > Datasources
   - Si dashboard manquant : Executer `kubectl get configmap grafana-mlops-dashboard` pour verifier
