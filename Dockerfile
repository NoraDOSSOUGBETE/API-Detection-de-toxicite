# Dockerfile pour l'API de détection toxique
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="API Digital Social Score Team"
LABEL version="1.0.0"
LABEL description="API de détection de commentaires toxiques avec SVM"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV MODEL_PATH=/app/model/svm_model.pkl

# Créer un utilisateur non-root pour la sécurité
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Répertoire de travail
WORKDIR /app

# Copier les requirements en premier (cache Docker)
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY app.py .
COPY model/ ./model/

# Créer les dossiers nécessaires
RUN mkdir -p logs && \
    chown -R appuser:appuser /app

# Changer vers l'utilisateur non-root
USER appuser

# Exposer le port
EXPOSE 8080

# Health check pour Kubernetes
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Commande par défaut
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]