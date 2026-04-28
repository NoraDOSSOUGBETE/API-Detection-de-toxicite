from locust import HttpUser, task, between
import random
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToxicDetectionUser(HttpUser):
    """
    Utilisateur simulé pour tester l'API de détection de toxicité
    Compatible avec FastAPI + OAuth2 + JWT
    """
    
    # Temps d'attente entre les requêtes (simulation réaliste)
    wait_time = between(1, 3)
    
    # Token JWT stocké après login
    token = None
    
    def on_start(self):
        """
        Exécuté une seule fois au démarrage de chaque utilisateur
        - Connexion et récupération du token JWT
        """
        self.login()
    
    def login(self):
        """
        Authentification OAuth2 avec Form Data
        Endpoint: POST /token
        """
        try:
            response = self.client.post(
                "/token",
                data={ 
                    "username": "test",
                    "password": "test"
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded"
                },
                name="/token (login)"
            )
            
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("access_token")
                logger.info(f"Login réussi - Token: {self.token[:20]}...")
            else:
                logger.error(f" Échec login: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f" Erreur login: {str(e)}")
    
    @task(10)  # Tâche principale (90% du trafic)
    def predict_toxicity(self):
        """
        Test de prédiction avec des textes variés
        Endpoint: POST /predict (protégé)
        """
        if not self.token:
            logger.warning(" Pas de token - tentative de reconnexion")
            self.login()
            return
        
        # Corpus de test diversifié
        test_texts = [
            # Cas toxic (50%)
            "I'm going to kill you, you're horrible!",
            "You're really stupid and useless.",
            "Get lost, asshole, nobody likes you",
            "You dirty bastard, you deserve to suffer",
            "Shut your mouth, you idiot",
            
            # Cas non-toxic (40%)
            "Merci beaucoup pour ton aide !",
            "I love this movie, it's awesome!",
            "What a beautiful sunny day",
            "Great job, it's fantastic!",
            "I'm glad to see you again",
            
            # Edge cases (10%)
            "a" * 500,  # Texte très long
            "Test 123",  # Texte court
            "🔥💀👿",  # Emojis
            "AAAAAAAAAA",  # Répétitions
            ""  # Texte vide (devrait échouer)
        ]
        
        text = random.choice(test_texts)
        
        try:
            response = self.client.post(
                "/predict",
                json={"text": text}, 
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json"
                },
                name="/predict"
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(
                    f"Prédiction: {result['label']} "
                    f"(prob: {result['probability']:.2f}, "
                    f"longueur: {result['text_length']})"
                )
            elif response.status_code == 401:
                logger.warning("Token expiré - reconnexion")
                self.login()
            else:
                logger.error(f"Erreur: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Erreur prédiction: {str(e)}")
    
    @task(2)  # Tâche secondaire (20% du trafic)
    def health_check(self):
        """
        Vérification de santé de l'API
        Endpoint: GET /health (public)
        """
        try:
            response = self.client.get("/health", name="/health")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(
                    f"Santé: {data['status']} "
                    f"(model_loaded: {data['checks']['model_loaded']}, "
                    f"response_time: {data['response_time_ms']}ms)"
                )
            else:
                logger.error(f"Health check failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Erreur health check: {str(e)}")
    
    @task(1)  # Tâche rare (10% du trafic)
    def model_info(self):
        """
        Récupération des informations du modèle
        Endpoint: GET /model/info (public)
        """
        try:
            response = self.client.get("/model/info", name="/model/info")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(
                    f" Modèle: {data['model_name']} "
                    f"(version: {data['version']}, loaded: {data['loaded']})"
                )
            else:
                logger.error(f"Erreur info modèle: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Erreur model info: {str(e)}")
