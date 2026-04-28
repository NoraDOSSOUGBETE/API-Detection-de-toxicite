import requests
import threading
import time

# URL de votre API locale
API_URL = "http://localhost:8080/predict"

# Données JSON à envoyer
PAYLOAD = {"text": "ceci est un test d'attaque dos"}

# Fonction exécutée par chaque "attaquant" (thread)
def send_request():
    """Envoie des requêtes en boucle."""
    while True:
        try:
            # Envoie la requête POST. Timeout court pour ne pas attendre.
            requests.post(API_URL, json=PAYLOAD, timeout=0.5)
            print(".", end="", flush=True) # Affiche un . (succès)
        except requests.exceptions.RequestException:
            print("X", end="", flush=True) # Affiche un X (erreur ou timeout)
        
        time.sleep(0.01) # Petite pause pour le réalisme

# --- Point d'entrée du script ---
NUM_THREADS = 50 # Nombre d'attaquants simultanés

print(f"--- Lancement de l'attaque DoS sur {API_URL} avec {NUM_THREADS} threads ---")
print("Appuyez sur Ctrl+C pour arrêter.")

threads = []
for i in range(NUM_THREADS):
    # Crée un nouveau thread "attaquant"
    thread = threading.Thread(target=send_request, daemon=True)
    threads.append(thread)
    thread.start() # Lance l'attaquant

# Laisser l'attaque tourner jusqu'à interruption manuelle
try:
    while True:
        time.sleep(10)
except KeyboardInterrupt:
    print("\n--- Attaque arrêtée par l'utilisateur ---")