from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import pickle
import pandas as pd

# CHARGER LES DONNÉES D'ENTRAÎNEMENT
print("Chargement des données...")
df = pd.read_csv('train_toxic_10k.csv')
texts = df['comment_text'].astype(str).tolist()
labels = df['toxic'].tolist()

print(f"Données chargées: {len(texts)} commentaires")
print(f"Distribution: {df['toxic'].value_counts().to_dict()}")

# PIPELINE TF-IDF + SVM
print("Configuration du pipeline TF-IDF + SVM...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),  # Utilise des unigrammes et bigrammes
        min_df=2,           # Ignore les mots qui apparaissent moins de 2 fois
        max_df=0.95,        # Ignore les mots trop fréquents
        strip_accents='ascii',
        lowercase=True,
        stop_words='english'
    )),
    ('svm', LinearSVC(
        random_state=42,
        class_weight='balanced',  # Gérer le déséquilibre des classes
        max_iter=10000
    ))
])

# LANCER L'ENTRAÎNEMENT
print(" Début de l'entraînement SVM...")
pipeline.fit(texts, labels)

# SAUVEGARDER LE MODÈLE
print("Sauvegarde du modèle...")
with open("./model/svm_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print(" ENTRAÎNEMENT SVM TERMINÉ!")
print(" Modèle sauvegardé dans ./model/svm_model.pkl")