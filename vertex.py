import os
from kfp.v2 import dsl
from kfp.v2.dsl import component, Input, Output, Model
from google.cloud import aiplatform
import mlflow

# Configuration
PROJECT_ID = os.getenv('PROJECT_ID')
REGION = os.getenv('REGION', 'europe-west1')
PIPELINE_ROOT = f"gs://mlops-models-{PROJECT_ID}/pipeline-root"

@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "datasets", "spacy", "tqdm", "google-cloud-storage"]
)
def prepare_data_op(
    project_id: str,
    bucket_name: str,
    output_path: str
):
    """Preparation des donnees"""
    import pandas as pd
    import spacy
    import re
    import string
    from tqdm import tqdm
    from datasets import load_dataset
    from google.cloud import storage
    import io
    
    print("=== PREPARATION DES DONNEES ===")
    
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name.replace('gs://', ''))
    
    # Essayer de charger les donnees deja traitees
    try:
        blob = bucket.blob('data/dataset_cleaned_and_anonymized_10k.csv')
        content = blob.download_as_text()
        df = pd.read_csv(io.StringIO(content))
        print(f"Données déjà traitées trouvées: {len(df)} lignes")
    except:
        print("Données traitées non trouvées, traitement complet...")
        
        # Chargement depuis Hugging Face
        print("Chargement dataset Toxic Comment Classification...")
        dataset = load_dataset("thesofakillers/jigsaw-toxic-comment-classification-challenge", split="train")
        df = pd.DataFrame(dataset)
        print(f"Dataset chargé: {len(df)} commentaires")
        
        # Installation spacy model si necessaire
        import subprocess
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")
        
        # Fonction d'anonymisation RGPD (logique main.ipynb)
        EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        PHONE_PATTERN = re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b')
        
        def detect_personal_data(text):
            if not text or pd.isna(text):
                return {'has_personal_data': False, 'names': [], 'emails': [], 'phones': []}
            
            text_str = str(text)
            emails = EMAIL_PATTERN.findall(text_str)
            phones = PHONE_PATTERN.findall(text_str)
            
            doc = nlp(text_str)
            names = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
            
            return {
                'has_personal_data': bool(names or emails or phones),
                'names': names, 'emails': emails, 'phones': phones
            }
        
        def anonymize_text(text, detection_result):
            if not detection_result['has_personal_data']:
                return text
            
            anonymized_text = str(text)
            for email in detection_result['emails']:
                anonymized_text = anonymized_text.replace(email, "[EMAIL]")
            for phone in detection_result['phones']:
                anonymized_text = anonymized_text.replace(phone, "[PHONE]")
            for name in detection_result['names']:
                anonymized_text = anonymized_text.replace(name, "[NAME]")
            
            return anonymized_text
        
        # Fonction de nettoyage (logique main.ipynb)
        def clean_text(text):
            if not text or pd.isna(text):
                return ""
            
            text = str(text).lower()
            
            # Supprimer emojis
            emoji_pattern = re.compile("["
                                      u"\U0001F600-\U0001F64F"
                                      u"\U0001F300-\U0001F5FF"
                                      u"\U0001F680-\U0001F6FF"
                                      u"\U0001F1E0-\U0001F1FF"
                                      "]+", flags=re.UNICODE)
            text = emoji_pattern.sub(r'', text)
            
            # Nettoyer URLs, mentions, hashtags, etc.
            text = re.sub(r'[^\x00-\x7F]+', ' ', text)
            text = re.sub(r'http[s]?://\S+', ' ', text)
            text = re.sub(r'@\w+', ' ', text)
            text = re.sub(r'#\w+', ' ', text)
            text = re.sub(r'\b\d+\b', ' ', text)
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        
        # Traitement sur 10k echantillons
        max_samples = 10000
        df_sample = df.head(max_samples).copy()
        
        print("Anonymisation RGPD...")
        comments_anonymized = 0
        for i in tqdm(range(len(df_sample)), desc="Anonymisation"):
            original_comment = df_sample['comment_text'].iloc[i]
            detection = detect_personal_data(original_comment)
            
            if detection['has_personal_data']:
                comments_anonymized += 1
                anonymized_comment = anonymize_text(original_comment, detection)
                df_sample.loc[df_sample.index[i], 'comment_text'] = anonymized_comment
        
        print(f"Commentaires anonymisés: {comments_anonymized}")
        
        print("Nettoyage des textes...")
        for i in tqdm(range(len(df_sample)), desc="Nettoyage"):
            original_comment = df_sample['comment_text'].iloc[i]
            cleaned_comment = clean_text(original_comment)
            df_sample.loc[df_sample.index[i], 'comment_text'] = cleaned_comment
        
        # Supprimer commentaires vides
        df_sample = df_sample[df_sample['comment_text'].str.len() > 0]
        df = df_sample
        
        print(f"Dataset final: {len(df)} commentaires")
    
    # Sauvegarde des donnees preparees
    output_blob = bucket.blob(f'{output_path}/prepared_data.csv')
    output_blob.upload_from_string(df.to_csv(index=False))
    
    print("Données préparées et sauvegardées")

@component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "google-cloud-storage", "joblib"]
)
def train_model_op(
    project_id: str,
    bucket_name: str,
    data_path: str,
    model_path: Output[Model]
):
    """Entrainement du modele SVM - utilise la logique de SVM.py"""
    import pandas as pd
    import pickle
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    from google.cloud import storage
    import io
    
    print("Chargement des données...")
    
    # Chargement des donnees depuis Cloud Storage
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name.replace('gs://', ''))
    blob = bucket.blob(f'{data_path}/prepared_data.csv')
    
    content = blob.download_as_text()
    df = pd.read_csv(io.StringIO(content))
    
    texts = df['comment_text'].astype(str).tolist()
    labels = df['toxic'].tolist()
    
    print(f"Données chargées: {len(texts)} commentaires")
    print(f"Distribution: {df['toxic'].value_counts().to_dict()}")
    
    # PIPELINE TF-IDF + SVM - logique identique a SVM.py
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
    print("Début de l'entraînement SVM...")
    pipeline.fit(texts, labels)
    
    # SAUVEGARDER LE MODÈLE
    print("Sauvegarde du modèle...")
    model_blob = bucket.blob('models/svm_model_vertex.pkl')
    model_data = pickle.dumps(pipeline)
    model_blob.upload_from_string(model_data)
    
    # Metadonnees pour Vertex AI
    model_path.uri = f"gs://{bucket_name}/models/svm_model_vertex.pkl"
    
    print("ENTRAÎNEMENT SVM TERMINÉ!")
    print("Modèle sauvegardé dans Cloud Storage")

@component(
    base_image="python:3.11", 
    packages_to_install=["pandas", "scikit-learn", "google-cloud-storage", "joblib"]
)
def evaluate_model_op(
    project_id: str,
    bucket_name: str, 
    model_path: Input[Model],
    data_path: str
) -> float:
    """Evaluation du modele"""
    import pandas as pd
    import pickle
    from sklearn.metrics import accuracy_score
    from google.cloud import storage
    import io
    
    print("Evaluation du modele...")
    
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name.replace('gs://', ''))
    
    # Chargement du modele
    model_blob = bucket.blob('models/svm_model_vertex.pkl')
    model_data = model_blob.download_as_bytes()
    model = pickle.loads(model_data)
    
    # Chargement des donnees de test
    data_blob = bucket.blob(f'{data_path}/prepared_data.csv')
    content = data_blob.download_as_text()
    df = pd.read_csv(io.StringIO(content))
    
    # Evaluation
    texts = df['comment_text'].astype(str).tolist()
    labels = df['toxic'].tolist()
    
    predictions = model.predict(texts)
    accuracy = accuracy_score(labels, predictions)
    
    print(f"Accuracy du modele: {accuracy:.4f}")
    return accuracy

@dsl.pipeline(
    name="toxic-detection-training-pipeline",
    pipeline_root=PIPELINE_ROOT
)
def training_pipeline(
    project_id: str = PROJECT_ID,
    bucket_name: str = f"mlops-models-{PROJECT_ID}"
):
    """Pipeline complet d'entrainement"""
    
    # Etape 1: Preparation des donnees
    prepare_task = prepare_data_op(
        project_id=project_id,
        bucket_name=bucket_name,
        output_path="processed"
    )
    
    # Etape 2: Entrainement
    train_task = train_model_op(
        project_id=project_id,
        bucket_name=bucket_name,
        data_path="processed",
        model_path=dsl.OutputPath(Model)
    )
    train_task.after(prepare_task)
    
    # Etape 3: Evaluation
    eval_task = evaluate_model_op(
        project_id=project_id,
        bucket_name=bucket_name,
        model_path=train_task.outputs['model_path'],
        data_path="processed"
    )
    
    # Logique conditionnelle: deployer si accuracy > 0.8
    with dsl.Condition(eval_task.output > 0.8):
        print("Modele accepte pour deploiement")

def run_vertex_pipeline():
    """
    Fonction simplifiee pour lancer le pipeline Vertex AI
    Basee sur l'exemple d'integration Cloud Build
    """
    print(f"Demarrage pipeline Vertex AI pour projet: {PROJECT_ID}")
    
    # Initialisation de Vertex AI
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,  # europe-west1 par defaut
    )
    
    # Compilation du pipeline si necessaire
    from kfp.v2 import compiler
    template_path = f"gs://{PROJECT_ID}-vertex-pipelines/toxic_training_pipeline.json"
    
    try:
        # Essayer de compiler le pipeline
        compiler.Compiler().compile(
            pipeline_func=training_pipeline,
            package_path="toxic_training_pipeline.json"
        )
        
        # Upload vers Cloud Storage pour reutilisation
        from google.cloud import storage
        client = storage.Client()
        bucket_name = f"{PROJECT_ID}-vertex-pipelines"
        
        try:
            bucket = client.bucket(bucket_name)
            if not bucket.exists():
                bucket = client.create_bucket(bucket_name, location='US')
            
            blob = bucket.blob("toxic_training_pipeline.json")
            blob.upload_from_filename("toxic_training_pipeline.json")
            print(f"Pipeline compile et uploade: {template_path}")
        except Exception as e:
            print(f"Upload pipeline: {e}, utilisation locale")
            template_path = "toxic_training_pipeline.json"
            
    except Exception as e:
        print(f"Erreur compilation: {e}")
        return False
    
    # Création de la définition du job pipeline
    pipeline_job = aiplatform.PipelineJob(
        display_name="mlops-toxic-detection-pipeline",
        template_path=template_path,
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "project_id": PROJECT_ID,
            "bucket_name": f"mlops-models-{PROJECT_ID}",
        },
        enable_caching=False
    )
    
    # Soumission du job pipeline qui lance l'exécution
    try:
        pipeline_job.submit()
        print("Pipeline Vertex AI declenche avec succes.")
        print(f"Monitoring: https://console.cloud.google.com/vertex-ai/pipelines/runs")
        return True
    except Exception as e:
        print(f"Erreur soumission pipeline: {e}")
        return False

def main():
    """Point d'entree principal"""
    success = run_vertex_pipeline()
    if success:
        print("Pipeline Vertex AI lance avec succes")
    else:
        print("Echec du lancement du pipeline Vertex AI")
        exit(1)

def run_vertex_with_mlflow():
    """
    VERTEX AI + MLFLOW INTEGRES - Ils travaillent ENSEMBLE
    """
    print("VERTEX AI + MLFLOW ENSEMBLE")
    print("MLflow: Tracking des experiences")  
    print("Vertex AI: Compute distribue")
    
    try:
        # 1. Configuration MLflow
        import mlflow
        mlflow.set_tracking_uri(f"gs://mlops-models-{PROJECT_ID or os.getenv('PROJECT_ID')}/mlflow")
        mlflow.set_experiment("vertex-ai-training")
        
        with mlflow.start_run(run_name=f"vertex-training-{os.getenv('BUILD_ID', 'local')}"):
            
            # 2. Initialisation Vertex AI
            aiplatform.init(
                project=PROJECT_ID or os.getenv('PROJECT_ID'),
                location=REGION,
            )
            
            # 3. Log des paramètres dans MLflow
            mlflow.log_param("compute_platform", "vertex_ai")
            mlflow.log_param("region", REGION)
            mlflow.log_param("pipeline_type", "distributed")
            
            # 4. Compilation et lancement Vertex AI
            from kfp.v2 import compiler
            compiler.Compiler().compile(
                pipeline_func=training_pipeline,
                package_path="vertex_mlflow_pipeline.json"
            )
            
            # 5. Job Vertex AI avec tracking MLflow
            job = aiplatform.PipelineJob(
                display_name=f"mlflow-vertex-{os.getenv('BUILD_ID', 'local')}",
                template_path="vertex_mlflow_pipeline.json", 
                pipeline_root=PIPELINE_ROOT,
                parameter_values={
                    "project_id": PROJECT_ID or os.getenv('PROJECT_ID'),
                    "bucket_name": f"mlops-models-{PROJECT_ID or os.getenv('PROJECT_ID')}",
                    "mlflow_tracking_uri": f"gs://mlops-models-{PROJECT_ID or os.getenv('PROJECT_ID')}/mlflow"
                }
            )
            
            # 6. Lancement avec suivi MLflow
            job.submit()
            
            # 7. Log des infos du job dans MLflow
            mlflow.log_param("vertex_job_name", job.display_name)
            mlflow.log_param("pipeline_root", PIPELINE_ROOT)
            mlflow.set_tag("platform", "vertex_ai_distributed")
            
            print("Vertex AI + MLflow lances ensemble")
            return True
            
    except Exception as e:
        print(f"Erreur Vertex AI + MLflow: {e}")
        # Fallback intelligent
        print("Fallback vers MLflow seul...")
        try:
            import subprocess
            result = subprocess.run(["python", "mlops/training.py"], 
                                  capture_output=True, text=True, cwd=".")
            if result.returncode == 0:
                print("Fallback MLflow reussi") 
                return True
            else:
                print(f"Fallback echoue: {result.stderr}")
                return False
        except Exception as fallback_error:
            print(f"Erreur fallback: {fallback_error}")
            return False

if __name__ == "__main__":
    # Nouveau: Toujours MLflow + Vertex AI ensemble
    mode = os.getenv('VERTEX_MODE', 'integrated')
    
    if mode == 'integrated':
        success = run_vertex_with_mlflow()
    else:
        success = run_vertex_pipeline()
    
    if not success:
        exit(1)