"""
Cloud Function pour déclencher automatiquement le réentraînement
quand des données sont ajoutées/modifiées dans GCS
"""

import os
from google.cloud import build_v1


def trigger_retraining(event, context):
    """
    Déclenché par un événement GCS (ajout/modification de fichier)
    """
    
    file_name = event['name']
    bucket_name = event['bucket']
    
    print(f" Fichier détecté: gs://{bucket_name}/{file_name}")
    
    # Ne déclencher que pour les fichiers de données (dans le dossier data/)
    if not file_name.startswith('data/'):
        print(f" Ignoré: fichier hors du dossier data/")
        return
    
    
    print(f" Déclenchement du pipeline de réentraînement...")
    
    project_id = os.environ.get('PROJECT_ID', 'simplifia-hackathon')
    
    # Créer un client Cloud Build
    client = build_v1.CloudBuildClient()
    
    # Définir le build
    build = build_v1.Build()
    build.source = build_v1.Source(
        repo_source=build_v1.RepoSource(
            project_id=project_id,
            repo_name='Khadija0203/API-Digital-Social-Score ',
            branch_name='main'
        )
    )
    
    # Utiliser le fichier cloudbuild-retraining.yaml
    build.steps = []  # définis dans cloudbuild-retraining.yaml
    
    # Substitutions pour passer des variables
    build.substitutions = {
        '_TRIGGER_FILE': file_name,
        '_TRIGGER_BUCKET': bucket_name
    }
    
    # Options
    build.options = build_v1.BuildOptions(
        machine_type='N1_HIGHCPU_8',
        logging='CLOUD_LOGGING_ONLY'
    )
    
    # Déclencher le build
    try:
        operation = client.create_build(
            project_id=project_id,
            build=build
        )
        
        build_id = operation.metadata.build.id
        
        print(f" Pipeline déclenché avec succès!")
        print(f" Build ID: {build_id}")
        print(f" URL: https://console.cloud.google.com/cloud-build/builds/{build_id}?project={project_id}")
        
        return {
            'status': 'success',
            'build_id': build_id,
            'trigger_file': file_name
        }
        
    except Exception as e:
        print(f" Erreur lors du déclenchement: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


# Pour tester localement
if __name__ == '__main__':
    # Simuler un événement GCS
    test_event = {
        'name': 'data/new_toxic_comments.csv',
        'bucket': 'mlops-models-simplifia-hackathon'
    }
    
    result = trigger_retraining(test_event, None)
    print(f"\n Résultat: {result}")
