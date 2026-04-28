import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

print("=== ENTRAÎNEMENT BERT - TOXIC DETECTION ===")

# CHARGER LES DONNÉES D'ENTRAÎNEMENT
print("Chargement des données...")
df = pd.read_csv('train_toxic_10k.csv')
texts = df['comment_text'].astype(str).tolist()
labels = df['toxic'].tolist()

print(f"Données chargées: {len(texts)} commentaires")
print(f"Distribution: {df['toxic'].value_counts().to_dict()}")

# MODÈLE BERT
print("Initialisation du modèle DistilBERT...")
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# TOKENISATION
print("Tokenisation des textes...")
train_encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')

# DATASET D'ENTRAÎNEMENT
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': labels
})

# CONFIGURATION D'ENTRAÎNEMENT
training_args = TrainingArguments(
    output_dir='./model/bert_model',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    save_strategy="epoch",
    report_to=None
)

# ENTRAÎNEUR
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# LANCER L'ENTRAÎNEMENT
print(" Début de l'entraînement BERT...")
trainer.train()

# SAUVEGARDER LE MODÈLE
print(" Sauvegarde du modèle...")
trainer.save_model('./model/bert_model')
tokenizer.save_pretrained('./model/bert_model')

print(" ENTRAÎNEMENT BERT TERMINÉ!")
print(" Modèle sauvegardé dans ./model/bert_model")