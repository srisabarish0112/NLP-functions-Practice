import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load dataset
data_path = '/Users/srisabarish/Downloads/SpaCy/spam.csv'
data = pd.read_csv(data_path, encoding='latin-1').iloc[:, [0, 1]]
data.columns = ['label', 'text']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Preprocess text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

data['text'] = data['text'].apply(preprocess_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# Function to evaluate models
def evaluate(y_test, predictions, model_name):
    print(f"=== {model_name} ===")
    print(classification_report(y_test, predictions))
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    return accuracy, precision, recall, f1

# ===== Bag-of-Words =====
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

bow_model = LogisticRegression()
bow_model.fit(X_train_bow, y_train)
predictions_bow = bow_model.predict(X_test_bow)

bow_results = evaluate(y_test, predictions_bow, "Bag-of-Words (BoW)")

# ===== GloVe Embeddings =====
# Load GloVe vectors
embedding_dim = 100
embedding_index = {}
with open('/Users/srisabarish/Downloads/SpaCy/glove.6B/glove.6B.100d.txt', 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

# Function to create embedding matrix
def get_glove_embeddings(texts, embedding_dim):
    embeddings = np.zeros((len(texts), embedding_dim))
    for i, text in enumerate(texts):
        words = text.split()
        word_vectors = [embedding_index[word] for word in words if word in embedding_index]
        if word_vectors:
            embeddings[i] = np.mean(word_vectors, axis=0)
    return embeddings

X_train_glove = get_glove_embeddings(X_train, embedding_dim)
X_test_glove = get_glove_embeddings(X_test, embedding_dim)

glove_model = LogisticRegression()
glove_model.fit(X_train_glove, y_train)
predictions_glove = glove_model.predict(X_test_glove)

glove_results = evaluate(y_test, predictions_glove, "GloVe Embeddings")

# ===== BERT =====
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_data(examples):
    return bert_tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

# Prepare datasets for Hugging Face Trainer
train_data = Dataset.from_pandas(pd.DataFrame({'text': X_train, 'label': y_train}))
test_data = Dataset.from_pandas(pd.DataFrame({'text': X_test, 'label': y_test}))

train_data = train_data.map(tokenize_data, batched=True)
test_data = test_data.map(tokenize_data, batched=True)

train_data = train_data.with_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_data = test_data.with_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=bert_tokenizer,
    compute_metrics=lambda p: {
        'accuracy': (torch.argmax(torch.tensor(p.predictions), dim=1) == torch.tensor(p.label_ids)).float().mean().item()
    }
)

trainer.train()
predictions_bert = trainer.predict(test_data)
predictions_bert = torch.argmax(torch.tensor(predictions_bert.predictions), dim=1).numpy()

bert_results = evaluate(y_test, predictions_bert, "BERT")

# ===== Final Comparison =====
results = {
    "Model": ["BoW", "GloVe", "BERT"],
    "Accuracy": [bow_results[0], glove_results[0], bert_results[0]],
    "Precision": [bow_results[1], glove_results[1], bert_results[1]],
    "Recall": [bow_results[2], glove_results[2], bert_results[2]],
    "F1-Score": [bow_results[3], glove_results[3], bert_results[3]],
}

results_df = pd.DataFrame(results)
print("\n=== Final Comparison ===")
print(results_df)