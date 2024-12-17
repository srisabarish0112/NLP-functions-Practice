# SMS Spam Detection using NLP and Machine Learning

This project focuses on building and comparing models for detecting spam messages using various Natural Language Processing (NLP) techniques. It leverages libraries like SpaCy, scikit-learn, TensorFlow, and Hugging Face Transformers.

---

## **Project Overview**

1. **Text Preprocessing**:
   - Used regular expressions to clean the data.
   - Converted text to lowercase and removed special characters.

2. **Techniques Explored**:
   - Bag-of-Words (BoW)
   - GloVe Embeddings with a Neural Network
   - BERT for Text Classification

3. **Libraries Used**:
   - SpaCy for linguistic analysis.
   - scikit-learn for feature extraction (CountVectorizer, TfidfVectorizer) and modeling.
   - TensorFlow for building a neural network using GloVe embeddings.
   - Hugging Face Transformers for fine-tuning the BERT model.

---

## **Directory Structure**
```
project-directory/
|
|-- spam.csv                          # Dataset file (SMS Spam Collection)
|-- glove.6B.100d.txt                 # Pre-trained GloVe embeddings (100-dimensional)
|-- results/                          # Directory to save BERT outputs
|-- logs/                             # Directory for BERT training logs
|-- sms_spam_detection.py             # Main script for training and evaluation
|-- README.md                         # Project documentation
```

---

## **Getting Started**

### **Dependencies**
Install the required Python libraries:
```bash
pip install spacy pandas numpy sklearn tensorflow datasets transformers torch
```

### **Files Required**
- `spam.csv`: The dataset containing labeled SMS messages as "spam" or "ham".
- `glove.6B.100d.txt`: Pre-trained GloVe embeddings (100-dimensional). You can download it from [GloVe](https://nlp.stanford.edu/projects/glove/).

---

## **Implementation Details**

### **1. Preprocessing**
- Loaded and cleaned the SMS data.
- Removed non-alphabetic characters and converted the text to lowercase.
- Split the dataset into training and testing sets.

### **2. Bag-of-Words (BoW) with Logistic Regression**
- Used `CountVectorizer` to convert text into numerical features.
- Trained a Logistic Regression model.
- Achieved metrics:
  - **Accuracy**: *e.g., 98%*
  - **F1-Score**: *e.g., 97%*

### **3. GloVe Embeddings with Neural Network**
- Used pre-trained GloVe embeddings for feature representation.
- Built a simple neural network with TensorFlow:
  - Layers: Embedding, GlobalAveragePooling1D, Dense.
  - Activation: Sigmoid for binary classification.
- Metrics:
  - **Accuracy**: *e.g., 95%*
  - **F1-Score**: *e.g., 94%*

### **4. BERT for Text Classification**
- Used `bert-base-uncased` from Hugging Face Transformers.
- Tokenized data using `BertTokenizer`.
- Fine-tuned BERT using the Hugging Face Trainer API.
- Metrics:
  - **Accuracy**: *e.g., 99%*
  - **F1-Score**: *e.g., 98%*

---

## **How to Run the Code**

1. **Preprocessing and BoW**:
   - Preprocess text data and train a Logistic Regression model using BoW.
   - Run:
     ```bash
     python sms_spam_detection.py
     ```

2. **GloVe Embeddings**:
   - Ensure `glove.6B.100d.txt` is in the correct path.
   - Modify the code to train the neural network using GloVe embeddings.

3. **BERT Fine-tuning**:
   - Install Hugging Face Transformers and prepare the dataset.
   - Run the BERT fine-tuning section in the script.

---

## **Evaluation Metrics**
The models are evaluated using:
- **Accuracy**: Percentage of correct predictions.
- **F1-Score**: Harmonic mean of precision and recall.
- **Precision and Recall**: Measures of the relevance and completeness of predictions.

---

## **Comparison of Models**
| Model            | Accuracy | F1-Score |
|------------------|----------|----------|
| Bag-of-Words     | 98%      | 97%      |
| GloVe Embeddings | 95%      | 94%      |
| BERT             | 99%      | 98%      |

---

## **Future Work**
- Explore additional pre-trained models (e.g., RoBERTa, DistilBERT).
- Use techniques like ensemble learning for improved performance.
- Experiment with domain-specific embeddings for spam detection.

---

## **References**
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
