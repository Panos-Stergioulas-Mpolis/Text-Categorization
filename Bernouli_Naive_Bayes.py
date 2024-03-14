import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB  # Change to BernoulliNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay

nltk.download('punkt')
nltk.download('stopwords')

positive_folder = "pos"
negative_folder = "neg"

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

preprocessed_documents = []
labels = []

def preprocess_document(file_path, label):
    with open(file_path, 'r', encoding='utf-8') as file:
        document = file.read()

    tokens = nltk.word_tokenize(document)
    filtered_tokens = [stemmer.stem(token) for token in tokens if token.lower() not in stop_words]
    preprocessed_document = ' '.join(filtered_tokens)

    return preprocessed_document, label


for file_name in os.listdir(positive_folder):
    file_path = os.path.join(positive_folder, file_name)
    preprocessed_doc, label = preprocess_document(file_path, 'positive')
    preprocessed_documents.append(preprocessed_doc)
    labels.append(label)


for file_name in os.listdir(negative_folder):
    file_path = os.path.join(negative_folder, file_name)
    preprocessed_doc, label = preprocess_document(file_path, 'negative')
    preprocessed_documents.append(preprocessed_doc)
    labels.append(label)


corpus = [' '.join(doc.split()) for doc in preprocessed_documents]


vectorizer = CountVectorizer(binary=True)  
X = vectorizer.fit_transform(corpus)
y = np.array(labels)


model = BernoulliNB()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)

all_predictions = []
all_true_labels = []

for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    all_predictions.extend(predictions)
    all_true_labels.extend(y_test)

accuracy = accuracy_score(all_true_labels, all_predictions)
recall = recall_score(all_true_labels, all_predictions, pos_label='positive', average='weighted')
precision = precision_score(all_true_labels, all_predictions, pos_label='positive', average='weighted')
conf_matrix = confusion_matrix(all_true_labels, all_predictions, labels=["positive", "negative"])

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['positive', 'negative'])
disp.plot(cmap='Blues')


print(f"Accuracy: {accuracy}")
print(f"Weighted Mean Recall: {recall}")
print(f"Weighted Mean Precision: {precision}")
print("Confusion Matrix:")
print(conf_matrix)

plt.show()
