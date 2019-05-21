import os
import re
from gensim.models import KeyedVectors
from nltk import word_tokenize
from langdetect import detect
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


directory = 'students/maksymenko/word-embeddings/1551'
PATH = 'students/maksymenko/word-embeddings/news.lowercased.tokenized.word2vec.300d'
word_vectors = KeyedVectors.load_word2vec_format(PATH, encoding='utf-8', binary=False)


def get_paragraph_vector(tokens):
    paragraph_embedding = 0
    tokens_num = 1
    for token in word_tokenize(tokens):
        token = token.lower()
        try:
            paragraph_embedding += word_vectors[token]
            tokens_num += 1
        except:
            pass
    return paragraph_embedding / tokens_num


filenames = []
texts = []
vectors = []
labels = []
labels_idxs = []


def label_to_idx(label):
    return filenames.index(label)


for filename in os.listdir(directory):
    with open(directory + '/'+ filename, 'r', encoding='utf-8') as file:
        filenames.append(filename)
        paragraphs = re.split(r'\n\d{7}\n', file.read())
        for par in paragraphs:
            try:
                if detect(par) != 'uk':
                    continue
            except:
                continue
            cleaned_par = re.sub(r'\n', ' ', par)
            vector = get_paragraph_vector(cleaned_par)
            if type(vector) != float and len(cleaned_par) > 0:
                labels.append(filename)
                labels_idxs.append(label_to_idx(filename))
                texts.append(cleaned_par)
                vectors.append(vector)

print(len(vectors), len(labels_idxs))


X_train_arr, X_test_arr, y_train_arr, y_test_arr = train_test_split(vectors, labels_idxs, test_size=0.1, random_state=42)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_arr, y_train_arr)
y_pred = knn.predict(X_test_arr)
print(metrics.classification_report(y_test_arr, y_pred))
# F1 35%

# Logistic Regression
clf = LogisticRegression(
    random_state=0,
    solver='lbfgs',
    multi_class='auto',
    max_iter=500
).fit(X_train_arr, y_train_arr)

clf.score(X_test_arr, y_test_arr)
# %60.61
