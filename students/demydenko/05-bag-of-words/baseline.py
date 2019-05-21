import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import importlib
importlib.import_module('data_loader', package=None)
from data_loader import load_and_cook_data


# ----------------------------------------------------
#                бейслайн рішення
# ----------------------------------------------------
# вивід пандас в консолі
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# завантаження відгуків про ноутбуки, фільтрація, базові фічі, розбивка на train test
x_train, x_test, y_train, y_test = load_and_cook_data(load_from_file=True)

vectorizer = DictVectorizer()


# vec_train_tokens = vectorizer.fit(x_train)
vec_train_tokens_transformed = vectorizer.fit_transform(x_train)
# print("\nTotal number of features: ", len(vec_train_tokens.get_feature_names()))

vec_test_tokens = vectorizer.transform(x_test)

clf = LogisticRegression()

clf.fit(vec_train_tokens_transformed, y_train)
predictions = clf.predict(vec_test_tokens)
print(classification_report(y_test, predictions))

print("Accuracy:", accuracy_score(y_test, predictions))

# precision    recall  f1-score   support
#
#           -1       1.00      0.09      0.16        23
#            1       0.86      1.00      0.92       127
#
#    micro avg       0.86      0.86      0.86       150
#    macro avg       0.93      0.54      0.54       150
# weighted avg       0.88      0.86      0.81       150
#
# Accuracy: 0.86
pass