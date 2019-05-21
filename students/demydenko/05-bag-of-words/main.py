import pandas as pd
import re
import os, sys
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn import tree

from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import importlib

importlib.import_module('data_loader', package=None)
from data_loader import load_and_cook_data


# ----------------------------------------------------
#                головне рішення
# ----------------------------------------------------
# вивід пандас в консолі
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


x_train, x_test, y_train, y_test = load_and_cook_data(load_from_file=True)

# пробував поєднувати фічі з бейслайну та фічі з CountVectorizer, виходило гірше ніж просто CountVectorizer
# біграми теж не покращували якість
vectorizer_c = CountVectorizer(ngram_range=(1, 1))

tokenize = vectorizer_c.build_tokenizer()

# залишаємо лише текст та частину тесту де недоліки та переваги.
x_train = [d['text'] + ' ' + str(d['text_minus'])[12:] + ' ' + str(d['text_plus'])[13:] for d in x_train]
x_test = [d['text'] + ' ' + str(d['text_minus'])[12:] + ' ' + str(d['text_plus'])[13:] for d in x_test]


# та передаємо на векторизацію
vec_train_tokens_transformed = vectorizer_c.fit_transform(x_train)


vec_test_tokens = vectorizer_c.transform(x_test)

clf = LogisticRegression()# - найкращий результат
# clf = MultinomialNB()
# clf = svm.SVC(kernel= 'linear')
# clf = tree.DecisionTreeClassifier()
# clf = BernoulliNB()
# clf = SGDClassifier()

clf.fit(vec_train_tokens_transformed, y_train)
predictions = clf.predict(vec_test_tokens)
print(classification_report(y_test, predictions))

print("Accuracy:", accuracy_score(y_test, predictions))

# -1 негативний відгук, 1 - позитивний

# LogisticRegression
#               precision    recall  f1-score   support
#
#           -1       0.62      0.35      0.44        23
#            1       0.89      0.96      0.92       127
#
#    micro avg       0.87      0.87      0.87       150
#    macro avg       0.75      0.65      0.68       150
# weighted avg       0.85      0.87      0.85       150
#
# Accuracy: 0.8666666666666667

pass
