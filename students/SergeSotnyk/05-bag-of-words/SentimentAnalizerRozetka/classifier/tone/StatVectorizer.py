from nltk import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class StatVectorizer(BaseEstimator, TransformerMixin):
    """
    Adds some features about passed text
    """
    def __init__(self):
        ...

    def fit(self, documents, y=None):
        return self

    def calc_stat(self, text):
        tokens = word_tokenize(text)
        return np.array([len(text), len(tokens), np.mean([float(len(t)) for t in tokens])], dtype=float)

    def transform(self, X):
        res = []
        for row in range(0, len(X)):
            res.append(self.calc_stat(X[row]))
        np_res = np.array(res, dtype=float)
        return np_res
