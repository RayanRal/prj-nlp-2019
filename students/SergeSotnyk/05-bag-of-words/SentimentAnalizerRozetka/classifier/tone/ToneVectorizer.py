from nltk import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from tone import ToneDict
from . import UkLemmatizer


class ToneVectorizer(BaseEstimator, TransformerMixin):
    """
    Adds some features about emotional words in text
    """
    _tone_dict = ToneDict()

    def __init__(self):
        ...

    def fit(self, documents, y=None):
        return self

    def calc_tone(self, text):
        tokens = word_tokenize(text)
        acc = {-2.0: 0, -1.0: 0, 0.0: 0, 1.0: 0, 2.0: 0}
        for t in tokens:
            l = UkLemmatizer.lemmatize(t)
            acc[self._tone_dict[l]] += 1
        res = [v for k, v in sorted(acc.items()) if k !=0]
        return np.array(res, dtype=float)

    def transform(self, X):
        res = []
        for row in range(0, len(X)):
            res.append(self.calc_tone(X[row]))
        np_res = np.array(res, dtype=float)
        return np_res
