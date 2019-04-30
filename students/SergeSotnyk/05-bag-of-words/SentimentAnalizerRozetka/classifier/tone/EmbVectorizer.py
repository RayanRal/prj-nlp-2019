from bpemb import BPEmb
from nltk import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class EmbVectorizer(BaseEstimator, TransformerMixin):
    """
    Adds embedding features for passed text
    """
    bpemb = BPEmb(lang="uk")

    def __init__(self):
        ...

    def fit(self, documents, y=None):
        return self

    def calc_emb(self, text):
        res = np.zeros(EmbVectorizer.bpemb.vectors.shape[1], dtype=np.float32)
        # tokens = word_tokenize(text)
        # for t in tokens:
        embs = EmbVectorizer.bpemb.embed(text)
        for e in embs:
            res += e
        n = len(embs)
        if n:
            res /= n
        return res

    def transform(self, X):
        res = []
        for row in range(0, len(X)):
            res.append(self.calc_emb(X[row]))
        np_res = np.array(res, dtype=float)
        return np_res
