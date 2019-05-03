from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from scipy.sparse import issparse

class L1NormVectorizer(BaseEstimator, TransformerMixin):
    """
    Adds some features about passed text
    """
    def __init__(self):
        self.norm = 0.0

    def fit(self, X, y=None):
        self.norm = abs(X).sum()
        return self

    def transform(self, X):
        if self.norm == 0.0:
            return X
        return X / self.norm
