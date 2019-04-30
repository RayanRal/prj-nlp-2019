import pickle

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.model_selection import ShuffleSplit, cross_validate
from sklearn.naive_bayes import GaussianNB
from scipy.sparse import hstack


def _read_comments():
    with open('./data/processed_comments_lemmatized.p', 'rb') as f:
        result = pickle.load(f)
    return result


def _remove_neutral(comments):
    for c in comments:
        if c['label'] == 'neu':
            c['label'] = 'neg'
    return comments


if __name__ == "__main__":
    comments = _read_comments()
    comments = _remove_neutral(comments)
    text = [c['text'] for c in comments]
    tone = [c['tone'] for c in comments]
    target = [c['label'] for c in comments]

    count_vect = CountVectorizer()
    features_vector = count_vect.fit_transform(text)
    tone = np.array(tone).reshape(-1, 1)
    features_vector = hstack((features_vector, tone))

    split = ShuffleSplit(test_size=0.3, train_size=0.7, random_state=42)
    scoring = ['precision_macro', 'recall_macro', 'f1_macro']

    clf_nb = GaussianNB()
    scores = cross_validate(clf_nb, features_vector.toarray(), target, scoring=scoring, cv=split)
    print('GaussianNB:\n\tf1 {}, precision {}, recall {}'
          .format(round(scores['test_f1_macro'].mean(), 4), round(scores['test_precision_macro'].mean(), 4), round(scores['test_recall_macro'].mean(), 4)))

    clf_per = Perceptron()
    scores = cross_validate(clf_per, features_vector.toarray(), target, scoring=scoring, cv=split)
    print('Perceptron:\n\tf1 {}, precision {}, recall {}'
          .format(round(scores['test_f1_macro'].mean(), 4), round(scores['test_precision_macro'].mean(), 4), round(scores['test_recall_macro'].mean(), 4)))

    clf_lr = LogisticRegression()
    scores = cross_validate(clf_lr, features_vector.toarray(), target, scoring=scoring, cv=split)
    print('LogReg:\n\tf1 {}, precision {}, recall {}'
          .format(round(scores['test_f1_macro'].mean(), 4), round(scores['test_precision_macro'].mean(), 4), round(scores['test_recall_macro'].mean(), 4)))

