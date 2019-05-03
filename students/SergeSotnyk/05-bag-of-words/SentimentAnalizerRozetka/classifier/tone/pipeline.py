from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, Normalizer

from .L1NormScaler import L1NormVectorizer
from .EmbVectorizer import EmbVectorizer
from .StatVectorizer import StatVectorizer
from .ToneVectorizer import ToneVectorizer
from .L2NormScaler import L2NormVectorizer
from .uk_utils import tokenize_lemmatize


def create_pipeline() -> Pipeline:
    def tokenizer(text: str):
        return tokenize_lemmatize(text)

    features = FeatureUnion([
        ('bow_features', Pipeline([
            ('bow', TfidfVectorizer(tokenizer=tokenizer, min_df=10, max_df=0.2)),
            #('scaler', StandardScaler(with_mean=False)),
        ])),
        ('stat_features', Pipeline([
            ('text_stat', StatVectorizer()),
            ('scaler', Normalizer()),
        ])),
        ('emb_features', Pipeline([
            ('embs', EmbVectorizer()),
            ('scaler', L2NormVectorizer()),
        ])),
        ('tone_features', Pipeline([
            ('text_tone', ToneVectorizer()),
            ('scaler', L2NormVectorizer()),
        ]))
    ],
        transformer_weights={
            'bow_features': 5,
            'stat_features': 1,
            'emb_features': 1,
            'tone_features': 1,
        }
    )

    steps = [
        ('feat_union', features),
        # ('vectorize', TfidfVectorizer(tokenizer=tokenizer)),
        # ('reductor', TruncatedSVD(n_components=1000)),
        # ('classifier', SGDClassifier(max_iter=1000, tol=1e-3))
        # ('classifier', RandomForestClassifier())
        ('classifier', LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=1000))
        # ('classifier', BernoulliNB())
    ]

    return Pipeline(steps)
