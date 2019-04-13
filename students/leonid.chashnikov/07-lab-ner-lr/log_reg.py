import pandas as pandas
import pymorphy2
from extract_ner_data import get_train_test
from read_files import get_locations
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report

morph = pymorphy2.MorphAnalyzer(lang='uk')


def _get_feature_dict(token, prev_token):
    result_dict = dict()
    result_dict['word'] = token.lower()
    result_dict['is_capitalized'] = token.istitle()
    result_dict['is_not_punct'] = token.isalnum()
    result_dict['is_uppercase'] = token.isupper()

    result_dict['word_lemma'] = morph.normal_forms(token)[0]
    result_dict['pos'] = str(morph.tag(token)[0].POS)

    if prev_token:
        result_dict['prev_word'] = prev_token.lower()
        result_dict['prev_pos'] = str(morph.tag(prev_token)[0].POS)
    else:
        result_dict['prev_word'] = ''
        result_dict['prev_pos'] = 'None'
    return result_dict


def _get_features(train_tokens, test_tokens):
    train_features = []
    test_features = []

    for index in range(len(train_tokens)):
        token = train_tokens[index]
        if index > 0:
            prev_token = train_tokens[index-1]
        else:
            prev_token = None
        feature_dict = _get_feature_dict(token, prev_token)
        train_features.append(feature_dict)

    for index in range(len(test_tokens)):
        token = test_tokens[index]
        if index > 0:
            prev_token = test_tokens[index-1]
        else:
            prev_token = None
        feature_dict = _get_feature_dict(token, prev_token)
        test_features.append(feature_dict)

    return train_features, test_features


if __name__ == "__main__":
    train_tokens, test_tokens, train_labels, test_labels = get_train_test()
    train_features, test_features = _get_features(train_tokens, test_tokens)

    dict_vect = DictVectorizer()
    train_features = dict_vect.fit_transform(train_features)
    test_features = dict_vect.transform(test_features)

    clf_lr = LogisticRegression(random_state=42, solver="sag", multi_class="multinomial")
    clf_lr.fit(train_features, train_labels)
    predicted_labels = clf_lr.predict(test_features)
    scores = classification_report(test_labels, predicted_labels)

    print(scores)
