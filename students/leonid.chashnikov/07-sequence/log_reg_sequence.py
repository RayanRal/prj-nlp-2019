import json

import spacy
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import baseline_solution
from test_data_tokenizer import WordTokenizer
from sklearn.metrics import classification_report


nlp = spacy.load('en_core_web_lg', disable=['textcat', 'ner'], max_length=10000001)


def _get_feature_dict(token, prev_token, next_token):
    result_dict = dict()
    token_str = str(token)
    result_dict['word_lower'] = token_str.lower()
    result_dict['word_lemma'] = token.lemma_
    result_dict['is_capitalized'] = token.is_title
    result_dict['is_punct'] = token.is_punct
    result_dict['is_uppercase'] = token.is_upper

    result_dict['word_pos'] = token.pos_
    result_dict['word_dep'] = token.dep_
    result_dict['word_head'] = str(token.head)
    result_dict['word_n_lefts'] = token.n_lefts
    result_dict['word_n_rights'] = token.n_rights
    # todo left_edge, right_edge

    if prev_token:
        prev_token_str = str(prev_token)
        result_dict['prev_word_lower'] = prev_token_str.lower()
        result_dict['prev_word_lemma'] = prev_token.lemma_
        result_dict['prev_pos'] = prev_token.pos_
    else:
        result_dict['prev_word_lower'] = ''
        result_dict['prev_word_lemma'] = ''
        result_dict['prev_pos'] = 'None'

    if next_token:
        next_token_str = str(next_token)
        result_dict['next_word_lower'] = next_token_str.lower()
        result_dict['next_word_lemma'] = next_token.lemma_
        result_dict['next_pos'] = next_token.pos_
    else:
        result_dict['next_word_lower'] = ''
        result_dict['next_word_lemma'] = ''
        result_dict['next_pos'] = 'None'
    return result_dict


def _get_features(tokens: list):
    features = []
    for index in range(len(tokens)):
        token = tokens[index]
        prev_token = tokens[index-1] if index > 0 else None
        next_token = tokens[index+1] if index < len(tokens) - 1 else None
        result_dict = _get_feature_dict(token, prev_token, next_token)
        features.append(result_dict)
    return features


def _read_test_data():
    with open('../../../tasks/07-language-as-sequence/run-on-test.json', 'r') as f:
        data = json.load(f)
    return data


def _corpus_to_list(sents):
    return [' '.join(sent) for sent in sents if len(sent) > 2]


def _preprocess_ptb(sents):
    result = []
    for sent in sents:
        sentence = ' '.join([w[0] for w in sent])
        result.append(sentence)
    return result


def _filter_out_stop_sign(sent):
    if str(sent[-1]) == '.' or str(sent[-1]) == '?' or str(sent[-1]) == '!':  # todo maybe there's full list somewhere
        return sent[:-1]
    else:
        return sent


def _transform_test_data(test_data):
    nlp.tokenizer = WordTokenizer(nlp.vocab)
    raw_tokens = []
    for sentence in test_data:
        sent = ' '.join([word[0] for word in sentence])
        sent = list(nlp(sent))
        raw_tokens.append(sent)
    labels = [word[1] for sentence in test_data for word in sentence]
    return _flatten(raw_tokens), labels


def _flatten(l):
    flat_list = [item for sublist in l for item in sublist]
    return flat_list


def _remove_big_letter(sent):
    import random
    random = random.uniform(0, 1)
    if random > 0.5:
        sent[1] = sent[1].lower()
    return sent[1]


def _merge_sentences_by_two_and_tag(sents):
    raw_tokens = []
    labels = []
    sents = ' '.join(sents)[0:700000]
    # split into several groups
    sents = list(nlp(sents).sents)
    # try tokenizing whole text
    # for index in range(start=0, stop=len(sents) - 1, step=2):
    for index in range(len(sents) - 1):
        sent_1 = _filter_out_stop_sign(sents[index])
        sent_2 = sents[index + 1]# _remove_big_letter(sents[index + 1])  # randomly remove Big letter at sentence start

        # doc_1 = nlp(sent_1)
        # doc_2 = nlp(sent_2)

        merged = list(sent_1) + list(sent_2)
        raw_tokens.append(merged)

        sent_label = [False] * len(merged)
        sent_label[len(sent_1) - 1] = True
        labels.append(sent_label)

    raw_tokens = _flatten(raw_tokens)
    labels = _flatten(labels)
    return raw_tokens, labels


def _download_train():
    # import nltk
    # nltk.download('treebank')
    # nltk.download('brown')

    from nltk.corpus import treebank
    from nltk.corpus import brown

    brown_sents = _corpus_to_list(brown.sents())  # looks ok, but needs cleaning from special symbols, list of sentence strings
    ptb_sents = list()  #  _preprocess_ptb(treebank.tagged_sents())  # - not raw sentences :( needs cleaning from special symbols and tags

    return brown_sents + ptb_sents


if __name__ == "__main__":
    # Baseline
    test_data = _read_test_data()
    # baseline_solution.evaluate_baseline(test_data)

    # Test data
    test_raw_tokens, test_labels = _transform_test_data(test_data)
    test_features = _get_features(test_raw_tokens)
    print('Test features {}, test labels {}'.format(len(test_features), len(test_labels)))

    # Train data
    train_data = _download_train()
    raw_train_tokens, train_labels = _merge_sentences_by_two_and_tag(train_data)
    train_features = _get_features(raw_train_tokens)
    print('Train features {}, train labels {}'.format(len(train_features), len(train_labels)))

    # DictVectorizer
    dict_vect = DictVectorizer()
    train_features = dict_vect.fit_transform(train_features)
    test_features = dict_vect.transform(test_features)

    # LogisticRegression
    clf_lr = LogisticRegression(random_state=42, solver="sag")
    clf_lr.fit(train_features, train_labels)
    predicted_labels = clf_lr.predict(test_features)
    scores = classification_report(test_labels, predicted_labels)

    print(scores)
