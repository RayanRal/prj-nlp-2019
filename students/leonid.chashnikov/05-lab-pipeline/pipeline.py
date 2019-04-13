from sklearn import tree
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate

import os
import re


PATH = './data/mds/'
TITLE_REGEX = r'^#+ '


class PagePart:
    label: str
    text: str

    def __init__(self, label, text):
        self.label = label
        self.text = text


class FScore:
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0


def _read_file_list():
    files = []
    for r, d, f in os.walk(PATH):
        for file in f:
            if '.md' in file:
                files.append(os.path.join(r, file))
    return files


def _is_title(l: str):
    return re.match(TITLE_REGEX, l)


def _parse_page_part(l: str):
    if _is_title(l):
        l = re.sub(TITLE_REGEX, '', l)
        return PagePart('title', l)
    else:
        return PagePart('text', l)


def _rule_is_title(s: str):
    return s.istitle()


def _rule_contains_invalids(s: str):
    invalid_chars = ['{', '}', '=', '+', '>', '<', '/', '\\', '|']
    contains_invalids = any([True for i in invalid_chars if i in l])
    return not contains_invalids


def _rule_is_alnum(s: str):
    return s.isalnum()


def _rule_count_words(s: str):
    return len(s.split())


def _ends_with_dot(s: str):
    if len(s) > 0:
        return s[-1] == '.'
    else:
        return False


def _count_label(inp: list, label: str):
    return sum([1 for i in inp if i.label == label])


def _prepare_data(data):
    result = []
    for t in data:
        is_title = _rule_is_title(t.text)
        contains_invalids = _rule_contains_invalids(t.text)
        is_alnum = _rule_is_alnum(t.text)
        word_count = _rule_count_words(t.text)
        is_end_with_dot = _ends_with_dot(t.text)
        result.append([is_title, contains_invalids, word_count, is_alnum, is_end_with_dot])
    return result


if __name__ == "__main__":
    files = _read_file_list()
    ground_truth = []

    # parse md into ground truth
    for input_file in files:
        with open(input_file) as inp:
            lines = [line.rstrip('\n') for line in inp]
            for l in lines:
                if l:
                    ground_truth.append(_parse_page_part(l))

    # Debug info
    title_amount = _count_label(ground_truth, 'title')
    text_amount = _count_label(ground_truth, 'text')
    all_docs_amount = len(ground_truth)
    print('Amount of titles {}'.format(title_amount))
    print('Amount of text {}'.format(text_amount))
    print('Amount of all docs {}'.format(all_docs_amount))
    print()

    target = [1 if t.label == 'title' else 0 for t in ground_truth]
    data = _prepare_data(ground_truth)

    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3, random_state=42)
    split = ShuffleSplit(test_size=0.3, train_size=0.7, random_state=42)
    scoring = ['precision_macro', 'recall_macro']

    clf_tree = tree.DecisionTreeClassifier()
    scores = cross_validate(clf_tree, data, target, scoring=scoring, cv=split)
    print('Decision Tree:\n\tprecision {}, recall {}'
          .format(scores['test_precision_macro'].mean(), scores['test_recall_macro'].mean()))
    clf_tree = clf_tree.fit(data_train, target_train)
    print(clf_tree.feature_importances_)

    clf_rf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42)
    scores = cross_validate(clf_rf, data, target, scoring=scoring, cv=split)
    print('Random Forest:\n\tprecision {}, recall {}'
          .format(scores['test_precision_macro'].mean(), scores['test_recall_macro'].mean()))
    clf_rf = clf_rf.fit(data_train, target_train)
    print(clf_rf.feature_importances_)

    # clf_nb = GaussianNB()
    # scores = cross_validate(clf_nb, data, target, scoring=scoring, cv=split)
    # print('Naive Bayes:\n\tprecision {}, recall {}'
    #       .format(scores['test_precision_macro'].mean(), scores['test_recall_macro'].mean()))

