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


def _guessed_title(l: str):
    rules_list = [_rule_is_title, _rule_contains_invalids]
    # invalid_chars = ['{', '}', '=', '+', '>', '<', '/', '\\', '|']
    # contains_invalids = any([True for i in invalid_chars if i in l])
    # is_one_sentence = len(l.split('.')) == 1
    # is_one_word = len(l.split()) == 1
    # is_short = len(l.split()) <= 3
    # is_title = l.istitle()
    # is_alnum = l.isalnum()
    # (is_title or is_alnum) and not contains_invalids
    result = all([rule(l) for rule in rules_list])
    return result


def _guess_class(texts: list):
    result = []
    for t in texts:
        if _guessed_title(t):
            result.append(PagePart('title', t))
        else:
            result.append(PagePart('text', t))
    return result


def _count_label(inp: list, label: str):
    return sum([1 for i in inp if i.label == label])


if __name__ == "__main__":
    files = _read_file_list()
    ground_truth = []
    fscore = FScore()
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

    # todo divide training/evaluation

    just_texts = [t.text for t in ground_truth]

    # create rule-based system
    text_guessed = _guess_class(just_texts)

    # evaluate quality
    for truth, guess in zip(ground_truth, text_guessed):
        if truth.label == 'title' and guess.label == 'title':
            fscore.true_positive += 1
        if truth.label == 'text' and guess.label == 'title':
            fscore.false_positive += 1
        if truth.label == 'title' and guess.label == 'text':
            fscore.false_negative += 1
        if truth.label == 'text' and guess.label == 'text':
            fscore.true_negative += 1

    precision = fscore.true_positive / float(fscore.true_positive + fscore.false_positive)
    recall = fscore.true_positive / float(fscore.true_positive + fscore.false_negative)

    accuracy = (fscore.true_positive + fscore.true_negative) / float(all_docs_amount)

    fscore_result = 2 * (precision * recall) / (precision + recall)

    print('Precision {}'.format(precision))
    print('Recall {}'.format(recall))
    print('F-score {}'.format(fscore_result))
    print('Accuracy {}'.format(accuracy))

