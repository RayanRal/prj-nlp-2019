from collections import OrderedDict

from read_files import read_file
from read_files import debug
from read_files import files
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier


class Actions:
    SHIFT = 'shift'
    RIGHT = 'right'
    LEFT = 'left'
    REDUCE = 'reduce'


ROOT = OrderedDict([('id', 0), ('form', 'ROOT'), ('lemma', 'ROOT'), ('upostag', 'ROOT'),
                    ('xpostag', None), ('feats', None), ('head', None), ('deprel', None),
                    ('deps', None), ('misc', None)])


def oracle(stack, top_queue, relations):
    top_stack = stack[-1]
    if top_stack and not top_queue:
        return Actions.REDUCE
    elif top_queue['head'] == top_stack['id']:
        return Actions.RIGHT
    elif top_stack['head'] == top_queue['id']:
        return Actions.LEFT
    elif top_stack['id'] in [i[0] for i in relations] and \
            (top_queue['head'] < top_stack['id'] or [s for s in stack if s['head'] == top_queue['id']]):
        return Actions.REDUCE
    else:
        return Actions.SHIFT


def make_step(step, stack, queue, relations):
    top_stack = stack[-1]
    top_queue = queue[0]
    if step == Actions.SHIFT:
        stack.append(queue.pop(0))
    elif step == Actions.REDUCE:
        stack.pop(-1)
    elif step == Actions.LEFT:
        relations.append((top_stack['id'], top_queue['id']))
        stack.pop(-1)
    elif step == Actions.RIGHT:
        relations.append((top_queue['id'], top_stack['id']))
        stack.append(queue.pop(0))


def generate_features(stack, queue, relations):
    feature_dict = dict()
    if len(stack) > 0:
        top_stack = stack[-1]
        feature_dict['stk_0_form'] = top_stack['form']
        feature_dict['stk_0_lemma'] = top_stack['lemma']
        feature_dict['stk_0_postag'] = top_stack['upostag']
    if len(queue) > 0:
        top_queue = queue[0]
        feature_dict['queue_0_form'] = top_queue['form']
        feature_dict['queue_0_lemma'] = top_queue['lemma']
        feature_dict['queue_0_postag'] = top_queue['upostag']
    return feature_dict


def filter_trees(trees):
    result = []
    for tree in trees:
        valid = True
        for node in tree:
            if type(node['head']) != int:
                valid = False
        if valid:
            result.append(tree)
    return result


def _get_labels_features(filename):
    trees = read_file(filename)
    trees = filter_trees(trees)
    train_labels, train_features = [], []
    for tree in trees:
        # stack - empty, queue - all words, relations - empty list
        # give all queue to oracle, it generates steps, then make step executes them and creates relations
        stack, queue, relations, steps = [ROOT], tree[:], [], []
        while len(queue) > 0 and len(stack) > 0:
            top_queue = queue[0]
            step = oracle(stack, top_queue, relations)
            train_labels.append(step)
            train_features.append(generate_features(stack, queue, relations))
            make_step(step, stack, queue, relations)

        # print("Gold relations:")
        # print([(node["id"], node["head"]) for node in tree])
        # print("Retrieved relations:")
        # print(sorted(relations))
    return train_labels, train_features


if __name__ == "__main__":
    train_labels, train_features = _get_labels_features(files[0])
    test_labels, test_features = _get_labels_features(files[1])

    dict_vect = DictVectorizer()
    train_features = dict_vect.fit_transform(train_features)
    test_features = dict_vect.transform(test_features)

    clf_lr = LogisticRegression(random_state=42, solver="sag", multi_class="multinomial", max_iter=1000)
    clf_lr.fit(train_features, train_labels)

    predicted = clf_lr.predict(test_features)
    print(classification_report(test_labels, predicted))

