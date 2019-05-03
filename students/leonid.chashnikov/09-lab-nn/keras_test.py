from keras import metrics
from keras.models import Sequential
from keras.layers import Dense

from collections import OrderedDict

from read_files import read_file
from read_files import debug
from read_files import files

import numpy as np

from sklearn.feature_extraction import DictVectorizer


class Actions:
    SHIFT = 0
    RIGHT = 1
    LEFT = 2
    REDUCE = 3


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


def generate_features(stack, queue):
    feature_dict = dict()
    if len(stack) > 0:
        top_stack = stack[-1]
        feature_dict['stk_0_form'] = top_stack['form']
        feature_dict['stk_0_lemma'] = top_stack['lemma']
        feature_dict['stk_0_postag'] = top_stack['upostag']
        if top_stack["feats"]:
            for k, v in top_stack["feats"].items():
                feature_dict["stk_0_feats_" + k] = v
    #     stk - ldep, rdep
    if len(stack) > 1:
        feature_dict['stk_1_postag'] = stack[-2]['upostag']
    if len(queue) > 0:
        top_queue = queue[0]
        feature_dict['queue_0_form'] = top_queue['form']
        feature_dict['queue_0_lemma'] = top_queue['lemma']
        feature_dict['queue_0_postag'] = top_queue['upostag']
        if top_queue["feats"]:
            for k, v in top_queue["feats"].items():
                feature_dict["queue_0_feats_" + k] = v
        #     stk - ldep, rdep
    if len(queue) > 1:
        feature_dict['queue_1_form'] = queue[1]['form']
        feature_dict['queue_1_postag'] = queue[1]['upostag']
    if len(queue) > 2:
        feature_dict['queue_2_postag'] = queue[2]['upostag']
    if len(queue) > 3:
        feature_dict['queue_3_postag'] = queue[3]['upostag']
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
    labels, features = [], []
    for tree in trees:
        # stack - empty, queue - all words, relations - empty list
        # give all queue to oracle, it generates steps, then make step executes them and creates relations
        stack, queue, relations, steps = [ROOT], tree[:], [], []
        while len(queue) > 0 and len(stack) > 0:
            top_queue = queue[0]
            step = oracle(stack, top_queue, relations)
            labels.append(step)
            features.append(generate_features(stack, queue))
            make_step(step, stack, queue, relations)

    return labels, features


if __name__ == "__main__":
    train_labels, train_features = _get_labels_features(files[0])
    train_labels = np.array(train_labels)

    test_labels, test_features = _get_labels_features(files[1])
    test_labels = np.array(test_labels)

    vectorizer = DictVectorizer()
    train_features = vectorizer.fit_transform(train_features)
    test_features = vectorizer.transform(test_features)

    # KERAS
    model = Sequential()

    # model.add(Dense(units=256, activation='relu', input_dim=train_features.shape[1]))
    # model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu', input_dim=train_features.shape[1]))
    model.add(Dense(units=4, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', metrics.categorical_accuracy])

    # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
    model.fit(train_features, train_labels, epochs=3, batch_size=50)

    loss_and_metrics = model.evaluate(test_features, test_labels, batch_size=128)
    print(model.metrics_names)
    print(loss_and_metrics)
