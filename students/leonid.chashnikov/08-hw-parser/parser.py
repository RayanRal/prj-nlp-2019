from collections import OrderedDict

from tokenize_uk import tokenize_uk
from read_files import read_file
from read_files import debug
from read_files import files
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


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

        # print("Gold relations:")
        # print([(node["id"], node["head"]) for node in tree])
        # print("Retrieved relations:")
        # print(sorted(relations))
    return labels, features


def dep_parse(sentence, oracle, vectorizer, log=False):
    stack, queue, relations = [ROOT], sentence[:], []
    while queue or stack:
        if stack and not queue:
            stack.pop()
        else:
            features = generate_features(stack, queue)
            action = oracle.predict(vectorizer.transform([features]))[0]
            if log:
                print("Stack:", [i["form"]+"_"+str(i["id"]) for i in stack])
                print("Queue:", [i["form"]+"_"+str(i["id"]) for i in queue])
                print("Relations:", relations)
                print(action)
                print("========================")
            # actual parsing
            if action == Actions.SHIFT:
                stack.append(queue.pop(0))
            elif action == Actions.REDUCE:
                stack.pop()
            elif action == Actions.LEFT:
                relations.append((stack[-1]["id"], queue[0]["id"]))
                stack.pop()
            elif action == Actions.RIGHT:
                relations.append((queue[0]["id"], stack[-1]["id"]))
                stack.append(queue.pop(0))
            else:
                print("Unknown action.")
    return sorted(relations)


def _calculate_uas(classifier, vectorizer):
    trees = read_file(files[1])
    total, tp = 0, 0
    for tree in trees:
        tree = [t for t in tree if type(t["id"]) == int]
        golden = [(node["id"], node["head"]) for node in tree]
        try:
            predicted = dep_parse(tree, classifier, vectorizer)
            tp += len(set(golden).intersection(set(predicted)))
        except Exception as e:
            print('Exception {}'.format(e))

        total += len(tree)

    print("Total:", total)
    print("Correctly defined:", tp)
    print("UAS:", round(tp / total, 2))


def _check_own_sents(classifier, vectorizer):
    sentences = []
    for sent in sentences:
        # tokenize with tokenize-uk
        words = tokenize_uk.tokenize_words(sent)
        for w in words:
            # parse with pymorphy
            # sentence - a list of dicts, with keys 'form', 'lemma', upostag, feats
            # convert POS with pymorphy_ud_convert.py
            # pass to dep_parse


if __name__ == "__main__":
    train_labels, train_features = _get_labels_features(files[0])
    test_labels, test_features = _get_labels_features(files[1])

    vectorizer = DictVectorizer()
    train_features = vectorizer.fit_transform(train_features)
    test_features = vectorizer.transform(test_features)

    classifier = LogisticRegression(random_state=42, solver="sag", multi_class="multinomial", max_iter=1000)
    # classifier = DecisionTreeClassifier(random_state=42)
    # classifier = RandomForestClassifier(random_state=42)
    classifier.fit(train_features, train_labels)

    predicted = classifier.predict(test_features)
    print(classification_report(test_labels, predicted))

    _calculate_uas(classifier, vectorizer)
    _check_own_trees(classifier, vectorizer)
