from collections import OrderedDict
from conllu import parse
from enum import Enum
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

PATH = "UD_Ukrainian-IU"

with open(PATH + "/uk_iu-ud-train.conllu", "r") as f:
    data = f.read()

trees = parse(data)


ROOT = OrderedDict([('id', 0), ('form', 'ROOT'), ('lemma', 'ROOT'), ('upostag', 'ROOT'),
                    ('xpostag', None), ('feats', None), ('head', None), ('deprel', None),
                    ('deps', None), ('misc', None)])


class Actions(str, Enum):
    SHIFT = "shift"
    REDUCE = "reduce"
    RIGHT = "right"
    LEFT = "left"


def oracle(stack, top_queue, relations):
    top_stack = stack[-1]
    if top_stack and not top_queue:
        return Actions.REDUCE
    elif top_queue["head"] == top_stack["id"]:
        return Actions.RIGHT
    elif top_stack["head"] == top_queue["id"]:
        return Actions.LEFT
    elif top_stack["id"] in [i[0] for i in relations] and \
        (top_queue["head"] < top_stack["id"] or
         [s for s in stack if s["head"] == top_queue["id"]]):
        return Actions.REDUCE
    else:
        return Actions.SHIFT


def trace_actions(tree, log=True):
    stack, queue, relations = [ROOT], tree[:], []
    while queue or stack:
        action = oracle(stack if len(stack) > 0 else None,
                        queue[0] if len(queue) > 0 else None,
                        relations)
        if log:
            print("Stack:", [i["form"]+"_"+str(i["id"]) for i in stack])
            print("Queue:", [i["form"]+"_"+str(i["id"]) for i in queue])
            print("Relations:", relations)
            print(action)
            print("========================")
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
    if log:
        print("Gold relations:")
        print([(node["id"], node["head"]) for node in tree])
        print("Retrieved relations:")
        print(sorted(relations))


def extract_features(stack, queue):
    features = dict()
    if len(stack) > 0:
        stack_top = stack[-1]
        features["s0-word"] = stack_top["form"]
        features["s0-lemma"] = stack_top["lemma"]
        features["s0-tag"] = stack_top["upostag"]
        if stack_top["feats"]:
            for k, v in stack_top["feats"].items():
                features["s0-" + k] = v
    if len(stack) > 1:
        features["s1-tag"] = stack_top["upostag"]
    if queue:
        queue_top = queue[0]
        features["q0-word"] = queue_top["form"]
        features["q0-lemma"] = queue_top["lemma"]
        features["q0-tag"] = queue_top["upostag"]
        if queue_top["feats"]:
            for k, v in queue_top["feats"].items():
                features["q0-" + k] = v
    if len(queue) > 1:
        queue_next = queue[1]
        features["q1-word"] = queue_next["form"]
        features["q1-tag"] = queue_next["upostag"]
    if len(queue) > 2:
        features["q2-tag"] = queue[2]["upostag"]
    if len(queue) > 3:
        features["q3-tag"] = queue[3]["upostag"]
    if stack and queue:
        features["distance"] = queue[0]["id"] - stack[-1]["id"]
    return features


def get_data(tree):
    features, labels = [], []
    stack, queue, relations = [ROOT], tree[:], []
    while queue or stack:
        action = oracle(stack if len(stack) > 0 else None,
                        queue[0] if len(queue) > 0 else None,
                        relations)
        features.append(extract_features(stack, queue))
        labels.append(action.value)
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
    return features, labels


train_features, train_labels = [], []
for tree in trees:
    tree_features, tree_labels = get_data([t for t in tree if type(t["id"]) == int])
    train_features += tree_features
    train_labels += tree_labels

with open(PATH + "/uk_iu-ud-test.conllu", "r") as f:
    data = f.read()
test_trees = parse(data)

test_features, test_labels = [], []
for tree in test_trees:
    tree_features, tree_labels = get_data([t for t in tree if type(t["id"])==int])
    test_features += tree_features
    test_labels += tree_labels

vectorizer = DictVectorizer()
vec = vectorizer.fit(train_features)

train_features_vectorized = vec.transform(train_features)
test_features_vectorized = vec.transform(test_features)

clf = MultinomialNB()
clf.fit(train_features_vectorized, train_labels)

predicted = clf.predict(test_features_vectorized)
print(classification_report(test_labels, predicted))
