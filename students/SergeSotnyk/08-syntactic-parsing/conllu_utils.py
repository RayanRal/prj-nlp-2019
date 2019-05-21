from os.path import join, isfile, isdir
from os import mkdir
from random import Random
from typing import List, Dict

import wget
from conllu import parse, TokenList

_url_prefix = "https://github.com/UniversalDependencies/UD_Ukrainian-IU/blob/master/"
DEV_FILENAME = "uk_iu-ud-dev.conllu"
TEST_FILENAME = "uk_iu-ud-test.conllu"
TRAIN_FILENAME = "uk_iu-ud-train.conllu"

_PATH = "data/"

_rnd = Random(1974)


def load_trees(filename: str) -> List[TokenList]:
    if not isdir(_PATH):
        mkdir(_PATH)

    full_name = join(_PATH, filename)
    if not isfile(full_name):
        wget.download(_url_prefix + filename + '?raw=true', full_name)

    with open(full_name, "r", encoding='utf-8') as f:
        data = f.read()
    result = parse(data)
    return result


def token_to_features(t: str, prefix: str) -> Dict[str, str]:
    casefolded = t.casefold()
    return {
        prefix + ':norm': casefolded,
        prefix + ':title': t.title(),
        prefix + ':[-1:]': casefolded[-1:],
        prefix + ':[-2:]': casefolded[-2:],
        prefix + ':[-3:]': casefolded[-3:],
        prefix + ':[:1]': casefolded[:1],
        prefix + ':[:2]': casefolded[:2],
        prefix + ':[:3]': casefolded[:3],
    }


def relation_to_features(tokens: List[str], child_pos: int, head_pos: int) -> Dict[str, object]:
    def pos_to_str(i: int) -> str:
        return tokens[i - 1] if i > 0 else "root"

    f = {'i': child_pos, 'head': head_pos, 'dist_abs': abs(child_pos - head_pos)}
    f.update(token_to_features(pos_to_str(child_pos), 't'))
    f.update(token_to_features(pos_to_str(head_pos), 'head:'))
    return f


def extract_features_and_y(tree: TokenList,
                           use_all_negative_samples: bool = False):
    features = []
    y = []
    tokens = [t["form"] for t in tree]
    for i, t in enumerate(tree):
        # retrieve positive scenarios
        head = t['head']
        if head is None:  # Some trees are not well formed
            break
        f = relation_to_features(tokens, i + 1, head)
        features.append(f)
        y.append(True)
        # construct negative samples
        if len(tree) < 2:
            break
        if use_all_negative_samples:
            # All another variants except proper are negative
            for neg_head in range(0, len(tree)+1):
                if neg_head==head or neg_head == i+1:
                    break
                f = relation_to_features(tokens, i + 1, neg_head)
                features.append(f)
                y.append(False)
        else:
            # Randomly select one variant as negative
            while True:
                neg_head = _rnd.randint(0, len(tree))
                if neg_head != i + 1 and neg_head != head:
                    break
            f = relation_to_features(tokens, i + 1, neg_head)
            features.append(f)
            y.append(False)
    return features, y


if __name__ == '__main__':
    trees = load_trees(DEV_FILENAME)
    tokens = sum(len(tree) for tree in trees)
    print(f"Dev set contains {tokens} tokens in {len(trees)} sentences")
    trees = load_trees(TEST_FILENAME)
    tokens = sum(len(tree) for tree in trees)
    print(f"Test set contains {tokens} tokens in {len(trees)} sentences")
    trees = load_trees(TRAIN_FILENAME)
    tokens = sum(len(tree) for tree in trees)
    print(f"Train set contains {tokens} tokens in {len(trees)} sentences")
    tree = trees[0]
    print(tree[0])
    for node in tree:
        head = node["head"]
        print("{} <-- {}".format(node["form"],
                                 tree[head - 1]["form"]
                                 if head > 0 else "root"))
    print("Features:")
    xs, ys = extract_features_and_y(tree)
    for x, y in zip(xs, ys):
        print(x, y)
