import conllu

files = ['./data/uk_iu-ud-train.conllu', './data/uk_iu-ud-test.conllu', './data/uk_iu-ud-dev.conllu',]


def read_file(filename):
    with open(filename, 'r') as f:
        data = f.read()
        parsed = conllu.parse(data)
    return parsed


def debug(tree):
    # debug mode
    for node in tree:
        head = node["head"]
        print("{} <-- {}".format(node["form"],
                                 tree[head - 1]["form"]
                                 if head > 0 else "root"))