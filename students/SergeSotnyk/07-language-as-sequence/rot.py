import os
from typing import List

import wget
import json


def load_data(filename: str) -> (List[List[str]], List[List[bool]]):
    with open(filename, 'rt', encoding='utf-8') as f:
        js = json.loads(f.read())
    tokens, flags = [], []
    for sent in js:
        ts, fs = [], []
        for t, f in sent:
            ts.append(t)
            fs.append(f)
        tokens.append(ts)
        flags.append(fs)
    return tokens, flags


def show_corpus_statistics(x: List[List[str]], y: List[List[bool]]):
    total_real_sentences, total_missed_ends, total_missed_ends_low_case = 0, 0, 0
    for sent in zip(x, y):
        for i, flag in enumerate(sent[1]):
            if flag:
                total_missed_ends += 1
                total_real_sentences += 1
                if sent[0][i+1].islower():
                    total_missed_ends_low_case += 1
        total_real_sentences += 1
    print(f"Real sentences: {total_real_sentences}")
    print(f"Missed ends: {total_missed_ends}")
    print(f"Started from lowercase: {total_missed_ends_low_case}")

def main():
    filename = 'data/run-on-test.json'
    if not os.path.isfile(filename):
        wget.download(
            url='https://github.com/vseloved/prj-nlp-2019/raw/master/tasks/07-language-as-sequence/run-on-test.json',
            out=filename
        )
    x, y = load_data(filename)
    show_corpus_statistics(x, y)
    print(f"Total sentences: {len(x)}")


if __name__ == "__main__":
    main()
