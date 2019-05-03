from typing import Sequence, Generator, List, Tuple, Set
import random

from jsonlines import jsonlines
from tqdm.auto import tqdm

random.seed = 1974

_glue_prob = 155. / 355.
_lowercase_prob = 80. / 155.


def original_sentences(filename: str) -> Sequence[List[str]]:
    stored_cache: Set = set()
    with jsonlines.open(filename) as reader:
        for sentence in tqdm(reader, total=2230373):
            key = '-:-'.join(sentence)
            if key in stored_cache:
                continue
            stored_cache.add(key)
            yield sentence


def add_flags(sent: List[str]) -> List[List[object]]:
    res = [[t, False] for t in sent]
    return res


def make_rot_sents(sentences: Sequence[List[str]]) -> Sequence[List[List[object]]]:
    buffer = []
    for s in sentences:
        sf = add_flags(s)
        if len(buffer) > 0:
            if _lowercase_prob < random.random():
                sf[0][0] = sf[0][0].lower()
        buffer += sf
        if _glue_prob < random.random():
            buffer = buffer[:-1]  # skip dot
            buffer[-1][1] = True
        else:
            yield buffer
            buffer = []
    if len(buffer) > 0:
        yield buffer


def div_raw(rot_seq: Sequence[List[List[object]]], prob_train: float,
            train_fn: str, dev_fn: str):
    with jsonlines.open(train_fn, mode='w') as writer_t:
        with jsonlines.open(dev_fn, mode='w') as writer_d:
            for s in rot_seq:
                if prob_train < random.random():
                    writer_t.write(s)
                else:
                    writer_d.write(s)


def shuffle(filename: str):
    print(f'Start reading file "{filename}"')
    with open(filename, "rt", encoding='utf-8') as f:
        lines = f.readlines()
    print(f'Has read {len(lines)} lines, shuffling in memory')
    random.shuffle(lines)
    print('Start writing back')
    with open(filename, "wt", encoding='utf-8') as f:
        f.writelines(lines)
    print('Done!')


def main():
    filename = 'data/nyt2000-sents.jsonl'
    train_fn = filename.replace('.jsonl', '.train.jsonl')
    dev_fn = filename.replace('.jsonl', '.dev.jsonl')
    all_sents = original_sentences(filename)
    div_raw(make_rot_sents(all_sents), 0.5, train_fn, dev_fn)

    shuffle(train_fn)
    shuffle(dev_fn)

    # print(f'Total proper sentences: {sum(1 for s in all_sents)}')


if __name__ == "__main__":
    main()
