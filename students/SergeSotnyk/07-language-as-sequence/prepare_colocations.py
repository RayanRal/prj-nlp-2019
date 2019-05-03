from collections import defaultdict
from typing import Sequence, List, Dict, Optional, Tuple

from make_train_dev_ny2000 import original_sentences
from jsonlines import jsonlines


def collect_bigrams_lowercase(all_sents: Sequence[List[str]],
                              min_num: int = 0) -> List[Tuple[str, int]]:
    acc = defaultdict(int)
    total = 0
    for sent in all_sents:
        prev: Optional[str] = None
        for t in sent:
            if prev:
                bigram = prev.lower() + '_' + t.lower()
                acc[bigram] += 1
                total += 1
            prev = t
    res = [(k, v) for k, v in acc.items() if v >= min_num]
    return res


def store_bigrams(bi_name: str, bigrams: List[Tuple[str, int]]):
    bi_list = sorted(bigrams, reverse=True, key=lambda x: x[1])
    with jsonlines.open(bi_name, mode='w') as writer:
        for i in bi_list:
            writer.write(i)


def main():
    filename = 'data/nyt2000-sents.jsonl'
    all_sents = original_sentences(filename)
    bi_name = 'data/bigrams.jsonl'
    bigrams = collect_bigrams_lowercase(all_sents, min_num=2)
    print(f"Collected {len(bigrams)}")
    store_bigrams(bi_name, bigrams)
    print("Stored")


if __name__ == "__main__":
    main()
