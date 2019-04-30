"""
This module filters scrapped csv file (leave Ukrainian records only, remove some strange records)
and divides records to two set - learn and test.
"""

from collections import namedtuple
import csv
from typing import List
from random import shuffle

from langid import langid

Reviews = namedtuple('Reviews', 'goods_code stars review author permalink')


def load_reviews(filename: str) -> List[Reviews]:
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        res = []
        for row in reader:
            res.append(Reviews(**row))
    return res


def save_reviews(filename: str, data: List[Reviews]):
    with open(filename, 'wt', encoding='utf-8', newline='') as f:
        fieldnames = Reviews._fields
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([r._asdict() for r in data])


def prepare_data():
    reviews: List[Reviews] = load_reviews('rozetka.csv')
    print(f'Total records: {len(reviews)}')
    filtered: List[Reviews] = []
    for r in reviews:
        if len(r.review) < 32:
            continue  # too short
        lang = langid.classify(r.review)
        if lang[0] != 'uk':
            continue  # not ukrainian
        filtered.append(r)
    print(f'Filtered records: {len(filtered)}')
    shuffle(filtered)
    threshold = int(len(filtered)*0.7)
    learn = filtered[:threshold]
    print(f"Learn records set: {len(learn)}")
    test = filtered[threshold:]
    print(f"Test records set: {len(test)}")
    save_reviews('rozetka_learn.csv', learn)
    save_reviews('rozetka_test.csv', test)


if __name__ == "__main__":
    prepare_data()
