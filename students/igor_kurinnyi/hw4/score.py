import random
from tqdm import tqdm
from typing import List, Tuple

import config
from models import Author, WikiText, FactSource, Book

random.seed(config.SEED)


def generate_train_ids():
    ids = [x.id for x in Author.select()]
    random.shuffle(ids)
    return set(ids[: round(len(ids) * config.TRAIN_SIZE)])


def data_set(stype='all'):
    train_ids = generate_train_ids()
    if stype == 'train':
        authors = Author.select().where(Author.id.in_(train_ids))
    elif stype == 'test':
        authors = Author.select().where(Author.id.not_in(train_ids))
    elif stype == 'all':
        authors = Author.select()

    for author in authors:
        yield author


def train_set():
    return data_set(stype='train')


def test_set():
    return data_set(stype='test')


def train_set_scores(source_name):
    return data_set_score(source_name, stype='train')


def test_set_scores(source_name):
    return data_set_score(source_name, stype='test')


def data_set_score(source_name, stype='all'):
    if stype == 'train':
        data = train_set()
    elif stype == 'test':
        data = test_set()
    elif stype == 'all':
        data = data_set()

    dbpedia = FactSource.get(FactSource.stype == 'dbpedia')
    source = FactSource.get(FactSource.stype == source_name)

    scores = list()
    for author in tqdm(data, desc='Computing score for authors'):
        db_books = [b for b in author.books if b.fact_source == dbpedia]
        wiki_books = [b for b in author.books if b.fact_source == source]
        scores.append(score(db_books, wiki_books))

    return mean_score(scores)


def score(books_1, books_2):
    tp = true_positive_value(intersection(books_1, books_2))
    fp = len(difference(books_2, books_1))
    fn = len(difference(books_1, books_2))

    precision = tp / (tp + fp + 0.0001)
    recall = tp / (tp + fn + 0.0001)
    f1 = 2 * (precision * recall) / (precision + recall + 0.001)
    return dict(f1=f1, precision=precision, recall=recall, count=len(books_1))


def mean_score(scores):
    score = dict(f1=0, precision=0, recall=0)
    for s in scores:
        score['f1'] += (s['f1'] * s['count'])
        score['precision'] += (s['precision'] * s['count'])
        score['recall'] += (s['recall'] * s['count'])

    sum_count = sum(x['count'] for x in scores)
    score['f1'] /= sum_count
    score['precision'] /= sum_count
    score['recall'] /= sum_count
    return score


def intersection(books_1: List[Book], books_2: List[Book]) -> List[Tuple[Book, float]]:
    result = list()
    for b1 in books_1:
        for b2 in books_2:
            sim = b1.similarity(b2)
            if sim >= config.SIMILARITY_THRESHOLD:
                result.append((b1, sim))
    return result


def difference(books_1: List[Book], books_2: List[Book]) -> List[Book]:
    result = list()
    for b1 in books_1:
        in_ = False
        for b2 in books_2:
            sim = b1.similarity(b2)
            if sim >= config.SIMILARITY_THRESHOLD:
                in_ = True
                break
        if not in_:
            result.append(b1)
    return result


def true_positive_value(books: List[Tuple[Book, float]]) -> float:
    return sum(x[1] for x in books)


if __name__ == '__main__':
    scores = train_set_scores('spacy_sm')
    print(scores)

