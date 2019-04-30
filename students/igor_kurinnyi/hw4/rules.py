import re
from tqdm import tqdm
from itertools import zip_longest

import spacy
import peewee
from models import Author, WikiText, FactSource, Book


nlp = spacy.load('en_core_web_lg')


def clean_text(text):
    text = re.sub(r'=+.*?=+', '', text)
    text = re.sub(r'[\s\t\n]+', ' ', text)
    return text


def filter_ents(doc, etype):
    return (ent.text for ent in doc.ents if ent.label_ == etype)


def extract_from_sent(doc):
    titles = list(filter_ents(doc, 'WORK_OF_ART'))
    dates = list(filter_ents(doc, 'DATE'))
    titles = [dict(title=t, date=d) for t, d in zip_longest(titles, dates) if t]
    titles = clean_titles(titles)
    return [dict(x) for x in set(tuple(x.items()) for x in titles)]


def clean_titles(titles):
    filter_pattern = re.compile(r'(?ix)(Award|Prize|Ph\.?D|Pulitzer|Best Novel)')

    sub_patterns = [
        r'of( the| a)?$',
        r'ISBN [\d-]+',
        r'\),? \d{4}',
        r',?Vol\.?( \d+)?',
        r'\(\d{4},?\s?[a-zA-Z]*\)',
        r'"|[,-:]$',
        r'\($'
    ]

    result = list()
    for x in titles:
        title = x['title']
        if not re.search(filter_pattern, title):
            for p in sub_patterns:
                title = re.sub(p, '', title).strip()

            result.append(dict(title=title, date=x['date']))
    return result


def extract_from_text(doc):
    result = list()
    for sent in doc.sents:
        result.extend(extract_from_sent(sent))
    return result


def extract_year_from_date(date):
    try:
        match = re.search(r'\d{4}', date)
        year = int(match.group(0)) if match else None
    except TypeError:
        return None
    else:
        return year


def books_to_db(author, source, books):
    for book in books:
        try:
            Book.create(author=author, fact_source=source, title=book['title'],
                        year=extract_year_from_date(book['date']))
        except peewee.IntegrityError:
            print(book)


def extract_books(source_name):
    try:
        source = FactSource.get(FactSource.stype == source_name)
    except:
        source = FactSource.create(stype=source_name)

    for author in tqdm(Author.select(), desc='Extracting Books from wiki pages'):
        wiki = author.wikipage.get()
        text = clean_text(wiki.text)
        books = extract_from_text(nlp(text))
        books_to_db(author=author, source=source, books=books)
