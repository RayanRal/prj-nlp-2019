import re
from datetime import datetime
from itertools import groupby

import wikipedia
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON

from models import Author, Book, FactSource, WikiText

sparql = SPARQLWrapper("http://dbpedia.org/sparql")

query = '''
SELECT DISTINCT ?aname, ?birth, ?alink, ?wiki_id, ?book_title, ?pdate
WHERE {
        ?author a dbo:Writer ;
            foaf:isPrimaryTopicOf ?alink ;
            dbo:wikiPageID ?wiki_id ;
            dbo:birthName ?aname ;
            dbo:birthDate ?birth .
        FILTER(regex(?birth, '[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}'))

        ?book rdf:type dbo:Book ;
            dbo:author ?author ;
            rdfs:label ?book_title ;
            dbp:language ?lang .
        FILTER(lang(?book_title) = 'en')
        FILTER(regex(?lang, 'English'))
        
        ?book dbp:releaseDate|dbo:publicationDate|dbp:pubDate|dbp:published ?pdate
        FILTER(datatype(?pdate) = xsd:date || datatype(?pdate) = xsd:integer || regex(?pdate, '^[a-zA-Z]+ [0-9]{4}$'))
}
ORDER BY ?aname
LIMIT 1000
'''


NROWS = 0


def main():
    data = query_dbpedia(query)
    global NROWS
    NROWS = len(data['results']['bindings'])

    data = group_data(data)
    populate_db(data)


def query_dbpedia(query):
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


def group_data(data):
    keyfunc = lambda x: (x['aname']['value'], x['birth']['value'])
    return groupby(data['results']['bindings'], key=keyfunc)


def populate_db(data):
    dbpedia = FactSource.get(FactSource.stype == 'dbpedia')

    for (name, birth), rows in tqdm(data, total=NROWS):
        author = Author.create(name=name, birthday=to_datetime(birth))

        for row in rows:
            create_book(row, author, dbpedia)

        create_wikitext(row, author)


def create_book(d, author, dbpedia):
    title = clean_title(d['book_title']['value'])
    year = extract_year(d['pdate']['value'])
    Book.create(author=author, title=title, year=year, fact_source=dbpedia)


def clean_title(title):
    pattern = r'''(?ix)"|
                       \(book(\s\d+)?\)|
                       \((short story )?collection\)|
                       \(anthology\)|
                       \(([a-zA-Z]+\s)*novel(a|ette)?(\sseries)?\)'''
    pattern = re.compile(pattern, re.VERBOSE)
    title = re.sub(pattern, '', title)
    return title


def extract_year(year):
    try:
        return int(re.search(r'\d{4}', year).group(0))
    except AttributeError:
        return year  # year is already int


def create_wikitext(d, person):
    try:
        link = d['alink']['value']
        wiki_id = d['wiki_id']['value']
        text = wikipedia.page(pageid=wiki_id).content
        WikiText.create(person=person, link=link, wiki_id=wiki_id, text=text)
    except wikipedia.exceptions.PageError:
        print(f'Cannot extract wiki page for {d["alink"]["value"]}')


def to_datetime(d):
    try:
        d = d.replace('-0', '-01')
        return datetime.strptime(d, '%Y-%m-%d')
    except ValueError:
        print(f'Cannot convert to datetime {d}')


def delete_books_from_source(source_name):
    try:
        source = FactSource.get(FactSource.stype == source_name)
        for b in tqdm(source.facts, desc='Clearing DB from old results'):
            b.delete_instance()
    except Exception as e:
        print(e)


if __name__ == '__main__':
    # main()
    pass



