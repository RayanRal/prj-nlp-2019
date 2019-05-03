import re
from bs4 import BeautifulSoup
import requests
import spacy


nlp = spacy.load('en_core_web_lg')
NAMED_ENTITY_TYPES = ['WORK_OF_ART', 'DATE']

WIKI_URL = "https://en.wikipedia.org/wiki/J._R._R._Tolkien"

DBPEDIA_REQ = "https://dbpedia.org/sparql?default-graph-uri=http%3A%2F%2Fdbpedia.org&query=PREFIX+dbo%3A+%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2F%3E%0D%0APREFIX+res%3A++%3Chttp%3A%2F%2Fdbpedia.org%2Fresource%2F%3E%0D%0APREFIX+rdfs%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F2000%2F01%2Frdf-schema%23%3E%0D%0APREFIX+dbp%3A+%3Chttp%3A%2F%2Fdbpedia.org%2Fproperty%2F%3E%0D%0ASELECT+DISTINCT+%3Fbook+%3Fyear%0D%0AWHERE+%7B%0D%0A%09%3Furi+dbo%3Aauthor+res%3AJ._R._R._Tolkien+.%0D%0A++++++++OPTIONAL+%7B%0D%0A++++++++++++++++++%3Furi+dbp%3AreleaseDate+%3Fyear+.%0D%0A++++++++++++++++++%7D+.%0D%0A%09%3Furi+rdfs%3Alabel+%3Fbook+.%0D%0A++++++++FILTER+%28lang%28%3Fbook%29+%3D+%27en%27%29%0D%0A%7D&format=text%2Fhtml&CXML_redir_for_subjs=121&CXML_redir_for_hrefs=&timeout=30000&debug=on&run=+Run+Query+"

YEAR_REGEX = re.compile(r'[1-3][0-9]{3}')


def _detect_ne(text: str):
    labels = dict()
    doc = nlp(text)
    for s in doc.sents:
        if any([True for e in s.ents if e.label_ == 'WORK_OF_ART']):
            books = [e.text.strip() for e in s.ents if e.label_ == 'WORK_OF_ART']
            if len(books) > 1 or all((len(b.strip()) == 0 for b in books)):
                continue
            years = [e.text.strip() for e in s.ents if e.label_ == 'DATE']
            if years:
                year = next((re.findall(YEAR_REGEX, year) for year in years if re.findall(YEAR_REGEX, year)), None)
            else:
                year = None
            if not labels.get(books[0]) and year:
                labels[books[0]] = year
            else:
                labels[books[0]] = None
    return labels


def _parse_dbpedia(text: str):
    parsed = BeautifulSoup(text, 'lxml')
    entries = parsed.table.find_all('tr')
    result = dict()
    for e in entries:
        td = e.find_all('td')
        if len(td) == 2:
            name = td[0].text.rstrip('@en').strip('"')
            year = td[1].text
            result[name] = year
    return result


def _compare(ground_truth, extracted_data):
    found_book_count = 0
    found_year_count = 0
    for true_book, true_year in ground_truth.items():
        if true_book in extracted_data.keys():
            found_book_count += 1
            if extracted_data[true_book] == true_year or (extracted_data[true_book] and not true_year):
                found_year_count += 1
    return found_book_count, found_year_count


if __name__ == "__main__":
    # get ground truth from dbpedia
    dbpedia_response = requests.get(DBPEDIA_REQ)
    ground_truth = _parse_dbpedia(dbpedia_response.text)

    # get all NERs from wikipedia page
    wiki_response = requests.get(WIKI_URL)
    wiki_page = BeautifulSoup(wiki_response.text, 'html.parser')

    extracted_data = _detect_ne(wiki_page.text)

    # compare dbpedia and wiki
    ground_truth_books_amount = len(ground_truth)
    ground_truth_years_amount = len([v for v in ground_truth.values() if v != ''])

    book_found, year_found = _compare(ground_truth, extracted_data)
    irrelevant_data = len(extracted_data) - book_found

    print('Correct books found: {} out of {} - {}%'
          .format(book_found, ground_truth_books_amount, round((book_found / ground_truth_books_amount) * 100)))
    print('Correct years of books found: {} out of {} - {}%'
          .format(year_found, ground_truth_years_amount, round((year_found / ground_truth_years_amount) * 100)))
    print('Extracted irrelevant data (non-books, titles from other author etc): {} - {}%'
          .format(irrelevant_data, round((irrelevant_data / len(extracted_data)) * 100)))


