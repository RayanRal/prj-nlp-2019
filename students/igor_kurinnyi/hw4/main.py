from score import train_set_scores, test_set_scores
from dbpedia_data import delete_books_from_source
from rules import extract_books

if __name__ == '__main__':
    source_name = 'spacy_clean_lg'

    delete_books_from_source(source_name)
    extract_books(source_name)
    scores = train_set_scores(source_name)
    print(scores)

