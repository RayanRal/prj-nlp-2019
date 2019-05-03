from bs4 import BeautifulSoup
import requests
import re
import spacy

ACTOR_NAME = 'Jean Marais'
WIKI_HOST = 'https://en.wikipedia.org/wiki/'


def scrape_article(actor_name):
    url = WIKI_HOST + '_'.join(ACTOR_NAME.split(' '))
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.text.strip()


def cut_table(raw_text):
    raw_text_list = raw_text.split(' ')
    for number, item in enumerate(raw_text_list):
        if 'Filmography[edit]' in item:
            raw_text_list = raw_text_list[:number]
    return ' '.join(raw_text_list)


def find_regex(raw_text):
    movies = []
    sentences = raw_text.split('.')
    for sentence in sentences:
        sentence_movies = re.findall(r'in ([^(\d{4})]+)', sentence)
        movies += sentence_movies
    normalized_movies = []
    for movie in movies:
        movie_title = movie.strip().split(' ')
        if len(movie_title) > 1 or movie_title[0].isupper():
            normalized_movies.append(movie.strip())
    return normalized_movies


def find_entities(raw_text):
    art_entities = []
    nlp = spacy.load('en_core_web_md')
    doc = nlp(raw_text)
    for ent in doc.ents:
        if ent.label_ == 'WORK_OF_ART':
            art_entities.append(ent.text)
    return art_entities


def analyze_results(raw_text, dbpedia_results_filename, regex=True, entities=True):
    evaluation_set = [line.rstrip('\n') for line in open(dbpedia_results_filename)]
    movies_regex = movies_entities = []
    if regex:
        movies_regex = find_regex(raw_text)
    if entities:
        movies_entities = find_entities(raw_text)
    movies_titles = movies_regex + movies_entities
    movies_titles = list(set(movies_titles))
    matches = len(set(movies_titles).intersection(set(evaluation_set)))
    precision = round(matches / len(movies_titles), 2)
    recall = round(matches / len(evaluation_set), 2)
    f1 = round(2 * precision * recall / (precision + recall), 2)
    return f1


if __name__ == "__main__":
    raw_text = cut_table(scrape_article(ACTOR_NAME))
    analyze_results(raw_text, 'movie_titles (dbpedia)')

# regex - 0.17, entities - 0.19, regex and entities - 0.28
