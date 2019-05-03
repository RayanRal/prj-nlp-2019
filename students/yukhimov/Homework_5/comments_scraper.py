from bs4 import BeautifulSoup
import requests
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import json

data = {'results': []}
HOST_NAME = 'https://rozetka.com.ua/notebooks/c80004/filter/'

for page_number in range(1, 101):
    url = HOST_NAME + 'page={}'.format(page_number)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    comments_links = soup.find_all('a', {'class': 'novisited g-rating-reviews-link'})
    for link in comments_links:
        if link.has_attr('data-count'):
            comments_url = link['href']
            response = requests.get(comments_url)
            soup = BeautifulSoup(response.text, "html.parser")
            comments = soup.find_all('article', {'class': 'pp-review-i'})
            for comment in comments:
                text = comment.find('div', {'class': 'pp-review-text'}).text.replace('\n', '')
                lang = False
                try:
                    if detect(text) == 'uk':
                        lang = True
                except LangDetectException:
                    pass
                if lang:
                    try:
                        stars = comment.find('span', {'class': 'sprite g-rating-stars-i'})['content']
                    except TypeError:
                        stars = None
                    data['results'].append({'stars': stars, 'comment': text})

with open('comments.txt', 'w') as outfile:
    json.dump(data, outfile, ensure_ascii=False, indent=4)
