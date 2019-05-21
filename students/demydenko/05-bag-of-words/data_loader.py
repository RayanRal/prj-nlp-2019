import requests
from bs4 import BeautifulSoup
import time
import json
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
from langdetect import detect
import re


# завантажує список посилань на ноутбуки та зберігає в файл note_link_.txt
def load_notebooks(load_from_file=False):
    note_links = []
    if load_from_file:
        # file = open('note_link_.txt', 'r')
        with open('note_link_.txt') as f:
            notebooks = f.readlines()

        notebooks = [x.strip() for x in notebooks]
        return notebooks
    else:
        file = open('note_link_.txt', 'w')
        for j in range(1, 100):
            url = 'https://rozetka.com.ua/notebooks/c80004/filter/'
            page = 'page={}/'.format(j)
            url = url + page
            print(j, '---', url)

            r = requests.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')

            product = soup.find_all('div', class_='g-i-tile-i-title')
            for div in product:
                note_link = div.find('a', href=True)['href']
                note_links.append(note_link)
                print(note_link)
                file.write(note_link + '\n')

            time.sleep(1)

        file.close()
        return note_links


# завантажує коментарі від одного ноутбука, і зберігає в json файл
def parse_product_comments(link, load_from_file=False):
    reviews = []
    filename = 'data/' + link.replace('/', '_') + '.json'
    if load_from_file:
        with open(filename, 'r') as fp:
            reviews = json.load(fp)
        return reviews
    else:
        file = open(filename, 'w')

        # link = link + '/comments/page={}/'.format(1)
        # r = requests.get(link)
        # soup = BeautifulSoup(r.text, 'html.parser')
        full_link = link + 'comments/page={}/'.format(1)
        r = requests.get(full_link)
        soup = BeautifulSoup(r.text, 'html.parser')
        page_count = 1
        try:
            page_count = int(soup.find_all('span', class_='paginator-catalog-l-i-active')[-1].text)
        except:
            pass
        # page_count = 3
        for j in range(1, page_count+1):
            # link = link + '#tab=comments'
            if j > 1:
                full_link = link + 'comments/page={}/'.format(j)
                r = requests.get(full_link)
                soup = BeautifulSoup(r.text, 'html.parser')

            # page_count = soup.find_all('span', class_='paginator-catalog-l-i-active')[-1].text

            comments = soup.find_all('div', class_="pp-review-inner")
            # comments > article:nth-child(1) > div > div.pp-review-inner
            for div in comments:
                # print(div)
                review = {'rating': None, 'text': '', 'text_plus': '', 'text_minus': ''}
                try:
                    review['rating'] = div.find('span', class_='sprite g-rating-stars-i')['content']
                except:
                    pass

                review['text'] = div.find_all('div', class_='pp-review-text-i')[0].text.strip()
                try:
                    review['text_plus'] = div.find_all('div', class_='pp-review-text-i')[1].text.strip()
                except:
                    pass
                try:
                    review['text_minus'] = div.find_all('div', class_='pp-review-text-i')[2].text.strip()
                except:
                    pass

                reviews.append(review)
                print('j-' , j)
            # time.sleep(1)
        json.dump(reviews, file)
        return reviews


# завантаження всіх даних, якщо load_from_file=True - завантажуються із підготовлених файлів
def load_data(load_from_file=False):
    if load_from_file:
        with open('reviews_uk.pickle', 'rb') as handle:
            reviews_uk = pickle.load(handle)
        return reviews_uk
    else:
        # product_links = load_notebooks(load_from_file=True)
        product_links = load_notebooks(load_from_file=False)
        df = pd.DataFrame(product_links)
        df = df.drop_duplicates()
        reviews = []
        i = 0
        #
        for index, row in df.iterrows():
            i = i + 1
            print(i)
            reviews = reviews + parse_product_comments(row[0], load_from_file=False)

        # for product_link in product_links:
        #     i = i + 1
        #     print(i)
        #     reviews = reviews + parse_product_comments(product_link, load_from_file=True)
        #

        # reviews_uk = []
        # for review in reviews:
        #     i = i + 1
        #     print(i)
        #     if review['rating'] and int(review['rating']) in range(0,6) and review['text'] != '':
        #         lang = detect(review['text'])
        #         if lang == 'uk':
        #             reviews_uk.append(review)
        #             pass
        #
        with open('reviews.pickle', 'wb') as handle:
            pickle.dump(reviews, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('reviews.pickle', 'rb') as handle:
            reviews = pickle.load(handle)

        reviews_uk = []
        i = 0
        for review in reviews:
            i = i + 1
            print(i)
            if review['rating'] and int(review['rating']) in range(0,6) and review['text'] != '':
                lang = ''
                try:
                    lang = detect(review['text'])
                except:
                    pass
                if lang == 'uk':
                    reviews_uk.append(review)
                    pass

        with open('reviews_uk.pickle', 'wb') as handle:
            pickle.dump(reviews_uk, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return reviews_uk


# завантаження даних, попередня обробка і розбивка та train test
def load_and_cook_data(load_from_file=False):
    reviews_uk = load_data(load_from_file=load_from_file)
    df_ = pd.DataFrame(reviews_uk)
    df_ = df_.drop_duplicates()

    #  питання ігноруємо!
    df_without_q = df_[df_['text'].str.find("?") == -1]
    df_ = df_without_q
    # only 498 left

    # фічі для бейслайну
    positive_feature_list = ["гарн", "легк", "шустр", "хорош", "супер", "добротн", "потужни", "чудо", "кльов",
                             "швидк", "зручн", "симпат", "непоган"]
    for f in positive_feature_list:
        df_['fp_' + f] = df_['text'].str.contains(f, flags=re.IGNORECASE, regex=True)
    df_['fp_plus_len'] = len(df_['text_plus'])
    negative_feature_list = ["Жах", "Тормоз", "поган", "висит", "Не раджу", "брак", "обереж"]
    for f in negative_feature_list:
        df_['fn_' + f] = df_['text'].str.contains(f, flags=re.IGNORECASE, regex=True)
    df_['fn_minus_len'] = len(df_['text_minus'])
    df_["rating"] = pd.to_numeric(df_["rating"])
    df_["fp_plus_len"] = pd.to_numeric(df_["fp_plus_len"])
    df_["fn_minus_len"] = pd.to_numeric(df_["fn_minus_len"])

    df_y = df_['rating']
    df_y = pd.DataFrame(df_y)

    # спрощуємо систему оцінок
    df_y[df_y['rating'] < 3] = -1
    df_y[df_y['rating'] > 3] = 1
    df_y[df_y['rating'] == 3] = -1

    df_x = df_.drop('rating', 1)
    # df_x = df_x.drop('text', 1)
    # df_x = df_x.drop('text_minus', 1)
    # df_x = df_x.drop('text_plus', 1)

    x_dict = df_x.to_dict('records')
    y_dict = list(df_y['rating'])

    x_train, x_test, y_train, y_test = train_test_split(x_dict, y_dict, test_size=0.3, random_state=0)

    return x_train, x_test, y_train, y_test
