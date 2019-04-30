import csv
from langdetect import detect
from pandas import pandas as pd

STOPWORDS = {'нет', 'нету', 'цена'}

def filter_data(data):
    pros, cons = data
    if pros in STOPWORDS or cons in STOPWORDS:
        return False
    return True

def prepare_data(dataframe):
    df = dataframe.copy()
    df['review'] = df['review'].str.lower()
    df['pros'] = df['pros'].str.lower().str.replace(r'[()+\-*=\/\\]+', ' ').str.replace(r'\d+', ' ').str.replace(r'\s+', ' ').str.strip()
    df['cons'] = df['cons'].str.lower().str.replace(r'[()+\-*=\/\\]+', ' ').str.replace(r'\d+', ' ').str.replace(r'\s+', ' ').str.strip()
    df = df[df['review'].str.contains('[А-ЯҐЄІЇа-яґєії]')]
    df = df[df[['pros', 'cons']].apply(filter_data, axis=1)]
    df['review'] = df['review'].str.replace(r'[()+\-*=\/\\]+', ' ').str.replace(r'\d+', ' ').str.replace(r'\s+', ' ').str.strip()
    df = df[df['review'].apply(lambda x: detect(x) == 'uk')]
    return df

if __name__ == '__main__':
    df = pd.read_csv('../01-crawling/data/output/data2.tsv', sep='\t', index_col=0, names=['id', 'rating', 'item_bought', 'review', 'pros', 'cons', 'upvotes', 'downvotes'])
    result = prepare_data(df)
    result.to_csv('data/input/comments2.tsv', sep='\t', quoting=csv.QUOTE_NONE)
