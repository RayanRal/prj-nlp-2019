from os import listdir
from gensim.models import KeyedVectors
import pandas as pd
import multiprocessing as mp
from langdetect import detect
from tokenize_uk.tokenize_uk import tokenize_words


DATA_DIR = './data/'
COMPLAINTS_DIR = DATA_DIR + 'complaints/'

uk_vectors_file = DATA_DIR + 'ubercorpus.lowercased.tokenized.word2vec.300d'
uk_vectors_lemma_file = DATA_DIR + 'ubercorpus.lowercased.lemmatized.word2vec.300d'
uk_vectors = KeyedVectors.load_word2vec_format(uk_vectors_file, binary=False)
uk_vectors_lemma = KeyedVectors.load_word2vec_format(uk_vectors_file, binary=False)

f = open('./data/input.tsv', 'a+')

empty_vector = [0]*300

def get_vector(tokens, lemma=False):
    if lemma:
        vectors = [uk_vectors_lemma.get_vector(token) for token in tokens if token and token in uk_vectors_lemma]
    else:
        vectors = [uk_vectors.get_vector(token) for token in tokens if token and token in uk_vectors]
    if len(vectors):
        return list(sum(vectors)/len(vectors))
    return empty_vector

def parse_appeal(i, doc):
    message = doc.strip().split('\n')
    doc_id = message.pop(0)
    if doc_id.isdigit():
        doc_id = int(doc_id)
        message = '\n'.join(message)
        if message and detect(message) == 'uk':
            tokens = [token.lower() for token in tokenize_words(message)]
            vector = get_vector(tokens)
            return doc_id, i, tokens, vector
    return None, None, None, None

def addToDataFrame(data):
    doc_id, category, tokens, vector = data
    if doc_id:
        f.write(f'{doc_id}\t{category}\t{tokens}\t{vector}\n')

def process():
    pool = mp.Pool()

    for i, filename in enumerate(listdir(COMPLAINTS_DIR)):
        with open(COMPLAINTS_DIR + filename, 'r') as data:
            docs = data.read().split('\n\n\n')
            for doc in docs:
                pool.apply_async(parse_appeal, args=[i, doc], callback=addToDataFrame)

    pool.close()
    pool.join()

if __name__ == '__main__':
    process()
