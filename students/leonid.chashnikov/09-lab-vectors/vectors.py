import os
import re
from array import array
import langdetect
import numpy as np
from langdetect.lang_detect_exception import LangDetectException

from scipy.spatial import distance
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

en_vectors_file = './data/numberbatch-en-17.06.txt'
uk_vectors_file = './data/news.lowercased.tokenized.word2vec.300d'

uk_vectors = KeyedVectors.load_word2vec_format(uk_vectors_file, binary=False)


def _read_files(path='./data/1551/'):
    files = os.listdir(path)
    results = []
    for filename in files:
        with open(path + filename, "r") as input_file:
            bulk = input_file.read()
            blocks = re.compile(r"\n{2,}").split(bulk)[:-1]
            results = results + blocks
    return results


def _get_vectors(sents) -> dict:
    vectors = dict()
    for sent in sents:
        sent_vects = []
        for w in sent.split(' '):
            if w and w != ' ':
                try:
                    word_vec = uk_vectors.get_vector(w.strip())
                    sent_vects.append(word_vec)
                except Exception as e:
                    pass
        sent_vect_len = len(sent_vects)
        if len(sent_vects) > 0:
            sent_vect = sum(sent_vects)
            sent_vect = [point / sent_vect_len for point in sent_vect]
            vectors[sent] = sent_vect
    return vectors


def _filter_sents(sents):
    result = []
    for sent in sents:
        try:
            lang = langdetect.detect(sent)
            if lang == 'uk':
                result.append(sent)
        except LangDetectException as e:
            pass
    return result


def _find_similar(ideal_vector, vectors, top=5):
    distanced_vectors = dict()
    ideal_vector_r = np.array(ideal_vector).reshape(1, -1)
    for sent, vector in vectors.items():
        vector_r = np.array(vector).reshape(1, -1)
        distanced_vectors[sent] = cosine_similarity(ideal_vector_r, vector_r)
    sort_vect = sorted(distanced_vectors.items(), key=lambda x: x[1])
    sort_vect.reverse()
    return sort_vect[1:top]


if __name__ == "__main__":
    sents = _read_files()
    print('All: {}'.format(len(sents)))
    sents = _filter_sents(sents)
    print('Ukr: {}'.format(len(sents)))
    vectors = _get_vectors(sents)
    sentence, vector = list(vectors.items())[19]
    print(sentence)
    print()
    print()
    print()

    similar_list = _find_similar(vector, vectors)
    print(similar_list)


    # print(len(sents))
    # print(len(vectors))

    # print(uk_vectors.similar_by_vector(vectors[0]))
    # print(uk_vectors.most_similar_to_given(vectors[0], vectors))

    # king_v = uk_vectors.get_vector('король')
    # print('Король {}'.format(uk_vectors.most_similar('король', topn=5)))
    # man_v = uk_vectors.get_vector('чоловік')
    # print('Чоловік {}'.format(uk_vectors.most_similar('чоловік', topn=5)))
    # woman_v = uk_vectors.get_vector('жінка')
    # print('Жінка {}'.format(uk_vectors.most_similar('жінка', topn=5)))
    # queen_v = (king_v - man_v) + woman_v
    # print(uk_vectors.similar_by_vector(queen_v))


# knn classifier, train on list of vectors, input - vector of sentence