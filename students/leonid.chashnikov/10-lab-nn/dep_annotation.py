from gensim.models import KeyedVectors
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout

from read_files import read_file
from read_files import debug
from read_files import files

import numpy as np

from sklearn.preprocessing import LabelEncoder

uk_vectors_file = './data/news.lowercased.tokenized.word2vec.300d'

uk_vectors = KeyedVectors.load_word2vec_format(uk_vectors_file, binary=False)


def filter_trees(trees):
    result = []
    for tree in trees:
        valid = True
        for node in tree:
            if type(node['head']) != int:
                valid = False
        if valid:
            result.append(tree)
    return result


def _get_embedding(node):
    word = node.get('form').lower()
    try:
        return uk_vectors.get_vector(word)
    except Exception as e:
        return None


def _get_feature_vectors(word, tree):
    # head1, head2, child, DEP
    # get this word, it's head and head's head
    child = word
    child_embedding = _get_embedding(child)
    if child_embedding is None:
        return None, None
    head_index = child.get('head')
    if head_index:
        head = tree[head_index] if head_index < len(tree) else None
        if not head:
            return None, None
        head_embedding = _get_embedding(head)
        if head_embedding is None:
            return None, None
        label = child.get('deprel')
        result_vector = np.concatenate((child_embedding, head_embedding), axis=None)
        return label, result_vector
    else:
        return None, None


def _get_labels_features(filename):
    trees = read_file(filename)
    trees = filter_trees(trees)
    labels, features = [], []
    for tree in trees:
        for word in tree:
            label, feature = _get_feature_vectors(word, tree)
            if label is not None and feature is not None and (':' not in label):
                features.append(feature)
                labels.append(label)

    return labels, features


if __name__ == "__main__":
    train_labels, train_features = _get_labels_features(files[0])
    test_labels, test_features = _get_labels_features(files[1])

    train_features = np.array(train_features)
    test_features = np.array(test_features)

    label_encoder = LabelEncoder()
    train_labels_enc = label_encoder.fit_transform(train_labels)
    test_labels_enc = label_encoder.transform(test_labels)

    print('Num classes {}'.format(label_encoder.classes_.size))
    print('Data preparation finished')

    model = Sequential()

    model.add(Dense(units=500, activation='relu', input_dim=600))
    # model.add(Dense(units=1000, activation='relu'))
    # model.add(Dropout(rate=0.4))
    # HERE BE DROPOUT!
    # model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=label_encoder.classes_.size, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(train_features, train_labels_enc, epochs=10, batch_size=10)

    loss_and_metrics = model.evaluate(test_features, test_labels_enc, batch_size=128)
    print(model.metrics_names)
    print(loss_and_metrics)
