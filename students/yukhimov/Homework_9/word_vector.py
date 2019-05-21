from gensim.models import KeyedVectors
import json
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy
import re

with open('request_messages.txt') as json_file:
    data = json.load(json_file)

messages = data['messages']
random.shuffle(messages)

train_set, test_set = messages[:43205], messages[43205:]

train_messages = [item['message'] for item in train_set]
train_labels = [item['category'] for item in train_set]
test_messages = [item['message'] for item in test_set]
test_labels = [item['category'] for item in test_set]

uk_vectors = KeyedVectors.load_word2vec_format('news.lowercased.tokenized.word2vec.300d', binary=False)


def vectorize_message(message):
    message_vector = []
    words = message.split()
    for word in words:
        word = re.sub(r'[^\w]', '', word)
        try:
            message_vector.append(uk_vectors.get_vector(word).tolist())
        except KeyError:
            pass
    if not message_vector:
        return numpy.array([0] * 300)
    message_vector = numpy.array(message_vector)
    return numpy.mean(message_vector, axis=0)


train_features = numpy.array([vectorize_message(message) for message in train_messages])
test_features = numpy.array([vectorize_message(message) for message in test_messages])

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(train_features, train_labels)
y_pred = knn.predict(test_features)

print("Accuracy:", metrics.accuracy_score(test_labels, y_pred))
print(metrics.classification_report(test_labels, y_pred))


'''
Vectors mean average


Accuracy: 0.3337653920933247

           precision    recall  f1-score   support

micro avg       0.33      0.33      0.33     18516
macro avg       0.26      0.18      0.19     18516
weighted avg    0.35      0.33      0.32     18516


'''