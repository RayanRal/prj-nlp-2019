import json
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics

with open('run_on_set.txt') as json_file:
    data = json.load(json_file)

with open('run-on-test.json') as json_file:
    test_data = json.load(json_file)

two_grams = []
with open('w2_.txt', 'r', encoding='latin-1') as input_file:
    for line in input_file:
        two_grams.append(line.split()[1:])

three_grams = []
with open('w3_.txt', 'r', encoding='latin-1') as input_file:
    for line in input_file:
        three_grams.append(line.split()[1:])


train_set, test_set = data['run_on_set'][:1200], data['run_on_set'][1200:]


def extract(dataset):
    token_set = []
    label_set = []
    for sentence in dataset:
        for word in sentence:
            token_set.append(word[0])
            label_set.append(word[1])
    return token_set, label_set


train_tokens, train_labels = extract(train_set)
test_tokens, test_labels = extract(test_set)
evaluation_tokens, evaluation_labels = extract(test_data)


def feature_set(sentence_set):
    features = []
    for index in range(len(sentence_set)):
        token_dict = {}
        try:
            token_dict['next_word_capitalized'] = sentence_set[index+1].istitle()
            two_gram = [sentence_set[index], sentence_set[index + 1]]
            if two_gram in two_grams:
                token_dict['two_gram'] = True
        except IndexError:
            pass
        if sentence_set[index] in ('for', 'and', 'nor', 'but', 'or', 'yet', 'so'):
            token_dict['conjunction'] = True
        try:
            three_gram = [sentence_set[index], sentence_set[index+1], sentence_set[index+2]]
            if three_gram in three_grams:
                token_dict['three_gram'] = True
        except IndexError:
            pass
        features.append(token_dict)
    return features


vect = DictVectorizer()
X_train = vect.fit_transform(feature_set(train_tokens))
Y_test = vect.transform(feature_set(evaluation_tokens))

logreg = LogisticRegression()

logreg.fit(X_train, train_labels)
y_pred = logreg.predict(Y_test)

print(metrics.classification_report(evaluation_labels, y_pred))
