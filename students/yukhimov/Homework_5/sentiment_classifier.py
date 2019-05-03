import json
from tokenize_uk import tokenize_words
import nltk
import pymorphy2

morph = pymorphy2.MorphAnalyzer()

uk_sentiment_dict = {}

with open('uk_sentiment.txt', "r") as input_file:
    for line in input_file:
        uk_sentiment_dict[line.split()[0]] = line.split()[1]

with open('comments.txt') as json_file:
    data = json.load(json_file)


def normalize_word(word):
    return morph.parse(word)[0].normal_form


def label_data(data, lemmatization=True, check_stars=True):
    for comment in data['results']:
        comment['sentiment'] = 0
        for word in tokenize_words(comment['comment']):
            if lemmatization:
                word = normalize_word(word)
            if word.lower() in uk_sentiment_dict.keys():
                comment['sentiment'] += int(uk_sentiment_dict[word.lower()])
        if comment['sentiment'] > 0:
            comment['sentiment'] = 'positive'
        elif comment['sentiment'] < 0:
            comment['sentiment'] = 'negative'
        else:
            comment['sentiment'] = 'neutral'
        if check_stars:
            if comment['stars'] == '5':
                comment['sentiment'] = 'positive'


def sentiment_features(comment, stars, check_stars=True, lemmatization=True):
    features = {}
    for word in tokenize_words(comment):
        if lemmatization:
            word = normalize_word(word)
        if word.lower() in uk_sentiment_dict.keys():
            features['sentiment'] = word.lower()
    if 'sentiment' not in features.keys():
        features['sentiment'] = None
    if check_stars:
        if stars == '5':
            features['stars'] = '5'
    return features


label_data(data)

featuresets = [(sentiment_features(comment['comment'], comment['stars']), comment['sentiment'])
               for comment in data['results']]
training_set, testing_set = featuresets[:round(len(featuresets) / 2)], featuresets[round(len(featuresets) / 2):]

classifier = nltk.NaiveBayesClassifier.train(training_set)

print(nltk.classify.accuracy(classifier, testing_set))
print(classifier.show_most_informative_features(15))


'''
Without lemmatization and stars in comments:

Accuracy - 0.9133216986620128
Most Informative Features
               sentiment = 'легкий'       positi : neutra =     59.3 : 1.0
               sentiment = 'проблема'     negati : neutra =     40.1 : 1.0
               sentiment = 'хороший'      positi : neutra =     37.1 : 1.0
               sentiment = 'гарантійний'  positi : neutra =     35.6 : 1.0
               sentiment = 'кредит'       negati : neutra =     32.8 : 1.0
               sentiment = 'швидкий'      positi : neutra =     27.7 : 1.0
               sentiment = 'поганий'      negati : neutra =     25.5 : 1.0
               sentiment = 'брак'         negati : positi =     12.0 : 1.0
               sentiment = 'зривати'      negati : neutra =     10.9 : 1.0
               sentiment = 'недолік'      negati : neutra =     10.9 : 1.0
               sentiment = 'справді'      positi : neutra =      9.2 : 1.0
               sentiment = 'дивний'       negati : neutra =      6.6 : 1.0
               sentiment = 'погано'       negati : positi =      4.6 : 1.0
               sentiment = 'приємно'      positi : neutra =      4.0 : 1.0
               sentiment = 'впевнено'     positi : neutra =      4.0 : 1.0


With lemmatization and stars in comments:

Accuracy - 0.9627690517742874
Most Informative Features
               sentiment = 'проблема'     negati : neutra =     98.8 : 1.0
               sentiment = 'швидкий'      positi : neutra =     77.8 : 1.0
               sentiment = 'помилка'      negati : neutra =     52.5 : 1.0
               sentiment = 'хороший'      positi : neutra =     36.8 : 1.0
               sentiment = 'ніякий'       negati : neutra =     31.5 : 1.0
               sentiment = 'покупка'      negati : positi =     26.8 : 1.0
               sentiment = 'поганий'      negati : neutra =     24.5 : 1.0
               sentiment = 'брак'         negati : neutra =     23.1 : 1.0
               sentiment = 'добро'        positi : neutra =     22.5 : 1.0
               sentiment = 'невеликий'    negati : neutra =     14.7 : 1.0
               sentiment = 'недолік'      negati : neutra =     10.5 : 1.0
               sentiment = 'напруга'      negati : neutra =     10.5 : 1.0
               sentiment = 'низький'      negati : neutra =     10.5 : 1.0
               sentiment = 'справді'      positi : neutra =      6.4 : 1.0
               sentiment = 'кредит'       negati : positi =      6.4 : 1.0
               sentiment = 'ідеальний'    positi : neutra =      4.6 : 1.0
               sentiment = None           neutra : positi =      4.1 : 1.0
               sentiment = 'слабкий'      negati : positi =      3.8 : 1.0
               sentiment = 'високий'      positi : negati =      3.2 : 1.0
               sentiment = 'дивний'       negati : positi =      2.3 : 1.0
               sentiment = 'повільний'    negati : positi =      2.3 : 1.0
                   stars = None           neutra : positi =      2.1 : 1.0
'''