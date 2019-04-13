import math
import random
from collections import defaultdict


def _read_input():
    result = []
    file = './smsspamcollection/SMSSpamCollection.txt'
    with open(file) as inp:
        lines = [line.rstrip('\n') for line in inp]
        for l in lines:
            if l:
                l = l.split('\t')
                text = ''.join(l[1:])
                result.append(dict(label=l[0], text=text))
    return result


def _split_data(input_data, split):
    test = []
    train = []
    for i in input_data:
        separator = random.random()
        if separator > split:
            test.append(i)
        else:
            train.append(i)
    return test, train


class NaiveBayes:

    spam_dict = dict()
    ham_dict = dict()

    prob_spam = math.log(0.13)
    prob_ham = math.log(0.87)

    def _add_to_dict(self, sentence, label):
        if label == 'spam':
            for word in sentence.split():
                w = word.lower()
                if w in self.spam_dict:
                    self.spam_dict[w] += 1
                else:
                    self.spam_dict[w] = 1
        else:
            for word in sentence.split():
                w = word.lower()
                if w in self.ham_dict:
                    self.ham_dict[w] += 1
                else:
                    self.ham_dict[w] = 1

    def train(self, input_data):
        for i in input_data:
            self._add_to_dict(i['text'], i['label'])

        # self.prob_spam = len(self.spam_dict) / (len(self.spam_dict) + len(self.ham_dict))
        # self.prob_ham = len(self.ham_dict) / (len(self.spam_dict) + len(self.ham_dict))

    def _check_dict(self):
        print('spam {}'.format(len(self.spam_dict)))
        print('ham {}'.format(len(self.ham_dict)))

    def _get_probability(self, word, class_dict):
        value_in_class = class_dict.get(word, math.pow(1, -10))
        all_dict_vals = sum(class_dict.values())
        return value_in_class / all_dict_vals

    def predict(self, sentence):
        prob_spam_full = []
        prob_ham_full = []
        for s in sentence.split():
            word = s.lower()
            prob_f_spam = self._get_probability(word, self.spam_dict)
            prob_s_spam = math.log(prob_f_spam)
            prob_spam_full.append(prob_s_spam)

            prob_f_ham = self._get_probability(word, self.ham_dict)
            prob_s_ham = math.log(prob_f_ham)
            prob_ham_full.append(prob_s_ham)

        prob_spam = sum(prob_spam_full)
        prob_ham = sum(prob_ham_full)

        if prob_spam < prob_ham:
            return 'ham'
        else:
            return 'spam'

    def _weighted_random_by_dct(self, dict):
        rand_val = random.random()
        total = 0
        for k, v in dict.items():
            total += v
            if rand_val <= total:
                return k

    def generate(self, num_words, label):
        class_dict = self.spam_dict if label == 'spam' else self.ham_dict
        result = []

        for i in range(num_words):
            result.append(self._weighted_random_by_dct(class_dict))
        return result


if __name__ == "__main__":
    input_data = _read_input()

    train_data, test_data = _split_data(input_data, 0.7)

    nb = NaiveBayes()
    nb.train(train_data)

    correct = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for i in test_data:
        predicted_label = nb.predict(i['text'])
        real_label = i['label']
        if predicted_label == real_label:
            correct += 1
        if predicted_label == 'spam' and real_label == 'spam':
            true_positive += 1
        if predicted_label == 'spam' and real_label == 'ham':
            false_positive += 1
        if predicted_label == 'ham' and real_label == 'spam':
            false_negative += 1
        if predicted_label == 'ham' and real_label == 'ham':
            true_negative += 1

    precision = true_positive / float(true_positive + false_positive)
    recall = true_positive / float(true_positive + false_negative)
    accuracy = (true_positive + true_negative) / float(len(test_data))

    print('Checked values: {}'.format(len(test_data)))
    print('Predicted correctly {}'.format(correct))
    print('Precision {}'.format(precision))
    print('Recall {}'.format(recall))
    print('Accuracy {}'.format(accuracy))

    print(nb.generate(7, 'ham'))

    # feature extraction / normalization:
    #   probably no lemmatization
    #   maybe add length as separate feature
    #   maybe add "has phone number" feature
    #   maybe add "has NER"
    #   n-grams - probably no, little data

    # main features:
    #   bof
    #   tf-idf - try next
    # word_count = _word_count(input_data)

    # DEBUG
    # print(len(word_count))
    # print(word_count['u'])
    # print(word_count['free'])

    # model:
    #   naive bayes







