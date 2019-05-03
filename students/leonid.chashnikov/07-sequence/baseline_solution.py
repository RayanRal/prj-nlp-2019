
def evaluate_baseline(sentences):
    guessed_labels = []
    for sentence in sentences:
        guessed_sentence = []
        for index in range(len(sentence) - 1):
            token = sentence[index][0]
            next_token = sentence[index + 1][0]
            if next_token.istitle():
                guessed_sentence.append([token, True])
            else:
                guessed_sentence.append([token, False])
        guessed_sentence.append([next_token, False])
        guessed_labels.append(guessed_sentence)

    true_positive = 0
    false_negative = 0
    false_positive = 0
    for sentence_index in range(len(sentences)):
        for token_index in range(len(sentences[sentence_index])):
            true_label = sentences[sentence_index][token_index][1]
            guessed_label = guessed_labels[sentence_index][token_index][1]

            if true_label and guessed_label:
                true_positive += 1
            elif true_label and not guessed_label:
                false_negative += 1
            elif not true_label and guessed_label:
                false_positive += 1

    # print('Baseline true positive {}'.format(true_positive))
    # print('Baseline false negative {}'.format(false_negative))
    # print('Baseline false positive {}'.format(false_positive))

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * ((precision * recall) / (precision + recall))

    print('Baseline:\n\tf1 {}, precision {}, recall {}'
          .format(round(f1, 4), round(precision, 4), round(recall, 4)))

