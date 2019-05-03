import pymorphy2
from pymorphy2.tokenizers import simple_word_tokenize


morph = pymorphy2.MorphAnalyzer(lang='uk')


def _read_tone_dict():
    result = {}
    with open('./data/tone-dict-uk.tsv', 'r') as f:
        lines = [line.rstrip('\n') for line in f]
        for l in lines:
            word = l.split('\t')[0]
            tag = int(l.split('\t')[1])
            result[word] = tag
    return result


tone_dict = _read_tone_dict()


def _get_sentence_normal_forms(text):
    normal_sentence = []
    for token in simple_word_tokenize(text):
        word_normal = morph.normal_forms(token)
        normal_sentence.append(word_normal[0])
    return normal_sentence


def _get_sentence_tone(text):
    tones = []
    for word in text:
        tone = tone_dict.get(word)
        if tone:
            tones.append(tone)
    return sum(tones)
