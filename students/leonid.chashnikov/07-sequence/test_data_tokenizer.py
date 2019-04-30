import spacy
from spacy.tokens.doc import Doc


nlp = spacy.load('en_core_web_lg', disable=['textcat', 'ner'])


class WordTokenizer(object):

    def _remove_empty_words(self, words):
        filtered = [w for w in words if w]
        for i in range(len(filtered)):
            w = words[i]
            if not w:
                print('empty token {} in {}'.format(i, list(filtered)))
        return filtered
    """
    Custom Tokenizer
    """
    def __init__(self, vocab=nlp.vocab, tokenizer=None, return_doc=True):
        self.vocab = vocab
        self._word_tokenizer = tokenizer
        self.return_doc = return_doc

    def __call__(self, text):
        if self._word_tokenizer:
            words = self._word_tokenizer.tokenize(text)
        else:
            words = text.split(' ')
        if self.return_doc:
            words = self._remove_empty_words(words)
            spaces = [True] * len(words)
            return Doc(self.vocab, words=words, spaces=spaces)
        else:
            return words