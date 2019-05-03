from typing import List, Optional, Set, Dict

import pymorphy2
from nltk import word_tokenize

uk_morph = pymorphy2.MorphAnalyzer(lang='uk')


class UkLemmatizer():
    _cache: Dict[str, str] = {}

    @staticmethod
    def lemmatize(word: str) -> str:
        res = UkLemmatizer._cache.get(word, None)
        if not res:
            res = uk_morph.parse(word)[0].normal_form
            UkLemmatizer._cache[word] = res
        return res


def tokenize_lemmatize(text: str, stop_words: Optional[Set[str]] = None) -> List[str]:
    tokens = word_tokenize(text)
    if stop_words:
        tokens = [t for t in tokens if t not in stop_words]
    lemmas = [UkLemmatizer.lemmatize(t) for t in tokens]
    if stop_words:
        lemmas = [t for t in lemmas if t not in stop_words]
    return lemmas
