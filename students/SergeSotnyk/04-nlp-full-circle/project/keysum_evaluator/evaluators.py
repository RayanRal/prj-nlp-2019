from collections import defaultdict
from typing import Dict, Set, List, Sequence

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sumeval.metrics.bleu import BLEUCalculator
from sumeval.metrics.rouge import RougeCalculator

from .DocumentForEval import DocumentForEval

_wordnet_lemmatizer = WordNetLemmatizer()
_english_stopwords = set(stopwords.words('english'))


def retrive_lemmatized_tokens(lang: str, keywords: List[str]) -> Set[str]:
    if lang.lower() != 'en':
        raise ValueError("Only English language is supported at the moment.")
    res = set()
    for kw in keywords:
        for word in word_tokenize(kw):
            lemma = _wordnet_lemmatizer.lemmatize(word.casefold())
            if lemma in _english_stopwords:
                continue
            res.add(lemma)
    return res


def evaluate_keywords(document: DocumentForEval) -> Dict[str, float]:
    ref_tokens: Set[str] = retrive_lemmatized_tokens(document.lang, document.ref_keywords)
    tokens: Set[str] = retrive_lemmatized_tokens(document.lang, document.keywords)
    tp = len(ref_tokens.intersection(tokens))
    fp = len(tokens) - tp
    fn = len(ref_tokens) - tp
    eps = 1e-6
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    res = {'precision': precision,
           'recall': recall,
           'f1': f1}
    return res


rouge_en = RougeCalculator(stopwords=True, lang="en")
bleu = BLEUCalculator()


def evaluate_summary(document: DocumentForEval) -> Dict[str, float]:
    summary = '\n'.join(document.summary)
    ref_summary = '\n'.join(document.ref_summary)
    if document.lang.lower() != 'en':
        raise ValueError("Only English language is supported at the moment.")

    rouge_1 = rouge_en.rouge_n(
        summary=summary,
        references=ref_summary,
        n=1)

    rouge_2 = rouge_en.rouge_n(
        summary=summary,
        references=ref_summary,
        n=2)

    rouge_3 = rouge_en.rouge_n(
        summary=summary,
        references=ref_summary,
        n=3)

    rouge_4 = rouge_en.rouge_n(
        summary=summary,
        references=ref_summary,
        n=4)

    bleu_score = bleu.bleu(
        summary=summary,
        references=ref_summary)

    res = {'rouge_1': rouge_1,
           'rouge_2': rouge_2,
           'rouge_3': rouge_3,
           'rouge_4': rouge_4,
           'bleu': bleu_score}
    return res


def evaluate_sequence(docs: Sequence[DocumentForEval],
                      process_keywords: bool = True,
                      process_summary: bool = True) -> Dict[str, float]:
    """
    Function evaluates metrics for sequence of documents item-by-item and calc mean for every metric type.
    :param docs: sequence of documents
    :param process_keywords: calculate metrics for keywords if is True
    :param process_summary: calculate metrics for summaries if is True
    :return: dictionary with mean value for every metric
    """

    res = defaultdict(float)
    counter = 0
    for d in docs:
        counter += 1
        if process_keywords:
            metrics = evaluate_keywords(d)
            for k, v in metrics.items():
                res[k] += v
        if process_summary:
            metrics = evaluate_summary(d)
            for k, v in metrics.items():
                res[k] += v
    result = {}
    if counter > 0:
        for k, v in res.items():
            result[k] = v / counter
    return result
