from os import path

from keysum_evaluator import *

__location__ = path.dirname(__file__)


def test_retrive_lemmatized_tokens():
    keywords = ['Technics', 'to be or not to be', 'Kings']
    lemmatized_tokens = retrive_lemmatized_tokens('en', keywords)
    assert len(lemmatized_tokens) == 2
    assert 'king' in lemmatized_tokens


def test_evaluate_keywords():
    doc = DocumentForEval(keywords=['the pens'], ref_keywords=['a Pen'])
    metrics = evaluate_keywords(doc)
    assert {'recall', 'precision', 'f1'}.issubset(metrics.keys())
    assert min(metrics.values()) > 0.999

    doc = DocumentForEval(keywords=['the pens'], ref_keywords=['a pencil'])
    metrics = evaluate_keywords(doc)
    assert max(metrics.values()) < 0.001


def test_evaluate_summary():
    doc = DocumentForEval(
        summary=['Ad sales boost Time Warner profit'],
        ref_summary=['Ad sales boost Time Warner profit', 'Testing started at 16:22'])
    metrics = evaluate_summary(doc)
    assert {'rouge_1', 'rouge_2', 'rouge_3', 'rouge_4', 'bleu'}.issubset(metrics.keys())

