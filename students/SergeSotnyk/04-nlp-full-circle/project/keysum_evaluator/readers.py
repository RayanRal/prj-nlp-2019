import os
from collections import Sequence
from os import path
from typing import TextIO, List, TYPE_CHECKING

from keysum_evaluator import DocumentForEval

TAG_DOCUMENT_ID: str = "@id"
TAG_KEYWORDS: str = "@keywords"
TAG_LANG: str = "@lang"
TAG_SUMMARY: str = "@summary"
TAG_SUMMARY2: str = "@digest"

if TYPE_CHECKING:
    DocumentsSeq = Sequence[DocumentForEval]
else:
    DocumentsSeq = Sequence


def _read_keywords(f: TextIO)->List[str]:
    res = []
    for line in _read_summary(f):
        for kw in line.split(',;'):
            res.append(line.strip())
    return res


def _read_summary(f: TextIO)->List[str]:
    res = []
    for l in f:
        ls = l.strip()
        if ls == '':
            break
        res.append(ls)
    return res


def _read_lang(f: TextIO)->str:
    lang = next(f, 'en')
    return lang


def read_doc(dir_name: str, f_name: str) -> DocumentForEval:
    with open(path.join(dir_name, f_name), 'rt', encoding='utf-8') as f:
        summary = []
        keywords = []
        lang = 'en'
        while True:
            line: str = next(f, None)
            if line is None:
                break
            line_lower = line.lower()
            if line_lower.startswith(TAG_KEYWORDS):
                keywords = _read_keywords(f)
            elif line_lower.startswith(TAG_SUMMARY) or line_lower.startswith(TAG_SUMMARY2):
                summary = _read_summary(f)
            elif line_lower.startswith(TAG_LANG):
                lang = _read_lang(f)

    return DocumentForEval(keywords=keywords,
                           summary=summary,
                           lang=lang)


def read_parallel_corpus_as_sequence(path_to_references: str,
                                     path_to_new: str,
                                     lang: str = 'en') -> DocumentsSeq:
    """
    Function returns sequence of documents for evaluation. Both path should contains
    same-named list of files with keywords and summaries.
    :param path_to_references: path to ideal (manually created) summaries and keywords
    :param path_to_new: path to automatically created summaries and keywords
    :param lang: corpus language
    :return: sequence of documents to evaluation
    """

    ref_files = os.listdir(path_to_references)
    new_files = os.listdir(path_to_new)
    if set(ref_files) != set(new_files):
        raise ValueError(f"Directory '{path_to_references}' and '{path_to_new}' contains different set of files.")

    for f_name in ref_files:
        doc = read_doc(path_to_new, f_name)
        doc_ref = read_doc(path_to_references, f_name)

        if doc.lang != lang:
            raise ValueError(f"Document {path.join(path_to_new, f_name)} language is not {lang}")
        if doc_ref.lang != lang:
            raise ValueError(f"Document {path.join(path_to_references, f_name)} language is not {lang}")

        doc.ref_summary = doc_ref.summary
        doc.ref_keywords = doc_ref.keywords

        yield doc
