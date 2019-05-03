import gzip
from typing import List
from zipfile import ZipFile

import html2text
import jsonlines
import spacy
from readability import Document
from tqdm.auto import tqdm

_h = html2text.HTML2Text()
_h.ignore_links = True
_h.body_width = 100000
print("Load spacy...")
_nlp = spacy.load("en_core_web_sm")
_nlp.remove_pipe('ner')
_nlp.remove_pipe('tagger')
print("...Done!")


def article2sentences(article: str) -> List[List[str]]:
    doc = _nlp(article)
    res = []
    for sent in doc.sents:
        res.append([str(t) for t in sent if str(t) != "\n\n"])
    res = [s for s in res if s[-1] == '.']
    return res


def refine_article(corpus: ZipFile, a_name: str) -> str:
    with corpus.open(a_name) as gz:
        gz_file = gzip.GzipFile(fileobj=gz)
        decompressed = gz_file.read().decode(encoding='utf-8')
    doc = Document(decompressed)
    res: str = _h.handle(doc.summary())
    res = res.replace("Continue reading the main story", '').replace("''", '"').strip()
    return res


def store_sentences_as_jsonl(filename: str, data: List[List[str]]):
    with jsonlines.open(filename, mode='w') as writer:
        for row in data:
            writer.write(row)


def main():
    filename = 'data/NY-Times-2000.zip'
    with ZipFile(filename) as zip_corpus:
        all_names = sorted([f for f in zip_corpus.namelist() if f.endswith('.gz')])
        all_sents = []
        for name in tqdm(all_names, total=len(all_names)):
            art = refine_article(zip_corpus, name)
            sents = article2sentences(art)
            all_sents += sents
    store_sentences_as_jsonl('data/nyt2000-sents.jsonl', all_sents)

    print(f'Total proper sentences: {len(all_sents)}')
    print(f'Total articles: {len(all_names)}')


if __name__ == "__main__":
    main()
