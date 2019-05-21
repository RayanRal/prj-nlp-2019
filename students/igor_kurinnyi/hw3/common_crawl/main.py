import re
import gzip
import json
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

from tqdm import tqdm
from pprint import pprint

import langdetect
from nltk import ngrams

warc_names_map = {
    'WARC-Type': 'warc_type',
    'WARC-Target-URI': 'uri',
    'WARC-Date': 'date',
    'WARC-Record-ID': 'record_id',
    'WARC-Refers-To': 'refers_to',
    'WARC-Block-Digest': 'block_digest',
    'Content-Type': 'content_type',
    'Content-Length': 'content_length'
}


class lazyproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value


@dataclass
class WarcHeader:
    warc_type: str
    uri: str
    date: str
    record_id: str
    refers_to: str
    block_digest: str
    content_type: str
    content_length: str

    def __str__(self):
        return '\n'.join([f'{k.upper()}: {v}' for k, v in asdict(self).items()])


class Document:

    def __init__(self, header, text):
        self.header = header
        self.text = text

    @lazyproperty
    def tokens(self):
        return [x.lower() for x in re.split(r'\W+', self.text)]

    @lazyproperty
    def lang(self):
        try:
            return langdetect.detect(self.text)
        except:
            return None

    def count_ngrams(self, n=1):
        if self.lang in ['zh-cn', 'zh-tw', 'ja', None]:
            return Counter()
        else:
            grams = ('_'.join(x) for x in ngrams(self.tokens, n=n))
            return Counter(grams)

    def __str__(self):
        return f'{self.header}\n{self.text}'


class WetReader:

    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for header, text in self._read_file():
            yield Document(header=WarcHeader(**header), text=text)

    def _read_file(self):
        in_header, in_text = False, False
        header, text = dict(), list()

        with gzip.open(filename=self.filename, mode='rt') as f:
            for _ in range(18):
                next(f)

            for line in f:
                line = line.strip()

                if in_text and line:
                    text.append(line)

                if in_header:
                    h, value = line.split(': ')
                    header[warc_names_map[h]] = value
                    if h == 'Content-Length':
                        in_header, in_text = False, True
                        text = list()

                if line == 'WARC/1.0':
                    in_header, in_text = True, False
                    if header and text:
                        yield header, ' '.join(text)
                    header = dict()


def count_stats(data_dir, wet_file):
    langs_counts = defaultdict(int)
    grams_counts = [defaultdict(Counter) for _ in range(3)]

    wet = WetReader(filename=f'{data_dir}{wet_file}')
    for doc in tqdm(wet):
        langs_counts[doc.lang] += 1
        for n in range(len(grams_counts)):
            grams_counts[n][doc.lang].update(**doc.count_ngrams(n=n+1))

    for n in range(len(grams_counts)):
        grams_counts[n] = take_top_grams(grams_counts[n], n=10)

    return langs_counts, grams_counts


def take_top_grams(grams, n=10):
    result = dict()
    for lang, grams_counts in grams.items():
        result[lang] = grams_counts.most_common(n=n)
    return result


def dump_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


if __name__ == '__main__':
    data_dir = '../../data/common-crawl/'
    wet_file = 'CC-MAIN-20190216004609-20190216030609-00286.warc.wet.gz'

    langs_counts, grams_counts = count_stats(data_dir, wet_file)

    dump_json('langs_counts.json', langs_counts)
    for n in range(len(grams_counts)):
        dump_json(f'{n+1}_grams_counts.json', grams_counts[n])

