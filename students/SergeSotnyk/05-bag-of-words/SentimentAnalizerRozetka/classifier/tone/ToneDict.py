from collections.abc import Mapping
from typing import Optional, Dict, Iterator
import os

import wget


class ToneDict(Mapping):
    _tone_dict: Optional[Dict[str, float]] = None

    def __init__(self):
        if not ToneDict._tone_dict:
            self._init_tone_dict()

    def _init_tone_dict(self):
        if self._tone_dict:
            return
        if not os.path.isfile('tone-dict-uk.tsv'):
            wget.download(url='https://github.com/lang-uk/tone-dict-uk/raw/master/tone-dict-uk.tsv')
        tone_dict = {}
        with open('tone-dict-uk.tsv', 'tr', encoding='utf-8') as tsv:
            for line in tsv:
                parts = line.split('\t')
                word = parts[0].casefold()
                weight = float(parts[1])
                tone_dict[word] = weight
        ToneDict._tone_dict = tone_dict

    def __getitem__(self, k: str) -> float:
        return self._tone_dict.get(k, 0.0)

    def __len__(self) -> int:
        return len(self._tone_dict)

    def __iter__(self) -> Iterator[str]:
        return iter(self._tone_dict)
