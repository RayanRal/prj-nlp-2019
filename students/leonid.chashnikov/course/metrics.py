from random import randint
from statistics import mean

from textstat.textstat import textstatistics, easy_word_set, legacy_round, textstat
import sys
import io
import os

import re

from os import listdir
from os.path import isfile, join

input_path = './raw_data/'
all_data_sents = []

files = [join(input_path, f) for f in listdir(input_path) if isfile(join(input_path, f))]
for filename in files:
    if filename.endswith('.txt'):
        with open(filename, encoding='utf-8', errors='ignore') as f:
            text = f.read().lower()
            text = text.replace('\n\n', '\n') \
                .replace('"', '') \
                .replace("\"", '') \
                .replace("...", "") \
                .replace("…", "") \
                .replace("—", "") \
                .replace(". ", " . ") \
                .replace(", ", " , ")
            # text = ' '.join(text.split("\n"))
            all_data_sents.append(text)

all_data_string = ' '.join(all_data_sents)
print(len(all_data_string))

splitted_data = all_data_string.split('\n')


flesh_score = []
gf_score = []

for x in range(10):
    index = randint(0, len(splitted_data))
    paragraph = all_data_string.split('\n')[index]
    flesh_score.append(textstat.flesch_reading_ease(paragraph))
    gf_score.append(textstat.gunning_fog(paragraph))

print('Mean Flesh {}'.format(mean(flesh_score)))
print('Mean GF {}'.format(mean(gf_score)))

